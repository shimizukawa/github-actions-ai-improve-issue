"""Issue自動改善スクリプト - Phase 2実装（設定ファイル対応版）

PEP-723対応: uvx で実行可能

実行モード:
1. 通常モード: GitHub Actionsから自動実行（Issue作成時）
2. --dry-run: ローカル検証用（コメント投稿をスキップ、読み取り操作は実行）
3. --index-issues: RAGデータ生成モード（全Issueをベクトル化）
4. --update-single-issue: 単一Issue更新モード
"""

import argparse
import dataclasses
import json
import os
import re
import sys
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import yaml
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold
import voyageai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)


# ==================== 設定管理 ====================


@dataclasses.dataclass
class Config:
    """環境変数の一元管理クラス"""

    # GitHub関連（GitHub Actions実行時に自動設定）
    github_repository: str = dataclasses.field(
        default_factory=lambda: os.environ.get("GITHUB_REPOSITORY", "")
    )
    github_token: str = dataclasses.field(
        default_factory=lambda: os.environ.get("GITHUB_TOKEN", "")
    )

    # Issue情報（通常モード実行時に必要）
    issue_body: str = dataclasses.field(
        default_factory=lambda: os.environ.get("ISSUE_BODY", "")
    )
    issue_title: str = dataclasses.field(
        default_factory=lambda: os.environ.get("ISSUE_TITLE", "")
    )
    issue_number: str = dataclasses.field(
        default_factory=lambda: os.environ.get("ISSUE_NUMBER", "")
    )

    # LLM API（通常モード実行時に必須）
    gemini_api_key: str = dataclasses.field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )

    # RAG機能（オプション - 全て設定された場合のみ有効化）
    qdrant_url: str = dataclasses.field(
        default_factory=lambda: os.environ.get("QDRANT_URL", "")
    )
    qdrant_api_key: str = dataclasses.field(
        default_factory=lambda: os.environ.get("QDRANT_API_KEY", "")
    )
    voyage_api_key: str = dataclasses.field(
        default_factory=lambda: os.environ.get("VOYAGE_API_KEY", "")
    )
    qdrant_collection_name: str = dataclasses.field(
        default_factory=lambda: os.environ.get(
            "QDRANT_COLLECTION_NAME", "ai-improve-issues"
        )
    )

    @property
    def is_rag_enabled(self) -> bool:
        """RAG機能が有効かどうか"""
        return bool(self.qdrant_url and self.qdrant_api_key and self.voyage_api_key)

    def validate_for_normal_mode(self):
        """通常モード実行時の必須環境変数チェック"""
        if not self.issue_number:
            raise ValueError("Error: ISSUE_NUMBER not set")
        if not self.gemini_api_key:
            raise ValueError("Error: GEMINI_API_KEY not set")

    def validate_for_github_operations(self):
        """GitHub操作が必要な場合の環境変数チェック"""
        if not self.github_token:
            raise ValueError("Error: GITHUB_TOKEN not set")
        if not self.github_repository:
            raise ValueError("Error: GITHUB_REPOSITORY not set")

    def validate_for_rag_operations(self):
        """RAG操作が必要な場合の環境変数チェック"""
        if not self.voyage_api_key:
            raise ValueError("Error: VOYAGE_API_KEY not set")
        if not self.qdrant_url:
            raise ValueError("Error: QDRANT_URL not set")
        if not self.qdrant_api_key:
            raise ValueError("Error: QDRANT_API_KEY not set")


# 設定を読み込み
config = Config()


# ==================== ユーティリティ ====================


def find_repo_root() -> Path:
    """リポジトリルートを探索

    Returns:
        Path: リポジトリルートディレクトリ
    """
    # カレントディレクトリから.gitディレクトリを探す
    # (uvx実行時は__file__がsite-packagesを指すため、cwdを起点とする)
    current = Path.cwd().resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent

    # .git が見つからない場合はカレントディレクトリを返す
    return current


# ==================== テンプレート設定 ====================


@dataclasses.dataclass
class TemplateConfig:
    """テンプレート設定"""

    name: str
    issue_template_file: str
    system_prompt: str
    keywords: list[str]


@dataclasses.dataclass
class ImproveIssueSettings:
    """Issue改善設定"""

    templates: dict[str, TemplateConfig]
    default_template: str

    def validate(self):
        """設定の妥当性をチェック"""
        if not self.templates:
            raise ValueError("Error: templates is empty")
        if self.default_template not in self.templates:
            raise ValueError(
                f"Error: default_template '{self.default_template}' not found in templates"
            )


def load_settings() -> ImproveIssueSettings:
    """設定ファイルを読み込む

    Returns:
        ImproveIssueSettings: 設定オブジェクト

    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合
        ValueError: 設定内容が不正な場合
    """
    # 設定ファイルパスの決定
    config_path = os.environ.get("AI_IMPROVE_ISSUE_CONFIG")
    if config_path:
        config_file = Path(config_path)
    else:
        repo_root = find_repo_root()
        config_file = repo_root / ".ai_improve_issue.yml"

    if not config_file.exists():
        raise FileNotFoundError(
            f"設定ファイルが見つかりません: {config_file}\n"
            f"環境変数 AI_IMPROVE_ISSUE_CONFIG で設定ファイルパスを指定するか、\n"
            f"リポジトリルートに .ai_improve_issue.yml を配置してください。"
        )

    # YAML読み込み
    with open(config_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # バリデーション
    if not data:
        raise ValueError("設定ファイルが空です")

    if "templates" not in data:
        raise ValueError("設定ファイルに 'templates' が定義されていません")

    if "default_template" not in data:
        raise ValueError("設定ファイルに 'default_template' が定義されていません")

    # テンプレート設定の構築
    templates = {}
    for name, tmpl_data in data["templates"].items():
        if not isinstance(tmpl_data, dict):
            raise ValueError(f"テンプレート '{name}' の定義が不正です")

        required_fields = ["issue_template_file", "system_prompt", "keywords"]
        for field in required_fields:
            if field not in tmpl_data:
                raise ValueError(
                    f"テンプレート '{name}' に必須フィールド '{field}' がありません"
                )

        templates[name] = TemplateConfig(
            name=name,
            issue_template_file=tmpl_data["issue_template_file"],
            system_prompt=tmpl_data["system_prompt"],
            keywords=tmpl_data["keywords"],
        )

    settings = ImproveIssueSettings(
        templates=templates,
        default_template=data["default_template"],
    )

    settings.validate()

    return settings


# ==================== テンプレート読み込み ====================


def load_template_content(template: TemplateConfig) -> str:
    """ISSUE_TEMPLATEファイルから実際のテンプレート内容を読み込む"""
    repo_root = find_repo_root()
    template_file = (
        repo_root / ".github" / "ISSUE_TEMPLATE" / f"{template.issue_template_file}.md"
    )

    if not template_file.exists():
        raise FileNotFoundError(
            f"Issueテンプレートファイルが見つかりません: {template_file}"
        )

    with open(template_file, encoding="utf-8") as f:
        content = f.read()

    # frontmatter (---で囲まれた部分) を除去
    lines = content.split("\n")
    if lines and lines[0] == "---":
        # 2つ目の---を探す
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i] == "---":
                end_idx = i
                break
        if end_idx is not None:
            content = "\n".join(lines[end_idx + 1 :])

    return content.strip()


# ==================== プロンプトテンプレート ====================


def get_improve_prompt(
    template_name: str,
    issue_body: str,
    issue_title: str = "",
    similar_issues: list[dict[str, Any]] | None = None,
    settings: ImproveIssueSettings | None = None,
) -> tuple[str, str]:
    """テンプレートに応じたプロンプトを取得（RAG対応）

    Args:
        template_name: テンプレート名
        issue_body: Issue本文
        issue_title: Issueタイトル
        similar_issues: 類似Issue情報（RAG検索結果）
        settings: 設定オブジェクト

    Returns:
        system_instruction, prompt:
    """
    if settings is None:
        raise ValueError("settings is required")

    tmpl = settings.templates[template_name]
    template_content = load_template_content(tmpl)
    system_instruction = f"""
{tmpl.system_prompt}

【Issue記述】について、【類似する過去Issue】の内容を参考にして、概要を敬体でまとめてください。
また、【類似する過去Issue】に類似する内容があるか判定し、【## 類似Issue】セクションにそのidを出力してください。
もし存在しない場合は「なし」と出力してください。
出力形式以外の文章は不要です。
"""

    prompt = f"""
【Issue記述】
タイトル: {issue_title}
本文: {issue_body}

【テンプレート】
{template_content}
"""

    # RAG検索結果があれば追加
    if similar_issues and len(similar_issues) > 0:
        similar_info = (
            "\n\n【類似する過去Issue】\n以下の過去Issueを参考にしてください：\n"
        )
        for i, issue in enumerate(similar_issues, 1):
            similar_info += f"""
【参考Issue {i}】
- タイトル: {issue["issue_title"]}
- 本文抜粋: {issue["issue_body"][:200]}...
- 類似度: {issue["similarity"]:.1%}
"""
        similar_info += "\n上記の参考Issueから、記述スタイルや必要な情報項目を学び、より具体的で実用的な例文を生成してください。"
        prompt += similar_info

    return system_instruction, prompt


# ==================== LLMクライアント ====================


class TextProcessAgent:
    """テキスト処理エージェント（テンプレート判定と文章生成）"""

    # 安全性設定を緩和（技術的な内容に対応）
    _SAFETY_SETTINGS = [
        genai.types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        genai.types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        genai.types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        genai.types.SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
    ]

    def __init__(self, llm_api_key: str, model: str = "gemini-2.5-flash"):
        """テキスト処理エージェント（テンプレート判定と文章生成を担当するLLMクライアント）

        Args:
            llm_api_key: APIキー
            model: モデル名（2025年11月時点の推奨）
                - Phase 0: 'gemini-2.0-flash-lite' (検証用、極低コスト)
                - Phase 1-2: 'gemini-2.5-flash' (コスパ良好)
                - Phase 2: 'claude-3.7-sonnet' (品質重視)
        """
        self.client = genai.Client(api_key=llm_api_key)
        self.model_name = model

    def detect_template(
        self, issue_body: str, issue_title: str, settings: ImproveIssueSettings
    ) -> str:
        """Issue本文とタイトルからテンプレートを判定（LLMベース単一選択）

        Args:
            issue_body: Issue本文
            issue_title: Issueタイトル
            settings: 設定オブジェクト

        Returns:
            テンプレート名（失敗時や曖昧時は `default_template`）
        """
        # テンプレート情報を要約してLLMに渡す
        tmpl_summaries: list[dict[str, Any]] = []
        for name, tmpl in settings.templates.items():
            sp = (tmpl.system_prompt or "").strip()
            if len(sp) > 300:
                sp = sp[:300]
            kws = (tmpl.keywords or [])[:10]
            tmpl_summaries.append(
                {
                    "name": name,
                    "keywords": kws,
                    "system_prompt": sp,
                }
            )

        system_instruction = (
            "あなたはIssueの内容から最適なテンプレートを1つだけ選ぶ分類器です。\n"
            "提供されたテンプレート候補の中から、記述の目的・構造に最も適合するものを厳密に1件選んでください。\n"
            "出力はJSONフォーマットデータのみ、追加説明は禁止です。"
        )

        prompt = (
            "【Issue】\n"
            f"タイトル: {issue_title}\n"
            f"本文: {issue_body}\n\n"
            "【テンプレート候補一覧(JSON)】\n"
            f"{json.dumps(tmpl_summaries, ensure_ascii=False)}\n\n"
            "以下の形式で厳密に1件のみ出力してください。\n"
            '{"template": "<name>"}'
        )

        # 低トークン・低温度で分類実行（API呼び出しとtext取得のみtry）
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=[system_instruction],
                max_output_tokens=256,
                temperature=0.1,
                safety_settings=self._SAFETY_SETTINGS,
            ),
        )

        try:
            text = response.text or ""
        except ValueError as e:
            print(
                f"response.text 取得でエラー ({e})。デフォルトテンプレートを使用します。"
            )
            return settings.default_template

        print(f"選択された内容: {text=}")
        # {.*} 部分のみ抽出
        text = re.sub(r"(?s).*?(\{.*?\}).*", r"\1", text)

        try:
            selected: dict[str, str] = json.loads(text)
        except json.JSONDecodeError as e:
            print(
                f"JSONデコードエラー: {e}。テキスト: {text}。デフォルトテンプレートを使用します。"
            )
            return settings.default_template

        name = selected.get("template", "")
        if name not in settings.templates:
            print(
                f"不明なテンプレート名 '{name}'。有効なテンプレート: {list(settings.templates.keys())}。デフォルトテンプレートを使用します。"
            )
            return settings.default_template

        return name

    def generate(
        self, system_instruction: str, prompt: str, max_tokens: int = 5000
    ) -> str:
        """プロンプトから文章を生成

        Returns:
            生成されたテキスト、またはエラーメッセージ
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=[system_instruction],
                max_output_tokens=max_tokens,
                temperature=0.7,
                safety_settings=self._SAFETY_SETTINGS,
            ),
        )

        try:
            return response.text
        except ValueError:
            pass

        # finish_reasonを確認してエラーメッセージを生成
        finish_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason

        if finish_reason == 2:  # SAFETY
            return "⚠️ AIによる生成が安全性フィルターによりブロックされました。Issue内容を確認し、手動で記入してください。"
        elif finish_reason == 3:  # RECITATION
            return "⚠️ AIによる生成が著作権保護によりブロックされました。別の表現で記入してください。"
        elif finish_reason == 4:  # OTHER
            return "⚠️ AIによる生成が制限されました。手動で記入してください。"

        return "⚠️ AIによる生成に失敗しました。手動で記入してください。"


# ==================== RAGクライアント (Phase 2) ====================


class VoyageEmbeddingClient:
    """Voyage AI Embeddingクライアント"""

    def __init__(self, api_key: str, model: str = "voyage-3.5-lite"):
        """
        Args:
            api_key: Voyage AI APIキー
            model: モデル名（デフォルト: voyage-3.5-lite）
        """
        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    def generate_embedding(self, text: str, dimensions: int = 256) -> list[float]:
        """テキストのEmbeddingを生成

        Args:
            text: 入力テキスト
            dimensions: 出力次元数（256, 512, 1024）

        Returns:
            Embeddingベクトル
        """
        result = self.client.embed(
            texts=[text], model=self.model, output_dimension=dimensions
        )
        return result.embeddings[0]


class QdrantSearchClient:
    """Qdrant検索クライアント"""

    def __init__(
        self, url: str, api_key: str, collection_name: str = "ai-improve-issues"
    ):
        """Qdrantクライアント初期化

        Args:
            url: Qdrant CloudのURL
            api_key: Qdrant APIキー
            collection_name: コレクション名（環境変数で上書き可能）
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    def ensure_collection(self, vector_size: int = 256):
        """コレクションが存在することを確認、なければ作成"""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Collection '{self.collection_name}' created")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="issue_number",
                field_schema=PayloadSchemaType.INTEGER,
            )

    def search_similar_issues(
        self,
        query_vector: list[float],
        limit: int = 3,
        exclude_issue_number: int | None = None,
    ) -> list[dict[str, Any]]:
        """類似Issue検索（チャンク対応）

        Args:
            query_vector: クエリベクトル
            limit: 取得件数（Top-K）- Issue数（チャンク数ではない）

        Returns:
            類似Issue情報のリスト
        """
        # より多くのチャンクを取得してIssueごとに集約
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit * 5,  # 余裕を持って取得
        )

        points = getattr(response, "points", [])
        if not points:
            return []

        # Issueごとに最高スコアのチャンクを集約
        issue_map = {}
        for result in points:
            issue_num = result.payload.get("issue_number")
            # 除外対象のIssue番号をスキップ
            if exclude_issue_number is not None and issue_num == exclude_issue_number:
                continue
            if (
                issue_num not in issue_map
                or result.score > issue_map[issue_num]["similarity"]
            ):
                # チャンクまたは全文を取得
                issue_body = result.payload.get(
                    "issue_body_chunk"
                ) or result.payload.get("issue_body", "")

                issue_map[issue_num] = {
                    "issue_number": issue_num,
                    "issue_title": result.payload.get("issue_title", ""),
                    "issue_body": issue_body[:500],
                    "state": result.payload.get("state", ""),
                    "url": result.payload.get("url", ""),
                    "similarity": result.score,
                }

        # スコア順でソートして上位limit件を返す
        similar_issues = sorted(
            issue_map.values(), key=lambda x: x["similarity"], reverse=True
        )[:limit]

        return similar_issues

    def upsert_issue_chunks(
        self,
        issue_number: int,
        chunks: list[str],
        vectors: list[list[float]],
        title: str,
        state: str,
        url: str,
        labels: list[str],
    ):
        """Issueをチャンク分割してインデックスに登録または更新

        Args:
            issue_number: Issue番号
            chunks: Issue本文のチャンクリスト
            vectors: 各チャンクのEmbeddingベクトルリスト
            title: Issueタイトル
            state: Issueステート（open/closed）
            url: IssueのURL
            labels: ラベルリスト
        """
        # 既存のチャンクを削除（issue_numberで始まるIDを検索して削除）
        ids_to_delete: list[str] = []
        offset: dict | None = None
        while True:
            existing_points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="issue_number",
                            match=MatchValue(value=issue_number),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )

            if not existing_points:
                break

            ids_to_delete.extend(str(point.id) for point in existing_points)

            if next_offset is None:
                break

            offset = next_offset

        if ids_to_delete:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids_to_delete),
            )

        # 新しいチャンクを登録
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "issue_number": issue_number,
                    "chunk_index": i,
                    "issue_title": title,
                    "issue_body_chunk": chunk,
                    "state": state,
                    "url": url,
                    "labels": labels,
                },
            )
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Issue #{issue_number} indexed with {len(chunks)} chunks")


# ==================== GitHub API ====================


def fetch_issue_from_github(issue_number: int) -> dict | None:
    """GitHub APIからIssue情報を取得

    Args:
        issue_number: Issue番号

    Returns:
        Issue情報の辞書、取得失敗時はNone
    """
    if not config.github_repository:
        print("Error: GITHUB_REPOSITORY not set")
        return None

    cmd = ["gh", "api", f"/repos/{config.github_repository}/issues/{issue_number}"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        env={
            "GH_TOKEN": config.github_token,
            "GH_REPO": config.github_repository,
        },
    )
    issue_data = json.loads(result.stdout)
    labels = [label["name"] for label in issue_data["labels"]]
    return {
        "number": int(issue_data["number"]),
        "title": issue_data["title"],
        "body": issue_data["body"],
        "state": issue_data["state"],
        "url": issue_data["html_url"],
        "labels": labels,
    }


def fetch_all_issues(start: int = 1, end: int | None = None) -> list[dict]:
    """全Issue情報を取得

    Args:
        start: 開始Issue番号
        end: 終了Issue番号（Noneの場合は全て）

    Returns:
        Issue情報のリスト
    """
    if not config.github_repository:
        print("Error: GITHUB_REPOSITORY not set")
        return []

    # gh issue list でIssue番号一覧を取得
    cmd = [
        "gh",
        "issue",
        "list",
        "--state",
        "all",
        "--limit",
        "1000",
        "--json",
        "number",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        cwd=os.getcwd(),
        env={
            "GH_TOKEN": config.github_token,
            "GH_REPO": config.github_repository,
        },
    )
    issues_data = json.loads(result.stdout)
    issue_numbers = [issue["number"] for issue in issues_data]

    # 範囲フィルタリング
    if end is not None:
        issue_numbers = [n for n in issue_numbers if start <= n <= end]
    else:
        issue_numbers = [n for n in issue_numbers if n >= start]

    # 各Issueの詳細を取得
    issues = []
    for num in issue_numbers:
        issue = fetch_issue_from_github(num)
        if issue:
            issues.append(issue)

    return issues


# ==================== チャンク処理 ====================


def create_issue_chunks(issue_title: str, issue_body: str) -> list[str]:
    """Issue本文をチャンク分割

    Args:
        issue_title: Issueタイトル
        issue_body: Issue本文

    Returns:
        チャンクリスト
    """
    # タイトルと本文を結合
    full_text = f"{issue_title}\n\n{issue_body}"

    # 短い場合はチャンク分割不要
    if len(full_text) <= 400:
        return [full_text]

    # LangChainのRecursiveCharacterTextSplitterを使用
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["。", "\n\n", "\n", " ", ""],  # 日本語優先の区切り文字
    )

    chunks = splitter.split_text(full_text)
    return chunks


def create_embeddings_for_chunks(
    chunks: list[str], embedding_client: "VoyageEmbeddingClient", dimensions: int = 256
) -> list[list[float]]:
    """チャンクリストのEmbeddingを生成

    Args:
        chunks: チャンクリスト
        embedding_client: Embeddingクライアント
        dimensions: Embedding次元数

    Returns:
        Embeddingベクトルリスト
    """
    # Batch embed all chunks at once
    result = embedding_client.client.embed(
        texts=chunks, model=embedding_client.model, output_dimension=dimensions
    )
    return result.embeddings


# ==================== メイン処理 ====================


def check_needs_improvement(issue_body: str, issue_title: str) -> bool:
    """Issue改善が必要かチェック

    Args:
        issue_body: Issue本文
        issue_title: Issueタイトル

    Returns:
        True: 改善が必要, False: 改善不要
    """
    # タイトルと本文を結合し、空白を除去して文字数をカウント
    combined = (issue_title or "") + (issue_body or "")
    text_without_spaces = combined.replace(" ", "").replace("\n", "").replace("\t", "")

    # 10文字未満の場合は改善不要
    if len(text_without_spaces) < 10:
        return False

    return True


def post_comment_via_gh(issue_number: str, content: str) -> None:
    """GitHub CLI経由でコメントを投稿

    Args:
        issue_number: Issue番号
        content: コメント内容
    """
    # 一時ファイルに本文を書き出し
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".md") as f:
        f.write(content)
        f.flush()
        # GitHub CLIでコメント投稿
        subprocess.run(
            ["gh", "issue", "comment", issue_number, "--body-file", f.name],
            check=True,
        )
    print(f"Comment posted successfully to issue #{issue_number}")


def generate_improved_content(
    issue_body: str,
    issue_title: str,
    llm_api_key: str,
    similar_issues: list[dict[str, Any]] | None = None,
    settings: ImproveIssueSettings | None = None,
) -> tuple[str, str]:
    """Issue内容を改善した例文を生成（RAG対応）

    Args:
        issue_body: Issue本文
        issue_title: Issueタイトル
        llm_api_key: LLM APIキー
        similar_issues: 類似Issue情報（RAG検索結果）
        settings: 設定オブジェクト

    Returns:
        (improved_content, template_name): 改善された内容とテンプレート名
    """
    if settings is None:
        raise ValueError("settings is required")

    # LLMクライアント初期化
    client = TextProcessAgent(llm_api_key=llm_api_key)

    # テンプレート判定
    template_name = client.detect_template(issue_body, issue_title, settings)
    print(f"Detected template: {template_name}")

    # 文章生成
    system_instruction, prompt = get_improve_prompt(
        template_name, issue_body, issue_title, similar_issues, settings
    )
    improved_content = client.generate(system_instruction, prompt)
    print("Content generated successfully")

    return improved_content, template_name


def format_comment(
    improved_content: str,
    template_name: str,
    similar_issues: list[dict[str, Any]] | None = None,
) -> str:
    """コメント用のフォーマット済み文字列を生成（RAG対応）

    Args:
        improved_content: 改善された内容
        template_name: テンプレート名
        similar_issues: 類似Issue情報（RAG検索結果）

    Returns:
        フォーマット済みのコメント文字列
    """
    template_display_names = {
        "feature_request": "機能要件",
        "bug_report": "バグ報告",
    }
    template_display = template_display_names.get(template_name, template_name)

    comment = f"""## 🤖 AIによるIssue記入例

**選定テンプレート**: {template_display}

---

{improved_content}

---
"""

    # RAG検索結果があれば追加
    if similar_issues and len(similar_issues) > 0:
        comment += "\n### 📚 参考にした類似Issue\n\nこの例文は以下の過去Issueを参考に生成しています：\n\n"
        for i, issue in enumerate(similar_issues, 1):
            comment += f"""{i}. **#{issue["issue_number"]}: {issue["issue_title"]}** ({issue["state"]})
   - 類似度: {issue["similarity"]:.0%}
   - {issue["url"]}

"""
        comment += "---\n\n"

    comment += """💡 **使い方**: 上記の例文を参考に、Issue本文を編集してください。"""
    if similar_issues and len(similar_issues) > 0:
        comment += "類似Issueも確認すると、より詳細な情報が得られます。"
    else:
        comment += "実際のプロジェクトに合わせて内容を修正してください。"

    comment += "\n\n<!-- AI-generated comment -->\n"

    return comment


def index_all_issues(
    start: int = 1, end: int | None = None, settings: ImproveIssueSettings | None = None
):
    """全Issueをインデックス登録（--index-issues モード）

    Args:
        start: 開始Issue番号
        end: 終了Issue番号（Noneの場合は全て）
        settings: 設定オブジェクト
    """
    if settings is None:
        raise ValueError("settings is required")

    config.validate_for_github_operations()
    config.validate_for_rag_operations()

    print("=== RAG Indexing Mode ===")
    print("Fetching issues from GitHub...")

    # Issue一覧取得
    issues = fetch_all_issues(start, end)
    if not issues:
        print("No issues found")
        sys.exit(0)

    print(f"Found {len(issues)} issues to index")

    # クライアント初期化
    voyage_client = VoyageEmbeddingClient(api_key=config.voyage_api_key)
    qdrant_client = QdrantSearchClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=config.qdrant_collection_name,
    )
    qdrant_client.ensure_collection(vector_size=256)

    # 各Issueをインデックス登録
    success_count = 0
    for i, issue in enumerate(issues, 1):
        print(f"[{i}/{len(issues)}] Indexing issue #{issue['number']}...")

        # チャンク分割
        chunks = create_issue_chunks(issue["title"], issue["body"])

        # 各チャンクのEmbeddingベクトル生成
        vectors = create_embeddings_for_chunks(chunks, voyage_client, dimensions=256)

        # Qdrantに登録
        qdrant_client.upsert_issue_chunks(
            issue_number=issue["number"],
            chunks=chunks,
            vectors=vectors,
            title=issue["title"],
            state=issue["state"],
            url=issue["url"],
            labels=issue.get("labels", []),
        )
        success_count += 1

    print("\n=== Indexing Complete ===")
    print(f"Success: {success_count}/{len(issues)} issues")


def update_single_issue(
    issue_number: int, settings: ImproveIssueSettings | None = None
):
    """単一Issueをインデックス更新（--update-single-issue モード）

    Args:
        issue_number: Issue番号
        settings: 設定オブジェクト
    """
    if settings is None:
        raise ValueError("settings is required")

    config.validate_for_github_operations()
    config.validate_for_rag_operations()

    print(f"=== Update Single Issue #{issue_number} ===")

    # Issue情報取得
    issue = fetch_issue_from_github(issue_number)
    if not issue:
        print(f"Error: Failed to fetch issue #{issue_number}")
        sys.exit(1)

    # クライアント初期化
    voyage_client = VoyageEmbeddingClient(api_key=config.voyage_api_key)
    qdrant_client = QdrantSearchClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=config.qdrant_collection_name,
    )
    qdrant_client.ensure_collection(vector_size=256)

    # チャンク分割
    chunks = create_issue_chunks(issue["title"], issue["body"])

    # 各チャンクのEmbeddingベクトル生成
    vectors = create_embeddings_for_chunks(chunks, voyage_client, dimensions=256)

    # Qdrantに登録
    qdrant_client.upsert_issue_chunks(
        issue_number=issue["number"],
        chunks=chunks,
        vectors=vectors,
        title=issue["title"],
        state=issue["state"],
        url=issue["url"],
        labels=issue.get("labels", []),
    )

    print(f"Issue #{issue_number} updated successfully")


def main():
    # 設定ファイルを読み込み
    try:
        settings = load_settings()
        print("設定ファイルを読み込みました")
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        sys.exit(1)

    # 引数解析
    parser = argparse.ArgumentParser(description="Issue自動改善スクリプト (Phase 2)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ローカル検証用（コメント投稿をスキップ）",
    )
    parser.add_argument(
        "--index-issues",
        action="store_true",
        help="RAGデータ生成モード（全Issueをベクトル化）",
    )
    parser.add_argument(
        "--update-single-issue",
        type=int,
        help="単一Issue更新モード（指定したIssue番号を更新）",
    )
    parser.add_argument(
        "--start", type=int, default=1, help="RAGインデックス開始Issue番号"
    )
    parser.add_argument("--end", type=int, help="RAGインデックス終了Issue番号")
    args = parser.parse_args()

    # RAGデータ生成モード
    if args.index_issues:
        index_all_issues(start=args.start, end=args.end, settings=settings)
        sys.exit(0)

    # 単一Issue更新モード
    if args.update_single_issue:
        update_single_issue(args.update_single_issue, settings=settings)
        sys.exit(0)

    # 通常モード: Issue改善
    # 必須環境変数チェック
    try:
        config.validate_for_normal_mode()
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    # 改善が必要かチェック
    if not check_needs_improvement(config.issue_body, config.issue_title):
        print(f"Issue #{config.issue_number} does not need improvement (too short)")
        sys.exit(0)

    print(f"Processing issue #{config.issue_number}")
    print(f"Title: {config.issue_title}")
    print(f"Body length: {len(config.issue_body)} characters")

    # RAG機能チェック
    similar_issues = None

    if config.is_rag_enabled:
        print("RAG mode: Enabled")
        # RAG検索
        voyage_client = VoyageEmbeddingClient(api_key=config.voyage_api_key)
        qdrant_client = QdrantSearchClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=config.qdrant_collection_name,
        )
        qdrant_client.ensure_collection(vector_size=256)

        # クエリベクトル生成
        query_text = f"{config.issue_title}\n{config.issue_body}"
        query_vector = voyage_client.generate_embedding(query_text, dimensions=256)

        # 類似Issue検索（自分自身を除外）
        similar_issues = qdrant_client.search_similar_issues(
            query_vector, limit=3, exclude_issue_number=int(config.issue_number)
        )

        if similar_issues:
            print(f"Found {len(similar_issues)} similar issues")
            for i, sim in enumerate(similar_issues, 1):
                print(
                    f"  {i}. #{sim['issue_number']}: {sim['issue_title'][:50]}... "
                    f"(similarity: {sim['similarity']:.1%})"
                )
        else:
            print("No similar issues found")
    else:
        print("RAG mode: Disabled")

    # 改善内容を生成
    improved_content, template_name = generate_improved_content(
        config.issue_body,
        config.issue_title,
        config.gemini_api_key,
        similar_issues,
        settings,
    )

    # コメント用にフォーマット
    output = format_comment(improved_content, template_name, similar_issues)

    # --dry-run モードではコンソール出力のみ
    if args.dry_run:
        print("\n" + "=" * 60)
        print("[DRY RUN] コメント投稿をスキップします")
        print("=" * 60)
        print(output)
        print("=" * 60)
        sys.exit(0)

    # 通常モード: GitHub CLIでコメント投稿
    if not config.github_token:
        print("Error: GITHUB_TOKEN not found")
        sys.exit(1)

    post_comment_via_gh(config.issue_number, output)

    # RAGインデックス登録（例文生成後）
    if not config.is_rag_enabled:
        print("QDRANT_* and VOYAGE_* env values are required to enable RAG mode.")
        sys.exit(0)

    print("Indexing current issue to RAG...")
    voyage_client = VoyageEmbeddingClient(api_key=config.voyage_api_key)
    qdrant_client = QdrantSearchClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=config.qdrant_collection_name,
    )
    qdrant_client.ensure_collection(vector_size=256)

    # チャンク分割
    chunks = create_issue_chunks(config.issue_title, config.issue_body)

    # 各チャンクのEmbeddingベクトル生成
    vectors = create_embeddings_for_chunks(chunks, voyage_client, dimensions=256)

    # IssueのURL生成
    issue_url = (
        f"https://github.com/{config.github_repository}/issues/{config.issue_number}"
        if config.github_repository
        else ""
    )

    qdrant_client.upsert_issue_chunks(
        issue_number=int(config.issue_number),
        chunks=chunks,
        vectors=vectors,
        title=config.issue_title,
        state="open",
        url=issue_url,
        labels=[],
    )
    print("Issue indexed successfully")


if __name__ == "__main__":
    main()
