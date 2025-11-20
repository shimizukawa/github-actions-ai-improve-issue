# /// script
# dependencies = [
#   "google-generativeai>=0.8.3",
#   "voyageai>=0.2.3",
#   "qdrant-client==1.16.*",
#   "langchain-text-splitters>=0.3.0",
# ]
# ///
"""Issue自動改善スクリプト - Phase 2実装

PEP-723対応: uvx で実行可能

実行モード:
1. 通常モード: GitHub Actionsから自動実行（Issue作成時）
2. --dry-run: ローカル検証用（コメント投稿をスキップ、読み取り操作は実行）
3. --index-issues: RAGデータ生成モード（全Issueをベクトル化）
4. --update-single-issue: 単一Issue更新モード
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Literal, Any

import google.generativeai as genai
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


# 型定義
TemplateType = Literal["feature_request", "bug_report"]

# テンプレートタイプに応じた役割と指示
ROLE_AND_INSTRUCTIONS = {
    "feature_request": """あなたはプロジェクト管理の専門家です。以下のIssue記述を、機能要件テンプレートに沿った具体的で詳細な内容に拡張してください。

【重要な指示】
- 抽象的な表現を避け、具体的に記述してください
- Issue記述から推測できる範囲で詳細化してください
- 不明な点は「要確認」として明示してください
- Markdown形式で出力してください
- 各項目は箇条書きで、少なくとも2-3項目記述してください""",
    "bug_report": """あなたはソフトウェアテストの専門家です。以下のバグ報告を、詳細で再現可能な形式に拡張してください。

【重要な指示】
- 再現手順を具体的に記述してください
- エラーメッセージやスクリーンショットの必要性を明示してください
- Markdown形式で出力してください""",
}

# ==================== テンプレート読み込み ====================


def load_template_content(template_name: str) -> str:
    """ISSUE_TEMPLATEファイルから実際のテンプレート内容を読み込む"""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    template_file = repo_root / ".github" / "ISSUE_TEMPLATE" / f"{template_name}.md"

    if not template_file.exists():
        return ""

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
) -> str:
    """テンプレートに応じたプロンプトを取得（RAG対応）

    Args:
        template_name: テンプレート名
        issue_body: Issue本文
        issue_title: Issueタイトル
        similar_issues: 類似Issue情報（RAG検索結果）

    Returns:
        LLMプロンプト
    """
    # テンプレートファイルから実際の内容を読み込む
    template_content = load_template_content(template_name)

    role = ROLE_AND_INSTRUCTIONS.get(
        template_name, ROLE_AND_INSTRUCTIONS["feature_request"]
    )

    prompt = f"""{role}

【Issue記述】
タイトル: {issue_title}
本文: {issue_body}

【出力テンプレート】
以下のテンプレートに沿って具体的に記述してください：

{template_content}
"""

    # RAG検索結果があれば追加
    if similar_issues and len(similar_issues) > 0:
        similar_info = "\n\n【参考情報】\n以下の過去Issueを参考にしてください：\n"
        for i, issue in enumerate(similar_issues, 1):
            similar_info += f"""
【参考Issue {i}】
- タイトル: {issue["issue_title"]}
- 本文抜粋: {issue["issue_body"][:200]}...
- 類似度: {issue["similarity"]:.1%}
"""
        similar_info += "\n上記の参考Issueから、記述スタイルや必要な情報項目を学び、より具体的で実用的な例文を生成してください。"
        prompt += similar_info

    return prompt


# ==================== テンプレート判定 ====================


class TemplateDetector:
    """Issue内容からテンプレートを判定"""

    KEYWORDS: dict[TemplateType, list[str]] = {
        "feature_request": ["機能", "追加", "変更", "改善", "したい", "欲しい", "必要"],
        "bug_report": ["バグ", "エラー", "不具合", "動かない", "失敗", "問題"],
    }

    def detect(self, issue_body: str, issue_title: str = "") -> TemplateType:
        """Issue本文とタイトルからテンプレートを判定（キーワードベース）"""
        text = f"{issue_title} {issue_body}".lower()

        scores: dict[TemplateType, int] = {}
        for template, keywords in self.KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[template] = score

        if not scores:
            return "feature_request"

        selected_tmpl = max(scores, key=scores.get)

        # スコアが0の場合（キーワードマッチなし）はデフォルトでfeature_request
        if scores[selected_tmpl] == 0:
            return "feature_request"

        return selected_tmpl


# ==================== LLMクライアント ====================


class LLMClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """LLMクライアント

        Args:
            api_key: APIキー
            model: モデル名（2025年11月時点の推奨）
                - Phase 0: 'gemini-2.0-flash-lite' (検証用、極低コスト)
                - Phase 1-2: 'gemini-2.5-flash' (コスパ良好)
                - Phase 2: 'claude-3.7-sonnet' (品質重視)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """プロンプトから文章を生成"""
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
            ),
        )
        return response.text


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

    COLLECTION_NAME = "improve-issues"

    def __init__(self, url: str, api_key: str):
        """
        Args:
            url: Qdrant CloudのURL
            api_key: Qdrant APIキー
        """
        self.client = QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, vector_size: int = 256):
        """コレクションが存在することを確認、なければ作成"""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Collection '{self.COLLECTION_NAME}' created")
            self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="issue_number",
                field_schema=PayloadSchemaType.INTEGER,
            )

    def search_similar_issues(
        self, query_vector: list[float], limit: int = 3
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
            collection_name=self.COLLECTION_NAME,
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
                    "template_type": result.payload.get("template_type", ""),
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
        template_type: str,
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
            template_type: テンプレートタイプ
            state: Issueステート（open/closed）
            url: IssueのURL
            labels: ラベルリスト
        """
        # 既存のチャンクを削除（issue_numberで始まるIDを検索して削除）
        ids_to_delete: list[str] = []
        offset: dict | None = None
        while True:
            existing_points, next_offset = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
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
                collection_name=self.COLLECTION_NAME,
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
                    "template_type": template_type,
                    "state": state,
                    "url": url,
                    "labels": labels,
                },
            )
            points.append(point)

        self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)
        print(f"Issue #{issue_number} indexed with {len(chunks)} chunks")


# ==================== GitHub API ====================


def fetch_issue_from_github(issue_number: int, github_token: str) -> dict | None:
    """GitHub APIからIssue情報を取得

    Args:
        issue_number: Issue番号
        github_token: GitHub Token

    Returns:
        Issue情報の辞書、取得失敗時はNone
    """
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        print("Error: GITHUB_REPOSITORY not set")
        return None

    cmd = ["gh", "api", f"/repos/{repo}/issues/{issue_number}"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        env={
            "GH_TOKEN": github_token,
            "GH_REPO": repo,
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


def fetch_all_issues(
    github_token: str, start: int = 1, end: int | None = None
) -> list[dict]:
    """全Issue情報を取得

    Args:
        github_token: GitHub Token
        start: 開始Issue番号
        end: 終了Issue番号（Noneの場合は全て）

    Returns:
        Issue情報のリスト
    """
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
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
            "GH_TOKEN": github_token,
            "GH_REPO": repo,
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
        issue = fetch_issue_from_github(num, github_token)
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
    api_key: str,
    similar_issues: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Issue内容を改善した例文を生成（RAG対応）

    Args:
        issue_body: Issue本文
        issue_title: Issueタイトル
        api_key: LLM APIキー
        similar_issues: 類似Issue情報（RAG検索結果）

    Returns:
        (improved_content, template_name): 改善された内容とテンプレート名
    """
    # テンプレート判定
    detector = TemplateDetector()
    template_name = detector.detect(issue_body, issue_title)
    print(f"Detected template: {template_name}")

    # LLM呼び出し
    client = LLMClient(api_key=api_key)
    prompt = get_improve_prompt(template_name, issue_body, issue_title, similar_issues)
    improved_content = client.generate(prompt)
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


def index_all_issues(start: int = 1, end: int | None = None):
    """全Issueをインデックス登録（--index-issues モード）

    Args:
        start: 開始Issue番号
        end: 終了Issue番号（Noneの場合は全て）
    """
    # RAG機能チェック
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        print("Error: GITHUB_TOKEN not set")
        sys.exit(1)

    print("=== RAG Indexing Mode ===")
    print("Fetching issues from GitHub...")

    # Issue一覧取得
    issues = fetch_all_issues(github_token, start, end)
    if not issues:
        print("No issues found")
        sys.exit(0)

    print(f"Found {len(issues)} issues to index")

    # クライアント初期化
    voyage_client = VoyageEmbeddingClient(api_key=os.environ["VOYAGE_API_KEY"])
    qdrant_client = QdrantSearchClient(
        url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"]
    )
    qdrant_client.ensure_collection(vector_size=256)

    # テンプレート判定器
    detector = TemplateDetector()

    # 各Issueをインデックス登録
    success_count = 0
    for i, issue in enumerate(issues, 1):
        print(f"[{i}/{len(issues)}] Indexing issue #{issue['number']}...")

        # チャンク分割
        chunks = create_issue_chunks(issue["title"], issue["body"])

        # 各チャンクのEmbeddingベクトル生成
        vectors = create_embeddings_for_chunks(chunks, voyage_client, dimensions=256)

        # テンプレート判定
        template_type = detector.detect(issue["body"], issue["title"])

        # Qdrantに登録
        qdrant_client.upsert_issue_chunks(
            issue_number=issue["number"],
            chunks=chunks,
            vectors=vectors,
            title=issue["title"],
            template_type=template_type,
            state=issue["state"],
            url=issue["url"],
            labels=issue.get("labels", []),
        )
        success_count += 1

    print("\n=== Indexing Complete ===")
    print(f"Success: {success_count}/{len(issues)} issues")


def update_single_issue(issue_number: int):
    """単一Issueをインデックス更新（--update-single-issue モード）

    Args:
        issue_number: Issue番号
    """
    # RAG機能チェック
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        print("Error: GITHUB_TOKEN not set")
        sys.exit(1)

    print(f"=== Update Single Issue #{issue_number} ===")

    # Issue情報取得
    issue = fetch_issue_from_github(issue_number, github_token)
    if not issue:
        print(f"Error: Failed to fetch issue #{issue_number}")
        sys.exit(1)

    # クライアント初期化
    voyage_client = VoyageEmbeddingClient(api_key=os.environ["VOYAGE_API_KEY"])
    qdrant_client = QdrantSearchClient(
        url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"]
    )
    qdrant_client.ensure_collection(vector_size=256)

    # テンプレート判定
    detector = TemplateDetector()
    template_type = detector.detect(issue["body"], issue["title"])

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
        template_type=template_type,
        state=issue["state"],
        url=issue["url"],
        labels=issue.get("labels", []),
    )

    print(f"Issue #{issue_number} updated successfully")


def main():
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
        index_all_issues(start=args.start, end=args.end)
        sys.exit(0)

    # 単一Issue更新モード
    if args.update_single_issue:
        update_single_issue(args.update_single_issue)
        sys.exit(0)

    # 通常モード: Issue改善
    # 環境変数取得
    issue_body = os.environ.get("ISSUE_BODY", "")
    issue_title = os.environ.get("ISSUE_TITLE", "")
    issue_number = os.environ.get("ISSUE_NUMBER", "")
    api_key = os.environ.get("LLM_API_KEY", "")

    # 環境変数チェック
    if not issue_number:
        print("Error: ISSUE_NUMBER not set")
        sys.exit(1)

    if not api_key:
        print("Error: LLM_API_KEY not set")
        sys.exit(1)

    # 改善が必要かチェック
    if not check_needs_improvement(issue_body, issue_title):
        print(f"Issue #{issue_number} does not need improvement (too short)")
        sys.exit(0)

    print(f"Processing issue #{issue_number}")
    print(f"Title: {issue_title}")
    print(f"Body length: {len(issue_body)} characters")

    # RAG機能チェック
    similar_issues = None

    if (
        os.getenv("QDRANT_URL")
        and os.getenv("QDRANT_API_KEY")
        and os.getenv("VOYAGE_API_KEY")
    ):
        print("RAG mode: Enabled")
        # RAG検索
        voyage_client = VoyageEmbeddingClient(api_key=os.environ["VOYAGE_API_KEY"])
        qdrant_client = QdrantSearchClient(
            url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"]
        )
        qdrant_client.ensure_collection(vector_size=256)

        # クエリベクトル生成
        query_text = f"{issue_title}\n{issue_body}"
        query_vector = voyage_client.generate_embedding(query_text, dimensions=256)

        # 類似Issue検索
        similar_issues = qdrant_client.search_similar_issues(query_vector, limit=3)

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
        issue_body, issue_title, api_key, similar_issues
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
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        print("Error: GITHUB_TOKEN not found")
        sys.exit(1)

    post_comment_via_gh(issue_number, output)

    # RAGインデックス登録（例文生成後）
    print("Indexing current issue to RAG...")
    detector = TemplateDetector()
    template_type = detector.detect(issue_body, issue_title)

    voyage_client = VoyageEmbeddingClient(api_key=os.environ["VOYAGE_API_KEY"])
    qdrant_client = QdrantSearchClient(
        url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"]
    )
    qdrant_client.ensure_collection(vector_size=256)

    # チャンク分割
    chunks = create_issue_chunks(issue_title, issue_body)

    # 各チャンクのEmbeddingベクトル生成
    vectors = create_embeddings_for_chunks(chunks, voyage_client, dimensions=256)

    # IssueのURL生成
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    issue_url = f"https://github.com/{repo}/issues/{issue_number}" if repo else ""

    qdrant_client.upsert_issue_chunks(
        issue_number=int(issue_number),
        chunks=chunks,
        vectors=vectors,
        title=issue_title,
        template_type=template_type,
        state="open",
        url=issue_url,
        labels=[],
    )
    print("Issue indexed successfully")


if __name__ == "__main__":
    main()
