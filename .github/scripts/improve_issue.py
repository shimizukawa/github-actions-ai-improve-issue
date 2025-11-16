# /// script
# dependencies = [
#   "google-generativeai>=0.8.3",
#   "voyageai>=0.2.3",
#   "qdrant-client>=1.7.0",
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
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed")
    print("This script should be run with 'uvx' which auto-installs dependencies")
    sys.exit(1)

# RAGライブラリは存在チェックのみ（未インストール時もエラーとしない）
RAG_AVAILABLE = False
try:
    import voyageai
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    RAG_AVAILABLE = True
except ImportError:
    pass  # RAG未使用モードで動作

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

    with open(template_file, "r", encoding="utf-8") as f:
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
    similar_issues: Optional[List[Dict[str, Any]]] = None,
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
        if not RAG_AVAILABLE:
            raise RuntimeError("RAG libraries not available")
        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    def generate_embedding(self, text: str, dimensions: int = 256) -> List[float]:
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
        if not RAG_AVAILABLE:
            raise RuntimeError("RAG libraries not available")
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

    def search_similar_issues(
        self, query_vector: List[float], limit: int = 3
    ) -> List[Dict[str, Any]]:
        """類似Issue検索

        Args:
            query_vector: クエリベクトル
            limit: 取得件数（Top-K）

        Returns:
            類似Issue情報のリスト
        """
        try:
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit,
            )

            similar_issues = []
            for result in results:
                similar_issues.append(
                    {
                        "issue_number": result.payload.get("issue_number"),
                        "issue_title": result.payload.get("issue_title", ""),
                        "issue_body": result.payload.get("issue_body", "")[:500],
                        "template_type": result.payload.get("template_type", ""),
                        "state": result.payload.get("state", ""),
                        "url": result.payload.get("url", ""),
                        "similarity": result.score,
                    }
                )
            return similar_issues
        except Exception as e:
            print(f"Warning: Failed to search similar issues: {e}")
            return []

    def upsert_issue(
        self,
        issue_number: int,
        vector: List[float],
        title: str,
        body: str,
        template_type: str,
        state: str,
        url: str,
        labels: List[str],
    ):
        """Issueをインデックスに登録または更新

        Args:
            issue_number: Issue番号
            vector: Embeddingベクトル
            title: Issueタイトル
            body: Issue本文
            template_type: テンプレートタイプ
            state: Issueステート（open/closed）
            url: IssueのURL
            labels: ラベルリスト
        """
        point = PointStruct(
            id=issue_number,
            vector=vector,
            payload={
                "issue_number": issue_number,
                "issue_title": title,
                "issue_body": body[:1000],  # 最初の1000文字のみ保存
                "template_type": template_type,
                "state": state,
                "url": url,
                "labels": labels,
            },
        )
        self.client.upsert(collection_name=self.COLLECTION_NAME, points=[point])
        print(f"Issue #{issue_number} indexed successfully")


# ==================== GitHub API ====================


def fetch_issue_from_github(issue_number: int, github_token: str) -> Optional[Dict]:
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

    cmd = [
        "gh",
        "api",
        f"/repos/{repo}/issues/{issue_number}",
        "--jq",
        ".number,.title,.body,.state,.html_url,.labels[].name",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env={"GH_TOKEN": github_token},
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) < 5:
            return None

        return {
            "number": int(lines[0]),
            "title": lines[1],
            "body": lines[2] if len(lines) > 2 else "",
            "state": lines[3] if len(lines) > 3 else "open",
            "url": lines[4] if len(lines) > 4 else "",
            "labels": lines[5:] if len(lines) > 5 else [],
        }
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to fetch issue #{issue_number}: {e}")
        return None


def fetch_all_issues(
    github_token: str, start: int = 1, end: Optional[int] = None
) -> List[Dict]:
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

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd(),
            env={"GH_TOKEN": github_token},
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
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to fetch issues: {e}")
        return []


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
    similar_issues: Optional[List[Dict[str, Any]]] = None,
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
    similar_issues: Optional[List[Dict[str, Any]]] = None,
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


def check_rag_available() -> tuple[bool, str]:
    """RAG機能が利用可能かチェック

    Returns:
        (available, reason): 利用可能かどうかと理由
    """
    if not RAG_AVAILABLE:
        return False, "RAG libraries not installed"

    qdrant_url = os.environ.get("QDRANT_URL", "")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
    voyage_api_key = os.environ.get("VOYAGE_API_KEY", "")

    if not qdrant_url or not qdrant_api_key:
        return False, "QDRANT_URL or QDRANT_API_KEY not set"

    if not voyage_api_key:
        return False, "VOYAGE_API_KEY not set"

    return True, ""


def index_all_issues(start: int = 1, end: Optional[int] = None):
    """全Issueをインデックス登録（--index-issues モード）

    Args:
        start: 開始Issue番号
        end: 終了Issue番号（Noneの場合は全て）
    """
    # RAG機能チェック
    rag_available, reason = check_rag_available()
    if not rag_available:
        print(f"Error: RAG not available - {reason}")
        sys.exit(1)

    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        print("Error: GITHUB_TOKEN not set")
        sys.exit(1)

    print("=== RAG Indexing Mode ===")
    print(f"Fetching issues from GitHub...")

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
        try:
            print(f"[{i}/{len(issues)}] Indexing issue #{issue['number']}...")

            # Embeddingベクトル生成
            text = f"{issue['title']}\n{issue['body']}"
            vector = voyage_client.generate_embedding(text, dimensions=256)

            # テンプレート判定
            template_type = detector.detect(issue["body"], issue["title"])

            # Qdrantに登録
            qdrant_client.upsert_issue(
                issue_number=issue["number"],
                vector=vector,
                title=issue["title"],
                body=issue["body"],
                template_type=template_type,
                state=issue["state"],
                url=issue["url"],
                labels=issue.get("labels", []),
            )
            success_count += 1
        except Exception as e:
            print(f"Error indexing issue #{issue['number']}: {e}")

    print(f"\n=== Indexing Complete ===")
    print(f"Success: {success_count}/{len(issues)} issues")


def update_single_issue(issue_number: int):
    """単一Issueをインデックス更新（--update-single-issue モード）

    Args:
        issue_number: Issue番号
    """
    # RAG機能チェック
    rag_available, reason = check_rag_available()
    if not rag_available:
        print(f"Error: RAG not available - {reason}")
        sys.exit(1)

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

    # Embeddingベクトル生成
    text = f"{issue['title']}\n{issue['body']}"
    vector = voyage_client.generate_embedding(text, dimensions=256)

    # Qdrantに登録
    qdrant_client.upsert_issue(
        issue_number=issue["number"],
        vector=vector,
        title=issue["title"],
        body=issue["body"],
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
    rag_available, reason = check_rag_available()
    similar_issues = None

    if rag_available:
        print("RAG mode: Enabled")
        try:
            # RAG検索
            voyage_client = VoyageEmbeddingClient(api_key=os.environ["VOYAGE_API_KEY"])
            qdrant_client = QdrantSearchClient(
                url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"]
            )

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
        except Exception as e:
            print(f"Warning: RAG search failed - {e}")
            print("Falling back to non-RAG mode")
            similar_issues = None
    else:
        print(f"RAG mode: Disabled ({reason})")

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
    if rag_available:
        try:
            print("Indexing current issue to RAG...")
            detector = TemplateDetector()
            template_type = detector.detect(issue_body, issue_title)

            voyage_client = VoyageEmbeddingClient(api_key=os.environ["VOYAGE_API_KEY"])
            qdrant_client = QdrantSearchClient(
                url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"]
            )
            qdrant_client.ensure_collection(vector_size=256)

            text = f"{issue_title}\n{issue_body}"
            vector = voyage_client.generate_embedding(text, dimensions=256)

            # IssueのURL生成
            repo = os.environ.get("GITHUB_REPOSITORY", "")
            issue_url = (
                f"https://github.com/{repo}/issues/{issue_number}" if repo else ""
            )

            qdrant_client.upsert_issue(
                issue_number=int(issue_number),
                vector=vector,
                title=issue_title,
                body=issue_body,
                template_type=template_type,
                state="open",
                url=issue_url,
                labels=[],
            )
            print("Issue indexed successfully")
        except Exception as e:
            print(f"Warning: Failed to index issue - {e}")


if __name__ == "__main__":
    main()
