# /// script
# dependencies = [
#   "google-generativeai>=0.8.3",
# ]
# ///
"""Issue自動改善スクリプト - Phase 1実装

PEP-723対応: uvx で実行可能

実行モード:
1. 通常モード: GitHub Actionsから自動実行（Issue作成時）
2. --dry-run: ローカル検証用（コメント投稿をスキップ、読み取り操作は実行）
3. --index-issues: RAGデータ生成モード（Phase 2以降で実装）
"""

import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed")
    print("This script should be run with 'uvx' which auto-installs dependencies")
    sys.exit(1)

# 型定義
TemplateType = Literal[
    "feature-1", "feature-2-design", "bug_report", "feature-3-coding"
]

# テンプレートタイプに応じた役割と指示
ROLE_AND_INSTRUCTIONS = {
    "feature-1": """あなたはプロジェクト管理の専門家です。以下のIssue記述を、機能要件テンプレートに沿った具体的で詳細な内容に拡張してください。

【重要な指示】
- 抽象的な表現を避け、具体的に記述してください
- Issue記述から推測できる範囲で詳細化してください
- 不明な点は「要確認」として明示してください
- Markdown形式で出力してください
- 各項目は箇条書きで、少なくとも2-3項目記述してください""",
    "feature-2-design": """あなたはソフトウェア設計の専門家です。以下のIssue記述を、機能設計テンプレートに沿った具体的な内容に拡張してください。

【重要な指示】
- 技術的な観点から具体的に記述してください
- 複数の選択肢がある場合は比較検討を記載してください
- Markdown形式で出力してください""",
    "bug_report": """あなたはソフトウェアテストの専門家です。以下のバグ報告を、詳細で再現可能な形式に拡張してください。

【重要な指示】
- 再現手順を具体的に記述してください
- エラーメッセージやスクリーンショットの必要性を明示してください
- Markdown形式で出力してください""",
    "feature-3-coding": """あなたはソフトウェアエンジニアです。以下の実装タスクを、具体的なチェックリスト形式に拡張してください。

【重要な指示】
- 実装範囲を具体的に記述してください
- テストケースを明示してください
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
    template_name: str, issue_body: str, issue_title: str = ""
) -> str:
    """テンプレートに応じたプロンプトを取得"""
    # テンプレートファイルから実際の内容を読み込む
    template_content = load_template_content(template_name)

    role = ROLE_AND_INSTRUCTIONS.get(template_name, ROLE_AND_INSTRUCTIONS["feature-1"])

    prompt = f"""{role}

【Issue記述】
タイトル: {issue_title}
本文: {issue_body}

【出力テンプレート】
以下のテンプレートに沿って具体的に記述してください：

{template_content}
"""
    return prompt


# ==================== テンプレート判定 ====================


class TemplateDetector:
    """Issue内容からテンプレートを判定"""

    KEYWORDS = {
        "feature-1": ["機能", "追加", "変更", "改善", "したい", "欲しい", "必要"],
        "feature-2-design": [
            "設計",
            "アーキテクチャ",
            "技術選定",
            "実装方針",
            "設計書",
        ],
        "bug_report": ["バグ", "エラー", "不具合", "動かない", "失敗", "問題"],
        "feature-3-coding": ["実装", "コーディング", "テスト", "PR", "修正"],
    }

    def detect(self, issue_body: str, issue_title: str = "") -> TemplateType:
        """Issue本文とタイトルからテンプレートを判定（キーワードベース）"""
        text = f"{issue_title} {issue_body}".lower()

        scores = {}
        for template, keywords in self.KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[template] = score

        if not scores:
            return "feature-1"

        best_template = max(scores, key=scores.get)

        # スコアが0の場合（キーワードマッチなし）はデフォルトでfeature-1
        if scores[best_template] == 0:
            return "feature-1"

        return best_template


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
    issue_body: str, issue_title: str, api_key: str
) -> tuple[str, str]:
    """Issue内容を改善した例文を生成

    Args:
        issue_body: Issue本文
        issue_title: Issueタイトル
        api_key: LLM APIキー

    Returns:
        (improved_content, template_name): 改善された内容とテンプレート名
    """
    # テンプレート判定
    detector = TemplateDetector()
    template_name = detector.detect(issue_body, issue_title)
    print(f"Detected template: {template_name}")

    # LLM呼び出し
    client = LLMClient(api_key=api_key)
    prompt = get_improve_prompt(template_name, issue_body, issue_title)
    improved_content = client.generate(prompt)
    print("Content generated successfully")

    return improved_content, template_name


def format_comment(improved_content: str, template_name: str) -> str:
    """コメント用のフォーマット済み文字列を生成

    Args:
        improved_content: 改善された内容
        template_name: テンプレート名

    Returns:
        フォーマット済みのコメント文字列
    """
    template_display_names = {
        "feature-1": "機能要件（親Issue）",
        "feature-2-design": "機能設計（子Issue）",
        "bug_report": "バグ報告",
        "feature-3-coding": "実装タスク",
    }
    template_display = template_display_names.get(template_name, template_name)

    return f"""## 🤖 AIによるIssue記入例

**選定テンプレート**: {template_display}

---

{improved_content}

---

💡 **使い方**: 上記の例文を参考に、Issue本文を編集してください。実際のプロジェクトに合わせて内容を修正してください。

<!-- AI-generated comment -->
"""


def main():
    # 引数解析
    parser = argparse.ArgumentParser(description="Issue自動改善スクリプト")
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
    parser.add_argument("--start", type=int, help="RAGインデックス開始Issue番号")
    parser.add_argument("--end", type=int, help="RAGインデックス終了Issue番号")
    args = parser.parse_args()

    # RAGデータ生成モード（Phase 2以降で実装）
    if args.index_issues:
        print("RAG indexing mode")
        print("This feature will be implemented in Phase 2")
        sys.exit(0)

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

    # 改善内容を生成
    improved_content, template_name = generate_improved_content(
        issue_body, issue_title, api_key
    )

    # コメント用にフォーマット
    output = format_comment(improved_content, template_name)

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


if __name__ == "__main__":
    main()
