# 設計書: Issue自動ブラッシュアップ機能

## 概要

簡易なブランクIssueに1-2行程度のやりたいことを書いたら、過去の類似Issue情報を参考に適切なテンプレートに沿った具体的な文章を自動生成し、コメントで提示する機能を実装する。Phase 0の技術検証を経て、段階的に機能を拡張していく。

## 親Issue

- （要件Issue番号を記載予定）

## アーキテクチャ

### システム構成図

```
┌─────────────────────────────────────────────────────────────┐
│                      GitHub Repository                       │
│                                                              │
│  ┌──────────────┐                                           │
│  │   Issue      │                                           │
│  │   Created    │                                           │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         GitHub Actions Workflow                       │  │
│  │                                                        │  │
│  │  1. Issue内容取得                                      │  │
│  │  2. 簡易記述判定（200文字以内？）                      │  │
│  │  3. テンプレート自動判定                               │  │
│  │  4. RAG検索（Phase 2以降）                            │  │
│  │  5. LLMで例文生成                                      │  │
│  │  6. コメント投稿                                       │  │
│  └──────┬───────────────────────────────────────────────┘  │
│         │                                                    │
└─────────┼────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    External Services                         │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  LLM API    │  │ Qdrant      │  │ Embedding   │        │
│  │ Gemini 2.5 │  │ Cloud       │  │ Service     │        │
│  │ or Claude  │  │ (Phase 2)   │  │ (Phase 2)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### ディレクトリ構造

```
.github/
├── workflows/
│   └── issue_auto_improve.yml          # メインWorkflow
└── scripts/
    └── improve_issue.py                # 単一スクリプト（全機能統合・PEP-723対応）
```

## Phase別詳細設計

### Phase 0: 技術検証・PoC

#### Workflow設計

**ファイル: `.github/workflows/issue_auto_improve.yml`**

```yaml
name: Issue Auto Improve (Phase 0)

on:
  issues:
    types: [opened]

jobs:
  improve-issue:
    runs-on: ubuntu-latest           # 標準ランナー (2コア7GB)
    timeout-minutes: 2               # API呼び出し中心の処理で15-20秒程度
    # 注: ubuntu-slim (1コア5GB) でも十分実行可能

    # no-ai-assistラベルがついている場合はスキップ
    if: "!contains(github.event.issue.labels.*.name, 'no-ai-assist')"

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          version: "latest"

      - name: Improve issue content
        id: improve
        run: |
          uvx .github/scripts/improve_issue.py
        env:
          ISSUE_BODY: ${{ github.event.issue.body }}
          ISSUE_TITLE: ${{ github.event.issue.title }}
          ISSUE_NUMBER: ${{ github.event.issue.number }}
          LLM_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPOSITORY: ${{ github.repository }}

      - name: Handle errors
        if: failure()
        run: |
          gh issue comment ${{ github.event.issue.number }} \
            --body '⚠️ Issue自動改善機能でエラーが発生しました。手動でIssue内容を記入してください。'
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

#### スクリプト設計

**ファイル: `.github/scripts/improve_issue.py`** （単一ファイル・全機能統合）

```python
# /// script
# dependencies = [
#   "google-generativeai>=0.8.3",
# ]
# ///
"""Issue自動改善スクリプト - 全機能統合版

PEP-723対応: uvx で実行可能

実行モード:
1. 通常モード: GitHub Actionsから自動実行（Issue作成時）
2. --dry-run: ローカル検証用（コメント投稿をスキップ、読み取り操作は実行）
3. --index-issues: RAGデータ生成モード（全Issue/指定範囲をベクトル化してQdrantへ登録）
"""
import argparse
import os
import sys
from typing import Literal
import google.generativeai as genai

# 型定義
TemplateType = Literal['feature-1', 'feature-2-design', 'bug_report', 'feature-3-coding']

# ==================== プロンプトテンプレート ====================

FEATURE_1_TEMPLATE = """あなたはプロジェクト管理の専門家です。以下のIssue記述を、機能要件テンプレートに沿った具体的で詳細な内容に拡張してください。

【Issue記述】
タイトル: {issue_title}
本文: {issue_body}

【出力テンプレート】
以下の構造で具体的に記述してください：

## 背景・目的
- なぜこの機能が必要なのか
- 現状の問題点
- 解決したい課題

## 完了条件（Acceptance Criteria）
- [ ] 条件1
- [ ] 条件2
- [ ] 条件3

## 影響範囲
- 影響を受ける既存機能
- ユーザーへの影響
- 他システムへの影響

## 実装に必要な情報
- 必要な情報・資料
- 確認が必要な事項

【重要な指示】
- 抽象的な表現を避け、具体的に記述してください
- Issue記述から推測できる範囲で詳細化してください
- 不明な点は「要確認」として明示してください
- Markdown形式で出力してください
"""

FEATURE_2_DESIGN_TEMPLATE = """あなたはソフトウェア設計の専門家です。以下のIssue記述を、機能設計テンプレートに沿った具体的な内容に拡張してください。

【Issue記述】
タイトル: {issue_title}
本文: {issue_body}

【出力テンプレート】
以下の構造で具体的に記述してください：

## 設計概要
- 機能の概要
- 設計方針

## 実装に必要な情報
- 現在のコード・ロジックの仕様
- APIやフォーマット仕様
- データ構造・スキーマ
- 既存の類似実装

## 技術的検討事項
- アーキテクチャ選定
- ライブラリ・フレームワーク選定
- パフォーマンス要件
- セキュリティ要件

## タスク
- [ ] 設計書の作成・修正
- [ ] 設計書レビュー

【重要な指示】
- 技術的な観点から具体的に記述してください
- 複数の選択肢がある場合は比較検討を記載してください
- Markdown形式で出力してください
"""

BUG_REPORT_TEMPLATE = """あなたはソフトウェアテストの専門家です。以下のバグ報告を、詳細で再現可能な形式に拡張してください。

【バグ報告】
タイトル: {issue_title}
本文: {issue_body}

【出力テンプレート】
以下の構造で具体的に記述してください：

## 現象
- 何が起きているか
- いつから発生しているか

## 再現手順
1. ステップ1
2. ステップ2
3. ステップ3

## 期待する動作
- 本来どうあるべきか

## 実際の動作
- 実際にどうなっているか

## 環境情報
- ブラウザ / OS / バージョン
- 関連する設定

## エラーメッセージ・ログ
```
（該当箇所）
```

【重要な指示】
- 再現手順を具体的に記述してください
- エラーメッセージやスクリーンショットの必要性を明示してください
- Markdown形式で出力してください
"""

FEATURE_3_CODING_TEMPLATE = """あなたはソフトウェアエンジニアです。以下の実装タスクを、具体的なチェックリスト形式に拡張してください。

【実装タスク】
タイトル: {issue_title}
本文: {issue_body}

【出力テンプレート】
以下の構造で具体的に記述してください：

## 実装内容
- 実装する機能・修正内容

## 実装タスク
- [ ] タスク1
- [ ] タスク2
- [ ] タスク3

## テスト
- [ ] ユニットテスト作成
- [ ] 動作確認

## 備考
- 実装時の注意点
- 参考資料

【重要な指示】
- 実装範囲を具体的に記述してください
- テストケースを明示してください
- Markdown形式で出力してください
"""

def get_improve_prompt(template_name: str, issue_body: str, issue_title: str = '') -> str:
    """テンプレートに応じたプロンプトを取得"""
    templates = {
        'feature-1': FEATURE_1_TEMPLATE,
        'feature-2-design': FEATURE_2_DESIGN_TEMPLATE,
        'bug_report': BUG_REPORT_TEMPLATE,
        'feature-3-coding': FEATURE_3_CODING_TEMPLATE,
    }
    template = templates.get(template_name, FEATURE_1_TEMPLATE)
    return template.format(issue_title=issue_title, issue_body=issue_body)

# ==================== テンプレート判定 ====================

class TemplateDetector:
    """Issue内容からテンプレートを判定"""

    KEYWORDS = {
        'feature-1': ['機能', '追加', '変更', '改善', 'したい', '欲しい'],
        'feature-2-design': ['設計', 'アーキテクチャ', '技術選定', '実装方針'],
        'bug_report': ['バグ', 'エラー', '不具合', '動かない', '失敗'],
        'feature-3-coding': ['実装', 'コーディング', 'テスト', 'PR'],
    }

    def detect(self, issue_body: str, issue_title: str = '') -> TemplateType:
        """Issue本文とタイトルからテンプレートを判定"""
        text = f"{issue_title} {issue_body}".lower()

        scores = {}
        for template, keywords in self.KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[template] = score

        best_template = max(scores, key=scores.get)

        if scores[best_template] == 0:
            return 'feature-1'

        return best_template

# ==================== LLMクライアント ====================

class LLMClient:
    def __init__(self, api_key: str, model: str = 'gemini-2.5-flash'):
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
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7,
                )
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"LLM API呼び出しエラー: {e}")

# ==================== メイン処理 ====================

def check_needs_improvement(issue_body: str, issue_title: str) -> bool:
    """Issue改善が必要かチェック"""
    if not issue_body.strip():
        return False

    if len(issue_body) > 200:
        return False

    template_markers = ['## 背景・目的', '## 完了条件', '## 影響範囲']
    if any(marker in issue_body for marker in template_markers):
        return False

    return True

def main():
    # 引数解析
    parser = argparse.ArgumentParser(description='Issue自動改善スクリプト')
    parser.add_argument('--dry-run', action='store_true',
                        help='ローカル検証用（コメント投稿をスキップ）')
    parser.add_argument('--index-issues', action='store_true',
                        help='RAGデータ生成モード（全Issueをベクトル化）')
    parser.add_argument('--start', type=int,
                        help='RAGインデックス開始Issue番号')
    parser.add_argument('--end', type=int,
                        help='RAGインデックス終了Issue番号')
    args = parser.parse_args()

    # RAGデータ生成モード
    if args.index_issues:
        print("RAG indexing mode")
        index_issues_to_qdrant(args.start, args.end)
        return

    # Issue改善モード（通常モード / --dry-run）
    issue_body = os.environ.get('ISSUE_BODY', '')
    issue_title = os.environ.get('ISSUE_TITLE', '')
    issue_number = os.environ.get('ISSUE_NUMBER', '')
    api_key = os.environ.get('LLM_API_KEY', '')

    # 改善が必要かチェック
    if not check_needs_improvement(issue_body, issue_title):
        print("Issue does not need improvement")
        if not args.dry_run:
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write('improved=false\n')
        sys.exit(0)

    print(f"Processing issue #{issue_number}")

    # テンプレート判定
    detector = TemplateDetector()
    template_name = detector.detect(issue_body, issue_title)
    print(f"Detected template: {template_name}")

    # LLM呼び出し
    client = LLMClient(api_key=api_key)
    prompt = get_improve_prompt(template_name, issue_body, issue_title)
    improved_content = client.generate(prompt)
    print("Content generated successfully")

    # 出力
    output = f"""## 🤖 AIによるIssue記入例

**選定テンプレート**: {template_name}

---

{improved_content}

---

💡 **使い方**: 上記の例文を参考に、Issue本文を編集してください。実際のプロジェクトに合わせて内容を修正してください。
"""

    # --dry-run モードではコンソール出力のみ
    if args.dry_run:
        print("[DRY RUN] コメント投稿をスキップします")
        print("\n" + "="*60)
        print(output)
        print("="*60)
        return

    # 通常モード: GitHub CLIでコメント投稿
    github_token = os.environ.get('GITHUB_TOKEN', '')
    if not github_token:
        print("Error: GITHUB_TOKEN not found")
        sys.exit(1)

    # 一時ファイルに本文を書き出し
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.md', delete=False) as f:
        f.write(output)
        temp_file = f.name

    try:
        # GitHub CLIでコメント投稿
        import subprocess
        result = subprocess.run(
            ['gh', 'issue', 'comment', issue_number, '--body-file', temp_file],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Comment posted successfully to issue #{issue_number}")
    except subprocess.CalledProcessError as e:
        print(f"Error posting comment: {e.stderr}")
        sys.exit(1)
    finally:
        # 一時ファイルを削除
        import os as os_module
        if os_module.path.exists(temp_file):
            os_module.unlink(temp_file)

def index_issues_to_qdrant(start: int = None, end: int = None):
    """RAGデータ生成: 全Issue（または指定範囲）をベクトル化してQdrantへ登録

    Args:
        start: 開始Issue番号（指定時は範囲指定モード）
        end: 終了Issue番号
    """
    # Phase 2以降で実装
    # 1. GitHub APIで全Issue取得（またはstart-end範囲）
    # 2. 各IssueをEmbedding APIでベクトル化
    # 3. Qdrant Cloudへアップロード
    raise NotImplementedError("Phase 2で実装予定")

if __name__ == '__main__':
    main()
```

### ローカル開発・検証モード

**--dry-run モード（ローカル検証用）**

```bash
# 環境変数を設定してローカル実行
export ISSUE_BODY="想定運転データを一括登録したい"
export ISSUE_TITLE="想定運転データCSV一括登録機能"
export ISSUE_NUMBER="123"
export LLM_API_KEY="your-api-key"

# dry-runモードで実行（コメント投稿スキップ、標準出力に結果表示）
uvx .github/scripts/improve_issue.py --dry-run
```

動作:
- GitHub API読み取り操作: 実行（テンプレート判定に必要）
- RAG検索（Phase 2以降）: 実行
- LLM API呼び出し: 実行
- コメント投稿: **スキップ**（結果は標準出力に表示）

**--index-issues モード（RAGデータ生成）**

```bash
# Phase 2以降で使用
# 全Issueをベクトル化
uvx .github/scripts/improve_issue.py --index-issues

# 範囲指定でインデックス更新
uvx .github/scripts/improve_issue.py --index-issues --start 100 --end 200
```

動作:
- GitHub APIで全Issue取得（または指定範囲）
- 各Issueのタイトル・本文をEmbedding APIでベクトル化
- Qdrant Cloudへアップロード
- Issue改善処理は実行しない

**依存ライブラリ管理:**

PEP-723 (Inline script metadata) を使用し、`improve_issue.py` の先頭に依存関係を記述：

**Phase 0-1:**
```python
# /// script
# dependencies = [
#   "google-generativeai>=0.8.3",
# ]
# ///
```

**Phase 2以降（RAG機能追加時）:**
```python
# /// script
# dependencies = [
#   "google-generativeai>=0.8.3",
#   "qdrant-client>=1.7.0",
#   "openai>=1.0.0",  # Embedding API用
# ]
# ///
```

### Phase 1: 基本機能実装

#### 追加機能

1. **エラーハンドリング強化**
2. **重複投稿防止**
3. **テンプレート判定精度向上（LLM使用）

        prompt = f"""以下のIssue記述から、最も適切なテンプレートを1つ選択してください。

【Issue記述】
タイトル: {issue_title}
本文: {issue_body}

【選択肢】
1. feature-1: 機能要件（新機能の追加、既存機能の変更）
2. feature-2-design: 機能設計（実装方針、技術選定、アーキテクチャ）
3. bug_report: バグ報告（不具合、エラー、想定外の動作）
4. feature-3-coding: 実装（コーディング、テスト実装）

【出力形式】
選択したテンプレート名のみを出力してください（例: feature-1）
"""

        response = self.llm_client.generate(prompt, max_tokens=50)
        detected = response.strip()

        # バリデーション
        valid_templates = ['feature-1', 'feature-2-design', 'bug_report', 'feature-3-coding']
        if detected in valid_templates:
            return detected

        # フォールバック
        return self._keyword_based_detect(issue_body, issue_title)
```

### Phase 2: RAG機能追加

#### Embeddingサービス選定（2025年11月更新）

**推奨構成: Voyage AI 3.5-lite + Qdrant + Matryoshka圧縮**

**モデル選定根拠:**

| モデル | 精度 | コスト/100万T | ベクトル次元 | コンテキスト長 |
|--------|------|-------------|------------|---------------|
| **Voyage 3.5** | OpenAI 3-large +0.3% | $0.06 | 1024 | 32,000 |
| **Voyage 3.5-lite** | OpenAI 3-large -2% | **$0.02** | 1024 | 32,000 |
| OpenAI 3-large | 基準 | $0.13 | 3072 | 8,191 |
| OpenAI 3-small | 低 | $0.02 | 1536 | 8,191 |
| Gemini Embedding-004 | 中 | **$0** (無料枠) | 768 | 2048 |

**Phase 2推奨: Voyage 3.5-lite**

1. **精度**: OpenAI text-embedding-3-largeに近い（MTEB実測-2%程度）
2. **コスト**: $0.02/100万トークン（text-embedding-3-smallと同等、しかし高精度）
3. **長文対応**: 32,000トークン（長文Issue対応可）
4. **Matryoshka Embeddings対応**: ベクトル可変圧縮（256-2048次元）

**PoC代替案: Google Gemini Embedding-004**
- 2025年現在無料枠利用可
- コスト0円でPoC実施可能
- 精度が十分であればVoyageスキップも検討

#### アーキテクチャ拡張

```
GitHub Issue作成
    ↓
Workflow起動
    ↓
Issue内容取得
    ↓
テンプレート判定
    ↓
┌───────────────────────────────────────────────────┐
│   RAG検索（Voyage 3.5-lite + Matryoshka）  │
│                                                   │
│  1. Embeddingベクトル化（1024次元）         │  → Voyage API
│  2. Matryoshka圧縮（1024→56次元）          │     (75%メモリ削減)
│  3. Qdrant検索（コサイン類似度）            │  → Qdrant Cloud
│  4. Top-3 Issue取得（Recall@3）             │
│  5. Optional: Reranker（順序最適化）        │  → Cohere Rerank
└───────────────────────────────────────────────────┘
    ↓
LLMで例文生成（類似Issue参考）
    ↓
コメント投稿（類似Issue情報付き）
```

#### ローカルベクトル圧縮（Truncation）

**正確な実装方針**:

1. **Embedding生成**: Voyage AI 3.5-lite API呼び出し（外部）
   - 出力: 1024次元ベクトル
   - 実行場所: Voyage AIサーバー（GitHub Actions外）
   - GitHub Actionsリソース: ネットワーク待機のみ（5-10秒）

2. **ローカル圧縮**: GitHub Actions内で実施
   - 入力: 1024次元ベクトル（8KB）
   - 処理: Truncationで256次元に切り詰め
   - 出力: 256次元ベクトル（2KB）
   - **CPU使用率**: 50%、100msec以下
   - **メモリ使用量**: 50MB以下

3. **効果**:
   - ✅ Qdrant Cloudストレージ: 75%削減
   - ✅ 検索速度: 向上（ベクトルサイズ小）
   - ❌ APIコスト削減: なし（Voyage AIは出力サイズ非依存）
   - ❌ GitHub Actionsコスト削減: なし（元から軽量）

**推奨タイミング**:
- Phase 0-1: 不要（Issue数少なく、節約効果低い）
- Phase 2: Issue数1000件超で推奨（ストレージ効率化）

#### Binary Quantization（Optional、Phase 3以降）

**用途**: Issue数1000件超のスケーラビリティ最適化

**実装方針**:
- float32（32ビット）→binary（1ビット）に変換
- 処理場所: GitHub Actions内（軽量、100msec以下）
- **効果**:
  - Qdrantストレージ: さらに96%削減
  - 検索速度: 40倍高速化
  - 精度低下: 5-10%（トレードオフあり）

**GitHub Actionsリソース影響**: 無視できる程度（CPU 100msec未満）

#### Qdrant設計

**コレクション設計:**

```python
# コレクション名: improve-issues
# ベクトル次元: 768 (text-embedding-3-small) or 1024 (voyage-2)

collection_config = {
    "vectors": {
        "size": 768,
        "distance": "Cosine"
    },
    "payload_schema": {
        "issue_number": "integer",
        "issue_title": "text",
        "issue_body": "text",
        "template_type": "keyword",
        "state": "keyword",  # open/closed
        "created_at": "datetime",
        "labels": "keyword[]",
        "url": "text"
    }
}
```

**インデックス更新フロー:**

```yaml
# .github/workflows/update_issue_index.yml
name: Update Issue Index

on:
  issues:
    types: [opened, edited, closed, reopened]
  schedule:
    - cron: '0 0 * * 0'  # 毎週日曜0時に全体同期

jobs:
  update-index:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          version: "latest"

      - name: Update Qdrant index
        run: |
          uvx .github/scripts/improve_issue.py --index-issues
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          QDRANT_URL: ${{ secrets.QDRANT_URL }}
          QDRANT_API_KEY: ${{ secrets.QDRANT_API_KEY }}
          VOYAGE_API_KEY: ${{ secrets.VOYAGE_API_KEY }}  # Voyage AI
```

**RAG・インデックス更新フロー:**

Phase 2以降で、`improve_issue.py` に以下の機能を追加統合：

```python
# Voyage AI Embeddingクライアント
class VoyageEmbeddingClient:
    def __init__(self, api_key: str, model: str = 'voyage-3.5-lite'):
        self.api_key = api_key
        self.model = model

    def generate_embedding(self, text: str, dimensions: int = 256) -> List[float]:
        """Matryoshka圧縮対応のEmbedding生成

        Args:
            text: 入力テキスト
            dimensions: 出力次元数（256, 512, 1024, 2048）
        """
        # Voyage API呼び出し
        import voyageai
        client = voyageai.Client(api_key=self.api_key)
        result = client.embed(
            texts=[text],
            model=self.model,
            output_dimension=dimensions
        )
        return result.embeddings[0]

# Qdrant検索クライアント
class QdrantSearchClient:
    def __init__(self, url: str, api_key: str, collection: str = 'improve-issues'):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection

    def search_similar_issues(self, query_vector: List[float], limit: int = 3) -> List[Dict]:
        """Top-K類似Issue検索"""
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit
        )
        return results
```

#### プロンプト拡張（類似Issue参考）

```python
def get_improve_prompt_with_rag(
    template_name: str,
    issue_body: str,
    issue_title: str,
    similar_issues: List[Dict]
) -> str:
    """類似Issue情報を含むプロンプト"""

    # 類似Issue情報を整形
    similar_info = ""
    for i, issue in enumerate(similar_issues, 1):
        similar_info += f"""
【参考Issue {i}】
- タイトル: {issue['issue_title']}
- 本文抜粋: {issue['issue_body'][:200]}...
- 類似度: {issue['similarity']:.1%}
"""

    base_prompt = get_improve_prompt(template_name, issue_body, issue_title)

    return f"""{base_prompt}

【参考情報】
以下の過去Issueを参考にしてください：
{similar_info}

上記の参考Issueから、記述スタイルや必要な情報項目を学び、より具体的で実用的な例文を生成してください。
"""
```

#### コメント形式拡張

```python
def format_comment_with_references(
    template_name: str,
    improved_content: str,
    similar_issues: List[Dict]
) -> str:
    """類似Issue情報を含むコメント"""

    references = ""
    for i, issue in enumerate(similar_issues, 1):
        references += f"""
{i}. **#{issue['issue_number']}: {issue['issue_title']}** ({issue['state']})
   - 類似度: {issue['similarity']:.0%}
   - {issue['url']}
   - 類似点: （生成された例文で参考にした箇所）
"""

    return f"""## 🤖 AIによるIssue記入例

**選定テンプレート**: {template_name}

---

{improved_content}

---

### 📚 参考にした類似Issue

この例文は以下の過去Issueを参考に生成しています：
{references}

---

💡 **使い方**: 上記の例文を参考に、Issue本文を編集してください。類似Issueも確認すると、より詳細な情報が得られます。
"""
```

### Phase 3: 複数テンプレート対応

Phase 1-2で実装済みのため、プロンプトチューニングと精度向上に注力。

### Phase 4: 運用改善・品質向上

#### Issue更新時の再生成

```yaml
on:
  issues:
    types: [opened, edited]  # editedイベントを追加
```

```python
# 既存コメント検索と更新
def find_bot_comment(issue_number: int) -> Optional[int]:
    """Bot投稿のコメントIDを取得"""
    comments = github.rest.issues.list_comments(
        owner=context.repo.owner,
        repo=context.repo.repo,
        issue_number=issue_number
    )

    for comment in comments.data:
        if comment.user.login == 'github-actions[bot]' and '🤖 AIによるIssue記入例' in comment.body:
            return comment.id

    return None

def update_or_create_comment(issue_number: int, content: str):
    """コメント更新または新規作成"""
    comment_id = find_bot_comment(issue_number)

    if comment_id:
        # 既存コメントを更新
        github.rest.issues.update_comment(
            owner=context.repo.owner,
            repo=context.repo.repo,
            comment_id=comment_id,
            body=content
        )
    else:
        # 新規コメント作成
        github.rest.issues.create_comment(
            owner=context.repo.owner,
            repo=context.repo.repo,
            issue_number=issue_number,
            body=content
        )
```

#### フィードバック機構

```python
# コメントにリアクション検知用のマーカーを追加
footer = """
---

👍 この例文が役に立った / 👎 改善が必要

※ リアクションは品質改善の参考にさせていただきます
"""
```

#### プロンプトA/Bテスト

```python
# 設定ファイルで複数プロンプトバージョンを管理
PROMPT_VERSIONS = {
    'v1': FEATURE_1_TEMPLATE,
    'v2': FEATURE_1_TEMPLATE_V2,  # 改善版
}

def select_prompt_version(issue_number: int) -> str:
    """Issue番号に基づいてプロンプトバージョンを選択（A/Bテスト）"""
    return 'v1' if issue_number % 2 == 0 else 'v2'
```

## セキュリティ設計

### Secrets管理

**必要なSecrets:**

```
# Phase 0-1
GEMINI_API_KEY          # Gemini API (LLM用)

# Phase 2以降
QDRANT_URL              # Qdrant CloudのURL
QDRANT_API_KEY          # Qdrant API Key
VOYAGE_API_KEY          # Voyage AI Embedding API Key

# Optional (Phase 2代替)
GEMINI_EMBEDDING_API_KEY  # Google Gemini Embedding-004 (PoC用)
```

**設定方法:**
1. GitHub Repository → Settings → Secrets and variables → Actions
2. 各Secretを登録

### ログ出力制限

```python
# API Keyをログに出力しない
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NG: logger.info(f"API Key: {api_key}")
# OK: logger.info("API Key configured")
```

## モニタリング・運用

### メトリクス

- Workflow実行回数（日次・月次）
- 実行時間の平均・最大
- API呼び出し成功率
- コスト（API利用料）
- ユーザーフィードバック（👍/👎）

### アラート設定

- Workflow失敗率が50%超
- API呼び出しエラー率が30%超
- 実行時間が5分超

## コスト見積もり（2025年11月更新）

### Phase 0-1: 基本機能のみ

| 項目 | 単価 | 想定利用量 | 月額 |
|------|------|-----------|------|
| Gemini 2.0 Flash-Lite | $0.075/1M入力, $0.30/1M出力 | 15リクエスト | 約1円 |
| GitHub Actions | 無料枠2000分 | 30分/月 | $0 |
| **合計** | - | - | **約1円/月** |

### Phase 2: RAG機能追加後（推奨構成）

**Voyage 3.5-lite + Truncation 256次元**

※ Truncation: 外部APIで生成された1024次元ベクトルを、GitHub Actions内で256次元に切り詰め（100msec未満）

| 項目 | 単価 | 想定利用量 | 月額 |
|------|------|-----------|------|
| Gemini 2.5 Flash | $0.30/1M入力, $2.50/1M出力 | 15回 × 800T入力/2000T出力 | 約7.5円 |
| Voyage 3.5-lite | $0.02/1M | 15回 × 300T | 約0.009円 |
| Qdrant Cloud | 無料プラン | 15Issue × 256dim × 4byte ≈ 15KB | $0 |
| GitHub Actions | $0.008/分 | 15回 × 0.33分 (20秒) | 約0.64円 |
| GitHub Actions無料枠 | 2000分/月 | 5分/月 | **$0** (無料枠内) |
| **合計** | - | - | **約8円/月** |

**PoC代替構成: Gemini Embedding-004（無料）**

| 項目 | 単価 | 想定利用量 | 月額 |
|------|------|-----------|------|
| Gemini 2.5 Flash | $0.30/1M入力, $2.50/1M出力 | 15回 × 800T/2000T | 約7.5円 |
| Gemini Embedding-004 | **無料枠** | 15回 | **$0** |
| **合計** | - | - | **約8円/月** |

### GitHub Actionsリソース消費実態（Phase 2）

**処理フロー**:
```
1. Issue取得 (100msec)
   ↓
2. Embedding API呼び出し → Voyage AI (外部、1-2秒)
   ↓
3. ローカルTruncation処理 (50-100msec、CPU 50%、メモリ 50MB)
   ↓
4. Qdrant検索 → Qdrant Cloud (外部、1-2秒)
   ↓
5. LLM API呼び出し → Gemini (外部、5-10秒)
```

**リソース実測値**:

| 処理 | 実行時間 | CPU使用率 | メモリ使用量 |
|------|----------|------------|-------------|
| Issue取得 | 100msec | 5% | 20MB |
| Embedding API呼び出し | 1-2秒 | 5% (待機) | 30MB |
| **Truncation処理** | 50-100msec | **50%** | **50MB** |
| Qdrant検索 API | 1-2秒 | 5% (待機) | 30MB |
| LLM API呼び出し | 5-10秒 | 10% (待機) | 50MB |
| **合計** | **15-20秒** | **平均10%** | **ピーク180MB** |

**判定**:
- ubuntu-latest (2コア7GB) で**十分すぎる余裕**
- ubuntu-slim (1コア5GB) でも**問題なく実行可能**
- 処理のほとんどがAPI呼び出し待機のため、CPU/メモリ消費は極めて軽量

## リリース計画

### Phase 0（技術検証）: Week 1-3

- Week 1: 環境構築・基本動作確認
- Week 2: 品質評価・改善
- Week 3: エラーハンドリング・ドキュメント

### Phase 1（基本機能）: Week 4-5

- Week 4: テンプレート判定強化、重複投稿防止
- Week 5: エラーハンドリング強化、テスト追加

### Phase 2（RAG導入）: Week 6-8

**Week 6: Embedding選定とベンチマーク**
- [ ] Google Gemini Embedding-004でPoC実施（無料）
- [ ] Voyage 3.5-lite vs OpenAI 3-smallの精度比較
- [ ] Matryoshka圧縮（1024→56次元）の精度影響測定
- [ ] ベンチマーク目標: Recall@3 ≥80%, Precision@5 ≥70%

**Week 7: Qdrantセットアップ、RAG検索実装**
- [ ] Qdrant Cloudセットアップ
- [ ] コレクション設計（256次元ベクトル、コサイン類似度）
- [ ] 類似Issue検索ロジック実装
- [ ] プロンプト統合（類似Issue情報参照）

**Week 8: 品質評価・最適化**
- [ ] RAG検索精度測定（Recall@3, Precision@5）
- [ ] LLM生成品質評価（類似Issue参照有無で比較）
- [ ] Optional: Reranker導入検討（Cohere Rerank等）
- [ ] レイテンシ測定（目標: 2秒以下）

### Phase 3（複数テンプレート）: Week 9

- プロンプト最適化、精度向上

### Phase 4（運用改善）: Week 10-12

- Issue更新対応、フィードバック機構
- モニタリング設定、ドキュメント整備

## 関連ドキュメント

- 要件書: `.dev/requirements/improve_issue.md`
- 技術検証: `.dev/designs/issue_auto_improve_技術検証.md`
- GitHub Issue Template: `.github/ISSUE_TEMPLATE/`

## 備考

- Phase 0の技術検証結果に基づき、LLM選定やアーキテクチャを最終決定
- RAG機能はオプション扱いとし、Phase 1でも十分な価値提供を目指す
- コスト超過時は手動トリガーへの変更を検討
