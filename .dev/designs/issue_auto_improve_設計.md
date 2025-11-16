# 設計書: Issue自動ブラッシュアップ機能

## 概要

簡易なブランクIssueに1-2行程度のやりたいことを書いたら、過去の類似Issue情報を参考に適切なテンプレートに沿った具体的な文章を自動生成し、コメントで提示する機能を実装する。Phase 0の技術検証を経て、段階的に機能を拡張していく。

## 親Issue

- （要件Issue番号を記載予定）

## アーキテクチャ
## Phase別詳細設計

### Phase 1: 現行実装（2025年11月時点）

#### Workflow設計

`issue_auto_improve.yml` は `issues.opened` をトリガーとし、`ai-processing` / `ai-processed` のラベルがあるIssueをスキップしながら、`uv run` スクリプトを実行します。成功時には `ai-processed` を付与し、失敗時には `ai-processing` のみを削除して再実行可能な状態を保ちます。

#### スクリプト設計

`improve_issue.py` は PEP-723 を使い `google-generativeai>=0.8.3` を明示し、以下の構成で処理します。

1. `.github/ISSUE_TEMPLATE/{feature_request,bug_report}.md` からテンプレート本文を `load_template_content` で読み込み（frontmatter はスキップ）
2. `TemplateDetector` がタイトル/本文のキーワードに対してスコアをつけ、`feature_request` / `bug_report` を選定（スコアゼロ時は `feature_request`）
3. **RAG環境変数チェック**（Phase 2）:
   - `QDRANT_URL`, `QDRANT_API_KEY`, `VOYAGE_API_KEY`（または `GEMINI_API_KEY`）の存在を確認
   - **全て揃っている場合**: RAG検索を実行し、類似Issue情報を取得
   - **不足している場合**: RAGをスキップし、Phase 1モード（RAGなし）で動作
   - ログに動作モードを出力（`[INFO] RAG mode: enabled` / `[INFO] RAG mode: disabled (missing: QDRANT_URL)`）
4. `get_improve_prompt` または `get_improve_prompt_with_rag` でプロンプトを構築
5. `LLMClient`（`gemini-2.5-flash`）で `generate` を呼び出し、出力を `format_comment` で整形
6. `post_comment_via_gh` が `gh issue comment` で Issue に投稿し、`ai-processing` → `ai-processed` のラベル遷移を行う
7. `--dry-run` モードではコメント投稿をスキップし、結果を標準出力に表示

#### ローカル検証モード

- `uv run .github/scripts/improve_issue.py --dry-run` でコメント投稿をスキップ
- ローカルでも `ISSUE_TITLE`, `ISSUE_BODY`, `ISSUE_NUMBER`, `LLM_API_KEY`, `GITHUB_TOKEN` を環境変数として設定して本番と同じ流れを検証可能


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

**インデックス更新方針:**

1. **イベント駆動型更新**: Issue作成・編集・コメント投稿時に自動更新
2. **初回インデックス**: CLIコマンドで既存Issue一括登録
3. **定期同期不要**: イベント駆動で常に最新状態を維持

**初回インデックス作成(CLI実行):**

```bash
# 既存Issue全体の初回インデックス作成
uv run .github/scripts/improve_issue.py --index-issues

# 範囲指定も可能(オプション)
uv run .github/scripts/improve_issue.py --index-issues --issue-range 1-100
```

**イベント駆動型インデックス更新:**

```yaml
# .github/workflows/issue_auto_improve.yml に統合
# Issue作成時: RAG検索 + 例文生成 + インデックス登録を同時実行
# Issue編集時: インデックス更新のみ

on:
  issues:
    types: [opened, edited, closed, reopened]
  issue_comment:
    types: [created, edited]

jobs:
  # Issue作成時: 例文生成 + インデックス登録
  improve-issue:
    if: github.event.action == 'opened'
    # ... (既存処理) ...
    # 処理完了後にインデックス登録

  # Issue更新時: インデックス更新のみ
  update-index:
    if: |
      github.event.action == 'edited' ||
      github.event.action == 'closed' ||
      github.event.action == 'reopened' ||
      github.event_name == 'issue_comment'
    runs-on: ubuntu-slim
    timeout-minutes: 1
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          version: "latest"

      - name: Update Qdrant index
        run: |
          uv run .github/scripts/improve_issue.py --update-single-issue ${{ github.event.issue.number }}
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          QDRANT_URL: ${{ secrets.QDRANT_URL }}
          QDRANT_API_KEY: ${{ secrets.QDRANT_API_KEY }}
          VOYAGE_API_KEY: ${{ secrets.VOYAGE_API_KEY }}
```

**RAG・インデックス更新フロー:**

Phase 2以降で、`improve_issue.py` に以下の機能を追加統合：

```python
# RAG環境変数チェック
def check_rag_availability() -> tuple[bool, list[str]]:
    """RAG機能の利用可否を判定
    
    Returns:
        (is_available, missing_vars): RAG利用可能か、不足している環境変数のリスト
    """
    required_vars = ['QDRANT_URL', 'QDRANT_API_KEY']
    # Embedding APIはVOYAGE_API_KEYまたはGEMINI_API_KEYのいずれか
    has_embedding_api = os.getenv('VOYAGE_API_KEY') or os.getenv('GEMINI_API_KEY')
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if not has_embedding_api:
        missing.append('VOYAGE_API_KEY or GEMINI_API_KEY')
    
    return (len(missing) == 0, missing)

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

### Phase 2: RAG機能追加
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

