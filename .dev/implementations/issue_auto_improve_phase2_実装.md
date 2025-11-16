# Phase 2 RAG機能実装

## 対応内容

Phase 2のRAG（Retrieval-Augmented Generation）機能を実装しました。

### 対応範囲

- RAG環境変数のチェック機能（不足時はPhase 1モードで動作）
- Voyage AI 3.5-lite Embeddingクライアント実装
- Qdrant Cloud検索クライアント実装
- `--index-issues` コマンド実装（既存Issue一括インデックス）
- `--update-single-issue` コマンド実装（単一Issue更新）
- Issue作成時の自動インデックス登録
- 類似Issue検索機能（Top-3）
- RAG情報を含むプロンプト拡張
- 参考Issue情報を含むコメント形式拡張
- GitHub Actions Workflow更新（edited, closed, reopened, issue_comment イベント対応）
- ドキュメント更新（README）
- .gitignore追加

### 対応範囲外

- Issue本文更新時の再生成機能（Phase 4で実装予定）
- 生成コメントの更新（上書き）機能（Phase 4で実装予定）
- フィードバック機構（Phase 4で実装予定）

## 実装詳細

### 1. RAG環境変数チェック機能

**ファイル**: `.github/scripts/improve_issue.py`

```python
def check_rag_available() -> tuple[bool, str]:
    """RAG機能が利用可能かチェック
    
    Returns:
        (available, reason): 利用可能かどうかと理由
    """
```

- RAGライブラリ（voyageai, qdrant-client）のインストール状態をチェック
- 必要な環境変数（QDRANT_URL, QDRANT_API_KEY, VOYAGE_API_KEY）の存在をチェック
- 不足している場合は理由を返し、Phase 1モードで動作

### 2. Voyage AI Embeddingクライアント

**ファイル**: `.github/scripts/improve_issue.py`

```python
class VoyageEmbeddingClient:
    def __init__(self, api_key: str, model: str = "voyage-3.5-lite")
    def generate_embedding(self, text: str, dimensions: int = 256) -> List[float]
```

- Voyage AI 3.5-lite モデルを使用
- 256次元のEmbeddingベクトルを生成（Matryoshka圧縮でコスト最適化）
- Issue本文とタイトルを結合してベクトル化

### 3. Qdrant検索クライアント

**ファイル**: `.github/scripts/improve_issue.py`

```python
class QdrantSearchClient:
    COLLECTION_NAME = "improve-issues"
    
    def __init__(self, url: str, api_key: str)
    def ensure_collection(self, vector_size: int = 256)
    def search_similar_issues(self, query_vector: List[float], limit: int = 3) -> List[Dict[str, Any]]
    def upsert_issue(self, issue_number: int, vector: List[float], ...)
```

- Qdrant Cloudに接続
- コレクション自動作成機能
- コサイン類似度による類似Issue検索（Top-K）
- Issue情報のインデックス登録・更新

**ペイロード構造**:
```python
{
    "issue_number": int,
    "issue_title": str,
    "issue_body": str[:1000],  # 最初の1000文字
    "template_type": str,
    "state": str,  # open/closed
    "url": str,
    "labels": List[str]
}
```

### 4. GitHub API統合

**ファイル**: `.github/scripts/improve_issue.py`

```python
def fetch_issue_from_github(issue_number: int, github_token: str) -> Optional[Dict]
def fetch_all_issues(github_token: str, start: int = 1, end: Optional[int] = None) -> List[Dict]
```

- `gh` CLIを使用してIssue情報を取得
- 単一Issue取得および全Issue取得に対応
- 範囲指定機能（--start, --end オプション）

### 5. インデックス管理コマンド

#### --index-issues（一括インデックス作成）

```python
def index_all_issues(start: int = 1, end: Optional[int] = None):
```

- 既存の全Issueを一括でインデックス登録
- 範囲指定可能（--start, --end）
- 初回セットアップ時に使用

**使用例**:
```bash
export GITHUB_TOKEN="..."
export GITHUB_REPOSITORY="owner/repo"
export QDRANT_URL="..."
export QDRANT_API_KEY="..."
export VOYAGE_API_KEY="..."

uv run .github/scripts/improve_issue.py --index-issues
uv run .github/scripts/improve_issue.py --index-issues --start 1 --end 100
```

#### --update-single-issue（単一Issue更新）

```python
def update_single_issue(issue_number: int):
```

- 指定したIssue番号のインデックスを更新
- Issue編集時のGitHub Actions Workflowから自動実行

**使用例**:
```bash
uv run .github/scripts/improve_issue.py --update-single-issue 123
```

### 6. RAG対応プロンプト拡張

**ファイル**: `.github/scripts/improve_issue.py`

```python
def get_improve_prompt(
    template_name: str,
    issue_body: str,
    issue_title: str = "",
    similar_issues: Optional[List[Dict[str, Any]]] = None,
) -> str:
```

- 類似Issue情報をプロンプトに追加
- 各Issue情報（タイトル、本文抜粋、類似度）を含める
- LLMに参考Issueから学習するよう指示

### 7. RAG対応コメント形式

**ファイル**: `.github/scripts/improve_issue.py`

```python
def format_comment(
    improved_content: str,
    template_name: str,
    similar_issues: Optional[List[Dict[str, Any]]] = None,
) -> str:
```

- 類似Issue情報をコメントに追加
- Issue番号、タイトル、類似度、URLを表示
- 参考Issue確認を促すメッセージ追加

### 8. メイン処理フロー

**ファイル**: `.github/scripts/improve_issue.py`

```python
def main():
```

**フロー**:
1. コマンドライン引数解析
2. `--index-issues` モード: 全Issue一括インデックス → 終了
3. `--update-single-issue` モード: 単一Issue更新 → 終了
4. 通常モード（Issue改善）:
   - RAG機能チェック
   - RAG有効時: 類似Issue検索
   - LLMで例文生成（RAG情報含む）
   - コメント投稿
   - RAG有効時: 現在のIssueをインデックス登録

### 9. GitHub Actions Workflow更新

**ファイル**: `.github/workflows/issue_auto_improve.yml`

#### improve-issue ジョブ（Issue作成時）

- トリガー: `issues.opened`
- RAG環境変数を追加（未設定時もエラーとしない）
- 処理: RAG検索 → 例文生成 → コメント投稿 → インデックス登録

#### update-index ジョブ（Issue更新時）

- トリガー: `issues.edited`, `issues.closed`, `issues.reopened`, `issue_comment.created`
- `--update-single-issue` コマンドを実行
- `continue-on-error: true` でRAG未設定時もエラーとしない

### 10. ドキュメント更新

**ファイル**: `README.md`

- Phase 2実装範囲の追加
- RAG機能の説明
- オプションのGitHub Secrets説明
- RAGインデックス管理方法の追加
- コマンド使用例の追加

**ファイル**: `.gitignore`

- Python関連ファイル（__pycache__等）を除外

## 技術的な注意点

### フォールバック機能

- RAG環境変数が不足している場合、Phase 1モード（RAG未使用）で動作
- RAGライブラリ未インストール時も同様にPhase 1モードで動作
- RAG検索失敗時もフォールバックし、処理を継続

### Embeddingベクトル次元

- Voyage AI 3.5-lite: 1024次元をサポート
- 256次元に圧縮してQdrantに保存（コスト最適化）
- Matryoshka Embeddings対応により、精度を保ちながら圧縮可能

### インデックス更新タイミング

**自動更新**:
- Issue作成時: 例文生成後に自動登録
- Issue編集時: `update-index` ジョブで自動更新
- Issue状態変更時（closed/reopened）: 自動更新
- コメント投稿時: 自動更新

**手動更新**:
- `--index-issues`: 初回セットアップ時
- `--update-single-issue N`: 特定Issueの再インデックス

### エラーハンドリング

- RAG処理失敗時もPhase 1モードで処理継続
- `continue-on-error: true` でWorkflow全体の失敗を回避
- エラーメッセージをログに出力

### コスト最適化

- Embedding次元数を256次元に削減（75%削減）
- Qdrant Cloud無料プラン利用（1GB、100万ベクトル）
- Voyage AI 3.5-lite使用（$0.02/100万トークン）
- GitHub Actions無料枠内で実行可能

## 制限事項

### 現時点での制限

- 類似Issue検索はTop-3に固定
- Issue本文は最初の1000文字のみをインデックス保存
- コメント内容はインデックス対象外（Issue本文のみ）
- テンプレート判定はキーワードベースのみ

### Phase 3以降で対応予定

- Issue本文更新時の再生成機能
- 生成コメントの自動更新（上書き）
- フィードバック機構（リアクション検知）
- プロンプトA/Bテスト基盤

## 動作確認事項

### 必須確認項目

- [ ] RAG環境変数未設定時にPhase 1モードで正常動作するか
- [ ] RAG環境変数設定時に類似Issue検索が動作するか
- [ ] `--index-issues` コマンドで既存Issue一括登録が動作するか
- [ ] `--update-single-issue` コマンドで単一Issue更新が動作するか
- [ ] Issue編集時にインデックス更新が動作するか
- [ ] コメント投稿時にインデックス更新が動作するか

### 推奨確認項目

- [ ] 類似Issue検索結果の精度確認
- [ ] RAG情報を含む例文の品質確認
- [ ] Qdrant Cloud無料プランの容量確認
- [ ] GitHub Actions実行時間の確認
- [ ] エラーログの確認

## 関連ファイル

- `.github/scripts/improve_issue.py`: メインスクリプト
- `.github/workflows/issue_auto_improve.yml`: GitHub Actions Workflow
- `README.md`: ユーザー向けドキュメント
- `.dev/requirements/improve_issue.md`: 要件定義
- `.dev/designs/issue_auto_improve_設計.md`: 設計書
- `.gitignore`: Git除外設定
