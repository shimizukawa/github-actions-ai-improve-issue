# Issue自動ブラッシュアップ機能

簡易なブランクIssueに1-2行程度のやりたいことを書いたら、適切なテンプレートに沿った具体的な文章を自動生成し、コメントで提示する機能です。

## 概要

### 目的

- Issue作成の心理的ハードルを下げる
- テンプレート選択の自動化
- 具体的な例文による記述支援
- Issue品質の底上げ

### 対象Issue

以下の条件を満たすIssueが処理対象となります：

- タイトルと本文（空白除く）の合計が10文字以上
- ユーザーが `ai-improve` ラベルを付与したIssueであること

## 使い方

### 通常の利用（GitHub Actions）

1. リポジトリにIssueを作成
2. タイトルと1-2行の簡単な説明を記入
3. 「このIssueにAI補助をかけたい」と判断したら、Issueに `ai-improve` ラベルを付与
4. `ai-improve` ラベル付与をトリガーにGitHub Actions Workflowが起動
5. AIが生成した例文がコメントとして投稿される
6. 例文を参考に、Issue本文を編集
7. 処理完了後、Workflowが `ai-improve` ラベルを自動で削除（再度ラベルを付ければ再実行可能）

### CLIからの実行
```bash
# インストールして実行
$ pip install github-actions-ai-improve-issue
$ improve-issue

# GitHubから直接実行
uvx --from git+https://github.com/shimizukawa/github-actions-ai-improve-issue improve-issue
```


### ローカル検証（開発者向け）

```bash
# 環境変数設定

export ISSUE_TITLE="Mermaid図の拡大縮小機能"
export ISSUE_BODY="ページ編集 でMermaid図の部分だけ別ウインドウで表示して拡大縮小したい。ページ全体の編集ではプレビューの縦位置がずれてしまうため使いづらい"
export ISSUE_NUMBER="123"
export LLM_API_KEY="your-gemini-api-key"

# dry-runモードで実行（コメント投稿なし、コンソールに出力）
uv run -m github_actions_ai_improve_issue --dry-run
```

## 設定ファイル

### 必須設定

設定ファイルが必須です。設定ファイルが見つからない場合はエラーで終了します。

設定ファイルの配置場所:
1. 環境変数 `IMPROVE_ISSUE_CONFIG` で指定したパス（優先）
2. リポジトリルート直下の `.improve_issue.yml`（デフォルト）

### 設定ファイルの構造

```yaml
default_template: feature_request

templates:
  feature_request:
    issue_template_file: feature_request
    system_prompt: |
      あなたはプロジェクト管理の専門家です。以下のIssue記述を、機能要件テンプレートに沿った具体的で詳細な内容に拡張してください。
      
      【重要な指示】
      - 抽象的な表現を避け、具体的に記述してください
      - Issue記述から推測できる範囲で詳細化してください
      - 不明な点は「要確認」として明示してください
      - Markdown形式で出力してください
      - 各項目は箇条書きで、少なくとも2-3項目記述してください
    keywords:
      - 機能
      - 追加
      - 変更
      - 改善
      - したい
      - 欲しい
      - 必要

  bug_report:
    issue_template_file: bug_report
    system_prompt: |
      あなたはソフトウェアテストの専門家です。以下のバグ報告を、詳細で再現可能な形式に拡張してください。
      
      【重要な指示】
      - 再現手順を具体的に記述してください
      - エラーメッセージやスクリーンショットの必要性を明示してください
      - Markdown形式で出力してください
    keywords:
      - バグ
      - エラー
      - 不具合
      - 動かない
      - 失敗
      - 問題
```

設定項目:
- `default_template`: テンプレート判定でキーワードマッチしない場合のデフォルトテンプレート名
- `templates`: テンプレート定義の辞書
  - `issue_template_file`: `.github/ISSUE_TEMPLATE/` 配下のテンプレートファイル名（拡張子 `.md` は不要）
  - `system_prompt`: LLM に渡すシステムプロンプト（テンプレートごとにカスタマイズ可能）
  - `keywords`: テンプレート判定に使用するキーワードリスト（日本語・英語対応）

### テンプレートのカスタマイズ

テンプレートを追加・変更する場合:
1. `.improve_issue.yml` にテンプレート定義を追加
2. `.github/ISSUE_TEMPLATE/` に対応する Issue テンプレートファイルを配置

## テンプレート判定

Issue本文とタイトルから、設定ファイルで定義されたテンプレートを自動判定します。
キーワードマッチ数が最も多いテンプレートが選択されます。
キーワードがマッチしない場合は `default_template` が使用されます。

デフォルト設定では、日本語と英語の両方のキーワードに対応しています。

### Phase 2の実装範囲

現在はPhase 2が実装されています：

### 基本機能（Phase 1）
- ✅ `ai-improve` ラベル付与時の起動（`issues.labeled`）
- ✅ テンプレート自動判定（キーワードベース）
- ✅ LLMによる例文生成（Gemini 2.5 Flash）
- ✅ Issueへのコメント投稿
- ✅ エラーハンドリング

### RAG機能（Phase 2）
- ✅ Voyage AI 3.5-lite Embedding生成
- ✅ Qdrant Cloud連携
- ✅ 類似Issue検索（Top-3）
- ✅ 検索結果を含む例文生成
- ✅ 参考Issue情報の表示
- ✅ Issue作成時・編集時・状態変更時の自動インデックス登録/更新

## 設定

### 必要なGitHub Secrets

#### 必須（Phase 1機能）
- `GEMINI_API_KEY`: Google Gemini APIキー

#### オプション（Phase 2 RAG機能）
以下のSecretsを設定すると、RAG機能が有効になります。未設定の場合はPhase 1モード（RAG未使用）で動作します。

- `QDRANT_URL`: Qdrant CloudのURL
- `QDRANT_API_KEY`: Qdrant APIキー
- `VOYAGE_API_KEY`: Voyage AI APIキー

### ラベルによる制御

- `ai-improve` ラベル: AIによるブラッシュアップを依頼するためのトリガー。付与時に提案コメント生成を行い、処理完了後に自動で外れる

## トラブルシューティング

### Workflowが起動しない

- Issue本文が10文字以上あるか確認
- Issueに `ai-improve` ラベルが正しく付与されているか確認

### エラーコメントが投稿される

- GitHub Secrets に `GEMINI_API_KEY` が正しく設定されているか確認
- Gemini APIの利用制限に達していないか確認

## 開発者向け情報

### ファイル構成

```
.improve_issue.yml              # 設定ファイル（必須）
README.md                       # this file
.github/
├── ISSUE_TEMPLATE/             # templates
│   ├── bug_report.md
│   └── feature_request.md
└── workflows/
    └── issue_auto_improve.yml  # GitHub Actions Workflow
src/
└── github_actions_ai_improve_issue/
    ├── __init__.py
    └── main.py                 # メインスクリプト
```

### 実行コマンド

スクリプトは以下の実行モードをサポート：

1. **通常モード**: GitHub Actionsから実行、コメント投稿、RAG検索（環境変数設定時）
   ```bash
   uv run -m github_actions_ai_improve_issue
   ```
2. **--dry-run**: ローカル検証用、コメント投稿スキップ
   ```bash
   uv run -m github_actions_ai_improve_issue --dry-run
   ```
3. **--index-issues**: 全Issue一括インデックス作成（初回セットアップ用）
   ```bash
   uv run -m github_actions_ai_improve_issue --index-issues
   ```
4. **--update-single-issue N**: 単一Issue更新（Issue番号Nを指定）
   ```bash
   uv run -m github_actions_ai_improve_issue --update-single-issue 123
   ```

### RAGインデックス管理

#### 初回セットアップ（全Issue一括インデックス）

```bash
# 環境変数設定
export GITHUB_TOKEN="your-github-token"
export GITHUB_REPOSITORY="owner/repo"
export QDRANT_URL="https://xxx.qdrant.io"
export QDRANT_API_KEY="your-qdrant-api-key"
export VOYAGE_API_KEY="your-voyage-api-key"

# 全Issueをインデックス
uv run -m github_actions_ai_improve_issue --index-issues

# 範囲指定も可能
uv run -m github_actions_ai_improve_issue --index-issues --start 1 --end 100
```

#### 単一Issue更新

```bash
# Issue番号123を更新
uv run -m github_actions_ai_improve_issue --update-single-issue 123
```

#### 自動更新

GitHub Actionsにより、以下のタイミングで自動的にインデックスが更新されます：

- Issue編集時: 自動更新（タイトル・本文の変更を反映）

## 今後の予定（Phase 3以降）

- [ ] Issue本文更新時の再生成機能
- [ ] 生成コメントの更新（上書き）機能
- [ ] フィードバック機構（リアクション検知）
- [ ] プロンプトA/Bテスト基盤

## 関連ドキュメント

- 要件書: `.dev/requirements/improve_issue.md`
- 設計書: `.dev/designs/issue_auto_improve_設計.md`
- 技術検証: `.dev/designs/issue_auto_improve_技術検証.md`
