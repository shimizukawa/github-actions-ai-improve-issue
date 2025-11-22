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

### ローカル検証（開発者向け）

```bash
# 環境変数設定

export ISSUE_TITLE="Mermaid図の拡大縮小機能"
export ISSUE_BODY="ページ編集 でMermaid図の部分だけ別ウインドウで表示して拡大縮小したい。ページ全体の編集ではプレビューの縦位置がずれてしまうため使いづらい"
export ISSUE_NUMBER="123"
export LLM_API_KEY="your-gemini-api-key"

# dry-runモードで実行（コメント投稿なし、コンソールに出力）
uv run .github/scripts/improve_issue.py --dry-run
```

## テンプレート判定

Issue本文とタイトルから、以下のテンプレートを自動判定します：

| テンプレート | キーワード例 | 用途 |
|------------|------------|------|
| feature_request | 機能、追加、変更、改善、したい | 機能要件 |
| bug_report | バグ、エラー、不具合、動かない | バグ報告 |

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
README.md                       # this file
.github/
├── ISSUE_TEMPLATE/             # templates
│   ├── bug_report.md
│   └── feature_request.md
├── scripts/
│   └── improve_issue.py        # main script
└── workflows/
	└── issue_auto_improve.yml  # GitHub Actions Workflow
```

### 依存関係

PEP-723に従い、スクリプト内で依存関係を定義：

```python
# /// script
# dependencies = [
#   "some-package==0.0.0",
# ]
# ///
```

`uv run` コマンドで実行すると、自動的に依存関係のインストール後に実行されます。

### 実装モード

スクリプトは以下の実行モードをサポート：

1. **通常モード**: GitHub Actionsから実行、コメント投稿、RAG検索（環境変数設定時）
2. **--dry-run**: ローカル検証用、コメント投稿スキップ
3. **--index-issues**: 全Issue一括インデックス作成（初回セットアップ用）
4. **--update-single-issue N**: 単一Issue更新（Issue番号Nを指定）

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
uv run .github/scripts/improve_issue.py --index-issues

# 範囲指定も可能
uv run .github/scripts/improve_issue.py --index-issues --start 1 --end 100
```

#### 単一Issue更新

```bash
# Issue番号123を更新
uv run .github/scripts/improve_issue.py --update-single-issue 123
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
