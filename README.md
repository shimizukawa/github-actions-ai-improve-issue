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
- `ai-processing`, `ai-processed` ラベルがついていない

## 使い方

### 通常の利用（GitHub Actions）

1. リポジトリにIssueを作成
2. タイトルと1-2行の簡単な説明を記入
3. 自動的にGitHub Actions Workflowが起動
4. AIが生成した例文がコメントとして投稿される
5. 例文を参考に、Issue本文を編集

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

## Phase 1の実装範囲

現在はPhase 1が実装されています：

- ✅ Issue作成時の自動起動
- ✅ テンプレート自動判定（キーワードベース）
- ✅ LLMによる例文生成（Gemini 2.5 Flash）
- ✅ Issueへのコメント投稿
- ✅ エラーハンドリング
- ❌ RAG機能（過去Issue参照）は未実装（Phase 2で実装予定）

## 設定

### 必要なGitHub Secrets

- `GEMINI_API_KEY`: Google Gemini APIキー

### ラベルによる制御

- `ai-processing` ラベル: 処理中
- `ai-processed` ラベル: 処理完了、自動処理をスキップ

## トラブルシューティング

### Workflowが起動しない

- Issue本文が10文字以上あるか確認
- `ai-processing`, `ai-processed` ラベルがついていないか確認

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
#   "google-generativeai>=0.8.3",
# ]
# ///
```

`uv run` コマンドで実行すると、自動的に依存関係のインストール後に実行されます。

### 実装モード

スクリプトは以下の実行モードをサポート：

1. **通常モード**: GitHub Actionsから実行、コメント投稿
2. **--dry-run**: ローカル検証用、コメント投稿スキップ
3. **--index-issues**: RAGデータ生成（Phase 2で実装予定）

## 今後の予定（Phase 2以降）

- [ ] RAG機能追加（過去Issue参照）
- [ ] Qdrant Cloud連携
- [ ] 類似Issue検索
- [ ] Issue更新時の再生成
- [ ] フィードバック機構

## 関連ドキュメント

- 要件書: `.dev/requirements/improve_issue.md`
- 設計書: `.dev/designs/issue_auto_improve_設計.md`
- 技術検証: `.dev/designs/issue_auto_improve_技術検証.md`
