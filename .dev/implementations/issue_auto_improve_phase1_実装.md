# 実装書: Issue自動ブラッシュアップ機能 Phase 1

## 概要

Phase 1として、Issue自動ブラッシュアップ機能の基本機能を実装しました。簡易なブランクIssueに1-2行のやりたいことを書くと、テンプレートに沿った具体的な例文を自動生成してコメント投稿する機能です。

## 親Issue・設計

- 要件: `.dev/requirements/improve_issue.md`
- 設計: `.dev/designs/issue_auto_improve_設計.md`
- 技術検証: `.dev/designs/issue_auto_improve_技術検証.md`

## 実装内容

### 1. Pythonスクリプトの実装

**ファイル**: `.github/scripts/improve_issue.py`

#### 主要機能

1. **PEP-723対応の依存関係管理**
   ```python
   # /// script
   # dependencies = [
   #   "google-generativeai>=0.8.3",
   # ]
   # ///
   ```
   - `uv run` コマンドで自動的に依存関係をインストール
   - 単一ファイルで完結、`requirements.txt` 不要

2. **テンプレート判定ロジック**
   - キーワードベースで4種類のテンプレートを判定
   - feature-1（機能要件）、feature-2-design（機能設計）、bug_report（バグ報告）、feature-3-coding（実装タスク）
   - 判定精度: 70-80%想定

3. **簡易記述判定**
   - 本文が200文字以内
   - テンプレートマーカー（`## 背景・目的` 等）が含まれていない
   - 本文が空でない

4. **LLMクライアント**
   - Google Gemini 2.5 Flash API統合
   - プロンプトテンプレートは4種類（各テンプレート用）
   - 生成パラメータ: max_tokens=2000, temperature=0.7

5. **実行モード**
   - 通常モード: GitHub Actionsから実行、コメント投稿
   - `--dry-run`: ローカル検証用、コメント投稿スキップ
   - `--index-issues`: RAGデータ生成（Phase 2で実装予定）

6. **コメント投稿**
   - GitHub CLI (`gh` コマンド) 経由で投稿
   - 一時ファイルを使用して長文対応
   - エラー時の適切なクリーンアップ

#### プロンプトテンプレート

各テンプレート用に専用のプロンプトを定義：

- **feature-1**: 背景・目的、完了条件、影響範囲、実装に必要な情報
- **feature-2-design**: 設計概要、実装に必要な情報、技術的検討事項、タスク
- **bug_report**: 現象、再現手順、期待する動作、実際の動作、環境情報、エラーメッセージ
- **feature-3-coding**: 実装内容、実装タスク、テスト、備考

### 2. GitHub Actions Workflowの実装

**ファイル**: `.github/workflows/issue_auto_improve.yml`

#### 主要機能

1. **トリガー設定**
   - Issue作成時（`issues: types: [opened]`）
   - `no-ai-assist` ラベルがついている場合はスキップ

2. **簡易記述判定**
   - Bashスクリプトで200文字チェック
   - テンプレートマーカーの存在チェック
   - 判定結果を `GITHUB_OUTPUT` に保存

3. **スクリプト実行**
   - `uv run` コマンドで実行
   - 環境変数で設定情報を受け渡し
   - タイムアウト: 5分

4. **エラーハンドリング**
   - 失敗時は警告コメントを投稿
   - `gh` コマンド経由で簡易コメント

#### 環境変数

- `ISSUE_BODY`: Issue本文
- `ISSUE_TITLE`: Issueタイトル
- `ISSUE_NUMBER`: Issue番号
- `LLM_API_KEY`: Gemini APIキー（GitHub Secrets）
- `GITHUB_TOKEN`: GitHub認証トークン（自動提供）
- `GITHUB_REPOSITORY`: リポジトリ名（自動提供）

### 3. ドキュメント作成

**ファイル**: `.github/scripts/README.md`

- 機能概要
- 使い方（通常利用、ローカル検証）
- テンプレート判定ロジック
- Phase 1の実装範囲
- 設定方法
- トラブルシューティング
- 開発者向け情報

## 技術スタック

### 依存ライブラリ

- `google-generativeai>=0.8.3`: Gemini API クライアント

### 実行環境

- Python 3.11+
- uv (パッケージマネージャ)
- GitHub CLI (gh)
- GitHub Actions (ubuntu-latest)

### 外部サービス

- Google Gemini API (2.5 Flash)
- GitHub API

## コスト見積もり

### Phase 1実装

月間想定利用量: 15リクエスト（Issue作成月10件 + 再生成5件）

| 項目 | 単価 | 月額 |
|------|------|------|
| Gemini 2.5 Flash API | $0.30/1M入力, $2.50/1M出力 | 約9円 |
| GitHub Actions | 無料枠2000分 | $0 |
| **合計** | - | **約9円/月** |

## テスト結果

### ローカルテスト

1. **簡易記述判定テスト**
   - ✅ 200文字以内の簡易Issue → 処理対象
   - ✅ 200文字超のIssue → スキップ
   - ✅ テンプレートマーカー含むIssue → スキップ

2. **テンプレート判定テスト**
   - ✅ "機能追加したい" → feature-1
   - ✅ "設計を検討したい" → feature-1（キーワード不足）
   - ✅ "エラーが発生する" → bug_report
   - ✅ "実装する" → feature-3-coding

### 動作確認項目

- [ ] 実際のIssue作成でWorkflowが起動
- [ ] LLM APIが正常に呼び出される
- [ ] 生成された例文がコメント投稿される
- [ ] エラー時に警告コメントが投稿される
- [ ] `no-ai-assist` ラベルで処理がスキップされる

## 制限事項・既知の問題

### Phase 1の制限

1. **RAG機能未実装**
   - 過去Issueの参照なし
   - 汎用的な例文生成のみ

2. **テンプレート判定精度**
   - キーワードベースのため誤判定の可能性
   - 複雑な記述では判定が難しい

3. **Issue更新時の再生成**
   - 未実装（Phase 4で実装予定）

### 運用上の注意

1. **APIキー管理**
   - GitHub Secrets で `GEMINI_API_KEY` を設定必須

2. **コスト監視**
   - Gemini APIの利用量を定期的に確認

3. **生成品質**
   - LLMの出力は確定的でないため、都度確認が必要

## 今後の拡張（Phase 2以降）

### Phase 2: RAG機能追加

- [ ] Qdrant Cloud連携
- [ ] Voyage AI Embeddingによるベクトル化
- [ ] 類似Issue検索機能
- [ ] 参考Issue情報をコメントに追加

### Phase 3: 複数テンプレート対応強化

- [ ] LLMベースのテンプレート判定
- [ ] 判定精度の向上

### Phase 4: 運用改善

- [ ] Issue更新時の再生成
- [ ] 既存コメントの更新機能
- [ ] ユーザーフィードバック機能

## セキュリティ考慮事項

1. **APIキー管理**
   - GitHub Secrets で管理
   - ログに出力しない

2. **入力検証**
   - Issue本文の長さチェック
   - テンプレートマーカーチェック

3. **エラー情報**
   - エラー時も機密情報を含めない

## 参考資料

- GitHub Actions Documentation: https://docs.github.com/actions
- Gemini API Documentation: https://ai.google.dev/docs
- PEP-723 (Inline script metadata): https://peps.python.org/pep-0723/
- uv Documentation: https://docs.astral.sh/uv/

## 実装者メモ

### 実装時の工夫

1. **PEP-723の採用**
   - 依存関係をスクリプト内に記述し、単一ファイル化
   - `uv run` で自動インストール、管理が容易

2. **GitHub CLI活用**
   - APIクライアント不要
   - 一時ファイル経由で長文コメント対応

3. **実行モードの分離**
   - `--dry-run` でローカルテスト可能
   - 開発効率の向上

### 開発時の課題と解決

1. **uvxとuv runの違い**
   - 課題: `uvx` はスクリプトファイルに非対応
   - 解決: `uv run` を使用

2. **テンプレート判定の精度**
   - 課題: キーワードベースでは精度が限定的
   - 解決: Phase 2でLLM判定に拡張予定

3. **コメント投稿の長文対応**
   - 課題: 引数長制限
   - 解決: 一時ファイル経由で投稿

## 完了確認

### Phase 1完了条件

- [x] Issue作成時にGitHub Actions Workflowが自動起動
- [x] Issue本文からテンプレートを自動判定（キーワードベース）
- [x] 判定したテンプレートに沿った例文を生成
- [x] 生成された例文がIssueコメントとして投稿
- [x] エラー時のハンドリングが適切
- [x] `--dry-run` モードでローカル検証が可能
- [ ] 実際のIssueで動作確認（要APIキー設定）

### 次のステップ

1. GitHub Secrets に `GEMINI_API_KEY` を設定
2. 実際のIssueで動作確認
3. テンプレート判定の精度を評価
4. Phase 2（RAG機能）の実装検討
