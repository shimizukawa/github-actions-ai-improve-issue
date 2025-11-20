# Issue自動ブラッシュアップ機能 - 要件

## 概要

簡易なブランクIssueに1-2行程度のやりたいことを書いたら、過去の類似Issue情報を参考に適切なテンプレートに沿った具体的な文章を自動生成し、コメントで提示する機能を実装する。

## 背景・現状の課題

### Issue作成時の課題

1. **テンプレート記入の心理的障壁**
   - テンプレートの各項目を埋めるのが面倒
   - 何を書けばいいのか分からず、手が止まる
   - 適切な文言を考えるのに時間がかかる

2. **テンプレート選択の難しさ**
   - 要件・設計・実装など、どのテンプレートを使うべきか判断が難しい
   - テンプレート選択を間違えると後で修正が必要

3. **記述内容の抽象化**
   - 書いた内容が抽象的になりがち
   - 目的や背景が不明確なまま提出される
   - 具体的な情報が不足し、レビュー時に追加質問が発生

4. **過去事例の活用不足**
   - 類似したIssueが過去にあっても参照されない
   - 同じような議論が繰り返される
   - 過去の知見が蓄積されない

### 結果として生じる問題

- Issue作成のハードルが高く、気軽に起票できない
- Issue品質のばらつきが大きい
- レビュアーが毎回同じ質問をする手間
- Issue完了までの往復コミュニケーションコストの増加

## 目的

### 解決したいこと

1. **Issue作成の心理的ハードルを下げる**
   - やりたいことを1-2行書くだけでOK
   - テンプレート選択を自動化
   - 文章作成の手間を削減

2. **AI生成例文による記述支援**
   - 過去の類似Issueを参考にした具体的な例文を提示
   - テンプレートの各項目を自動的に埋めた下書きを生成
   - ユーザーは生成された例文を実情に合わせて微修正するだけ

3. **Issue品質の底上げ**
   - テンプレートに沿った構造化された内容
   - 過去の知見を反映した具体的な記述
   - 必要な情報項目の記入漏れを防止

4. **低コストでの情報充実化**
   - AI生成文をベースに人間が最小限の編集で完成
   - 一から書くよりも圧倒的に低コスト
   - 具体的な例があるため、何を書けばいいか明確

### 解決しないこと

- Issue内容の妥当性判断（技術的実現可能性等）
- Issue優先度の自動決定
- 完全自動でのIssue完成（人間の確認・修正は必須）

## ユースケース

### 基本フロー

1. ユーザーがブランクIssueを作成し、やりたいことを1-2行で記述
   ```
   例: 「Mermaid図の拡大縮小機能を追加したい」
   ```

2. ユーザーが「このIssueにAI補助をかけたい」と判断した場合、`ai-improve` ラベルを付与

3. `ai-improve` ラベル付与をトリガーに GitHub Actions Workflow が起動

4. 簡易記述から適切なテンプレートを判定
   - 機能要件Issue（feature_request）
   - バグ報告（bug_report）

4. RAGで類似Issue情報を検索・取得（RAG有効時のみ）

5. 判定したテンプレートに沿って、過去事例を参考にした具体的な文章を生成

6. Issueにコメントとして生成された例文を投稿
   ```markdown
   ## 🤖 AIによるIssue記入例
   
   [選定テンプレート: 機能要件（親Issue）]
   
   ### 背景・目的
   （具体的な文章例）
   
   ### 完了条件
   - [ ] ...
   
   （以下、テンプレートに沿った内容）
   ```

7. ユーザーは生成された例文を見て、実情に合わせて微修正
   - Issue本文を更新 or コメント内容を参考に追記

8. 処理完了後、Workflow は `ai-improve` ラベルを自動で削除（再度ラベルを付ければ再ブラッシュアップ可能）

### 将来的な拡張

- Issue本文が更新された際にも再度ブラッシュアップを実行
- ブラッシュアップコメントを更新（上書き or 追記）

## 完了条件（Acceptance Criteria）

### Phase 1（実装済み）

- [x] `ai-improve` ラベル付与時にGitHub Actions Workflowが起動する（`on: issues.types: [labeled]`）
- [x] キーワードベースで `feature_request` / `bug_report` のテンプレートを選定
- [x] `.github/ISSUE_TEMPLATE` に沿ったプロンプトを組み立てて、Gemini API (`gemini-2.5-flash`) から例文を生成
- [x] `gh issue comment` でフォーマット済みコメントを投稿する
- [x] `uv run .github/scripts/improve_issue.py --dry-run` でローカル検証が行える

### Phase 2 以降（未実装・将来対応）

- [x] RAG で類似Issueを取得し、Top-Kをプロンプトに含める
- [x] 類似Issueのメタ情報（スコア・番号・URL）をコメントで明示
- [x] Issue本文更新時の再生成・コメント更新を自動化
- [x] Qdrant Cloud 等のベクトルDBに Issue をインデックス登録
- [x] `--index-issues` / Embedding API による RAG インジェスト機能

## 機能仕様

### テンプレート自動判定

**判定対象:**
- ブランクIssue、または短い記述（1-2行程度）のIssue

**判定先テンプレート:**
- 機能要件Issue（`.github/ISSUE_TEMPLATE/feature_request.md`）
- バグ報告（`.github/ISSUE_TEMPLATE/bug_report.md`）

**判定方法:**
- `feature_request` / `bug_report` それぞれのキーワードリスト（`機能`, `バグ`, `改善` など）と本文・タイトルを照合
- 最もスコアが高いテンプレートを選択し、スコアがゼロの場合は `feature_request` をデフォルト
- 判定結果をもとにテンプレートファイルの本文を読み込み、LLMプロンプトを構築

**非対象:**
- 特定のラベルが設定されたissue

### 参考情報の範囲

**含める情報（Phase 1）:**
- 作成直後の Issue タイトル・本文（1-2行程度の簡易記述）
- テンプレートファイル（`.github/ISSUE_TEMPLATE/*.md`）の構造

**含める情報（Phase 2 - RAG有効時）:**
- Phase 1の情報に加えて:
- 類似Issue Top-3（タイトル・本文抜粋・類似度・URL）
- 類似Issueのメタデータ（状態・ラベル・作成日）

**RAG動作モード:**
- **RAG有効**: `QDRANT_URL`, `QDRANT_API_KEY`, `VOYAGE_API_KEY` が全て設定されている場合
- **RAGなし**: 上記環境変数のいずれかが不足している場合、Phase 1モードで動作（エラーにはしない）
- 動作モードはログに出力され、ユーザーは環境変数の設定状況を確認可能

**含めない情報:**
- Pull Request（実装済み/実装中のため）

### ブラッシュアップの観点

1. **テンプレート準拠性**: 対応するテンプレートの項目が埋まっているか
2. **情報の網羅性**: 必要な項目（背景・目的、完了条件等）が漏れていないか
3. **具体性**: 抽象的な記述を具体的で再現性のある表現に改善
4. **関連情報の補完**: 依存関係・参考資料・前提知識が示されているか

### 出力形式（Phase 1）

**コメント投稿:**
- GitHub Actions Bot が `gh issue comment` で対象IssueにMarkdownコメントを投稿
- コメント末尾に `<!-- AI-generated comment -->` を残し、Botによる生成物であることを明示

**コメント構成例:**
```markdown
## 🤖 AIによるIssue記入例

**選定テンプレート**: 機能要件（feature_request）

---

（LLMが生成したテンプレート準拠の具体例）

---

💡 **使い方**: 上記の例文を参考にIssue本文を更新してください。実情に合わせて適宜修正してください。

<!-- AI-generated comment -->
```

### アーキテクチャ概要（Phase 1）

```
[GitHub Issue作成]
   ↓
[ユーザーが ai-improve ラベルを付与]
   ↓
[Workflow起動]  # `issues` イベント（labeled, label.name == "ai-improve"）
   ↓
[TemplateDetector]  # キーワード判定＋テンプレート読み込み
   ↓
[LLM (Gemini 2.5 Flash) で例文生成]
   ↓
[コメント投稿（gh）]
   ↓
[Workflowが ai-improve ラベルを削除]
```

### ローカル検証モード

- `uv run .github/scripts/improve_issue.py --dry-run` で依存関係を自動インストールしつつ、コメント投稿をスキップして出力のみ表示
- 環境変数には `ISSUE_TITLE`, `ISSUE_BODY`, `ISSUE_NUMBER`, `LLM_API_KEY`, `GITHUB_TOKEN`（コメントなしでも取得のために必要）を設定

### RAG（Phase 2 以降の予定）

RAG検索・ベクトルDB連携はPhase 2で実装予定。現行では `--index-issues` の処理は含まれていませんが、今後以下のような機能を追加する計画です。

**主な構成:**
1. 全Issueまたは指定範囲を対象にIssue本文/タイトルを結合
2. Embedding APIでベクトルを生成
3. Qdrant等のベクトルDBへ登録（メタデータ付き）
4. メンテナンス用の進捗ログとエラーハンドリング

**メタデータ例:**
- `issue_number`, `issue_title`, `issue_body`（先頭500文字）、`template_type`, `state`, `created_at`, `url`

**用途:**
- RAG検索用ベクトルDBの初期構築
- 新規Issue作成時の情報追加（Phase 2で自動化予定）
- 定期的な全体再インデックス

**実行タイミング:**
- Phase 2導入時の初回データ投入
- 大量Issue作成後の一括インデックス
- インデックス再構築が必要な場合

#### モード比較（予定）

| 項目 | モード1: 単一Issue改善 | モード2: RAGデータ生成 |
|------|----------------------|----------------------|
| **実行** | `uvx .github/scripts/improve_issue.py [--dry-run]` | `uvx .github/scripts/improve_issue.py --index-issues` |
| **対象** | 単一Issue（ISSUE_NUMBER指定） | 全Issue or 範囲指定 |
| **GitHub API** | Issue取得（読取り） | Issue取得（読取り） |
| **RAG検索** | なし（Phase 1） | 類似Issue検索（予定） |
| **LLM API** | 例文生成 | なし |
| **Embedding API** | なし | ベクトル化 |
| **Qdrant** | なし | 登録（書込み） |
| **コメント投稿** | あり（`--dry-run`でスキップ） | なし |
| **出力** | コメント or コンソール | 進捗ログ |

#### データ構造設計（Phase 2以降の予定）

**Issueデータの扱い:**
- 現時点では Issue本文を主要なドキュメントとして扱う
- 将来的には Issueコメントや外部ドキュメントサービスのナレッジも "関連ドキュメント" として紐付け、検索コンテキストとして利用する
- メタデータ: Issue番号・作成日・ラベル・状態（open/closed）など

**インデックス更新タイミング:**
- Issue作成時(例文生成後に自動登録)
- Issue本文更新時(edited イベント)
- Issue状態変更時(closed, reopened イベント)
- 初回セットアップ時(CLI実行: `--index-issues`)

> メモ（現時点の方針）
> - コメント投稿をトリガーに Issue本文を毎回再インデックスする運用は、コスト対効果が低いため行わない
> - 将来的にコメント本文や外部ナレッジをインデックス対象に含める際に、改めてコメントイベントをトリガーとして利用する

### GitHub Actions Workflow

#### トリガー・条件

- `issues` イベントの `opened` のみを対象に起動
- `jobs.improve-issue.if` で `ai-processing` / `ai-processed` ラベルをチェックし、既に処理済みのIssueはスキップ
- `runs-on: ubuntu-slim`、`timeout-minutes: 2` で短時間のAPI中心処理に対応

#### 処理フロー（実装バージョン）

```yaml
name: Issue Auto Improve (Phase 1)

on:
   issues:
      types: [opened]

jobs:
   improve-issue:
      runs-on: ubuntu-slim
      timeout-minutes: 2
      if: |
         !contains(github.event.issue.labels.*.name, 'ai-processing') &&
         !contains(github.event.issue.labels.*.name, 'ai-processed')
      steps:
         - name: Checkout repository
            uses: actions/checkout@v4
         - name: Install uv
            uses: astral-sh/setup-uv@v5
            with:
               version: "latest"
         - name: Add processing label
            run: gh issue edit ${{ github.event.issue.number }} --add-label "ai-processing"
            env:
               GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
         - name: Improve issue content
            id: improve
            run: uv run .github/scripts/improve_issue.py
            env:
               ISSUE_BODY: ${{ github.event.issue.body }}
               ISSUE_TITLE: ${{ github.event.issue.title }}
               ISSUE_NUMBER: ${{ github.event.issue.number }}
               LLM_API_KEY: ${{ secrets.GEMINI_API_KEY }}
               GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
               GITHUB_REPOSITORY: ${{ github.repository }}
         - name: Mark as processed
            if: success()
            run: gh issue edit ${{ github.event.issue.number }} --remove-label "ai-processing" --add-label "ai-processed"
            env:
               GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
         - name: Remove processing label on failure
            if: failure()
            run: gh issue edit ${{ github.event.issue.number }} --remove-label "ai-processing"
            env:
               GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

#### ラベルとエラーハンドリング

- `ai-processing` で処理中ステータスを示し、完了時に `ai-processed` を付与して二重実行を防止
- 失敗時には専用ステップで `ai-processing` をはずし、再実行できる状態にする

### セキュリティ・認証

- **GitHub Token**: `GITHUB_TOKEN`（自動提供）
- **Qdrant API Key**: GitHub Secrets で管理
- **LLM API Key**: GitHub Secrets で管理

## 非機能要件

### パフォーマンス

- Workflow実行時間: 5分以内
- RAG検索レスポンス: 3秒以内
- LLM生成時間: 20秒以内

### コスト

- GitHub Actions実行時間: 月間1000分以内（無料枠活用）
- Qdrant Cloud: 無料プラン範囲内
- Embedding生成: 月間XX円以内（要算定）
- LLM API: 月間XX円以内（要算定）

### 可用性

- Workflow失敗時はIssue作成者に通知（オプション）
- 外部API障害時はグレースフルに失敗

## 実装スコープ

### Phase 0（技術検証・PoC）

**目的**: RAGなしで基本フローの動作確認、技術選定の妥当性検証

**実装内容:**
- [x] GitHub Actions Workflowの基本構造構築
- [x] Issue作成イベントのトリガー設定
- [x] LLM APIとの接続確認（Gemini 2.5 Flash を中心に）
- [x] 固定プロンプトでの例文生成テスト
- [x] Issueコメント投稿の実装

**検証ポイント:**
- GitHub Actions環境でのLLM API呼び出しが正常動作するか
- 実行時間・コストが許容範囲か
- 生成される例文の品質が使えるレベルか

**完了条件:**
- ✅ ブランクIssueに「機能追加したい」と書くと、例文がコメントされる（Phase 0 で確認済み）

**期待工数**: 2-3日

---

### Phase 1（基本機能実装）

**目的**: テンプレート判定と汎用的な例文生成でIssue記述支援を行う

**実装内容:**
- [x] Issue本文/タイトルからのテンプレート自動判定（feature_request / bug_report）
- [x] テンプレート構造を読み取ってプロンプトを構築
- [x] Gemini 2.5 Flash でテンプレート準拠の例文を生成
- [x] `gh issue comment` + `ai-processing` / `ai-processed` ラベル管理を組み合わせたコメント投稿
- [x] `--dry-run` モードでコメント投稿をスキップするローカル検証

**検証ポイント:**
- テンプレート判定ロジックが安定して `feature_request` / `bug_report` を返すか
- 生成された例文にテンプレートの主要項目が含まれているか
- コメント投稿とラベル遷移（`ai-processing`→`ai-processed`）が正しく動作するか

**完了条件:**
- ✅ 1-2行のIssueに対して、テンプレート準拠のコメントが投稿される
- ✅ `ai-processing` で処理中を示し、成功時に `ai-processed` を付与

**期待工数**: 3-5日

---

### Phase 2（RAG機能追加）

**目的**: 過去の類似Issue情報を活用し、より具体的で実用的な例文を生成

**実装内容:**
- [ ] Qdrant Cloudのセットアップ
- [ ] Embedding生成サービスの選定・実装(Voyage AI 3.5-lite)
- [ ] **RAG環境変数チェック機能**: 必要な環境変数の有無を判定し、不足時はRAGなしモードで動作
- [ ] 動作モードのログ出力（RAG有効/無効の明示）
- [ ] `--index-issues` コマンドによる既存Issue一括インデックス機能
- [ ] `--update-single-issue` コマンドによる単一Issue更新機能
- [ ] Issue作成時の自動インデックス登録(例文生成後、RAG有効時のみ)
- [ ] Issue編集時の自動インデックス更新(イベント駆動、RAG有効時のみ)
- [ ] 類似Issue検索機能の実装(Top-3、RAG有効時のみ)
- [ ] 検索結果をプロンプトに含めた例文生成(RAG有効時のみ)
- [ ] 参考にしたIssueをコメントに明示(RAG有効時のみ)

**検証ポイント:**
- 類似Issue検索の適合率は十分か
- RAG導入により例文の具体性・品質が向上したか
- 実行時間・コストは許容範囲か

**完了条件:**
- [ ] 類似Issue Top-3を取得し、それを参考にした例文が生成される
- [ ] 類似Issue適合率60%以上
- [ ] 例文の具体性がPhase 1より向上（主観評価）

**期待工数**: 5-7日

---

### Phase 3（複数テンプレート対応）

**目的**: 要件以外のIssueタイプにも対応し、適用範囲を拡大

**実装内容:**
- [ ] バグ報告（bug_report）テンプレート対応
- [ ] 各テンプレート用のプロンプト最適化
- [ ] テンプレート判定精度の向上

**検証ポイント:**
- 2種類のテンプレートを正しく判別できるか
- 各テンプレートで適切な例文が生成されるか

**完了条件:**
- [ ] 2種類すべてのテンプレートに対応
- [ ] テンプレート判定精度80%以上

**期待工数**: 3-5日

---

### Phase 4（運用改善・品質向上）

**目的**: 継続的な利用に向けた改善とフィードバック機構の実装

**実装内容:**
- [ ] Issue本文更新時の再生成機能
- [ ] 生成コメントの更新（上書き）機能
- [ ] プロジェクトドキュメント（`.dev/`配下）の参考情報化
- [ ] ユーザーフィードバック機能（👍/👎リアクション検知）
- [ ] 生成品質の継続的モニタリング
- [ ] プロンプトのA/Bテスト基盤

**検証ポイント:**
- ユーザー満足度は向上したか
- フィードバックを元にした改善が機能するか

**完了条件:**
- [ ] ユーザーが「使いやすくなった」と感じる（アンケート）
- [ ] 生成品質の定量的な向上を確認

**期待工数**: 5-7日

---

### スコープ外（将来検討事項）

- Issue優先度の自動判定
- 工数見積もりの自動算出
- PRへの自動レビューコメント
- 類似Issue間の依存関係の自動検出

## 技術的検討事項

### 要検討項目

1. **Embedding生成サービスの選定**
   - OpenAI text-embedding-3-small/large
   - Cohere Embed
   - Voyage AI
   - OSS（sentence-transformers等）
   - Qdrant fastembed

2. **チャンキング戦略**
   - Issue本文の分割単位（文字数・段落単位）
   - コメントの扱い（個別 or 連結）
   - 重複・オーバーラップの設定

3. **RAG検索パラメータ**
   - 類似Issue取得件数（Top-K）
   - スコア閾値
   - フィルタリング条件（ラベル、状態等）

4. **LLMプロンプト設計**
   - システムプロンプトの定義
   - Few-shotサンプルの用意
   - 出力形式の制約

5. **Workflow実行最適化**
   - キャッシング戦略
   - 並列実行可能性
   - 失敗時のリトライロジック

## 議論ポイント

1. **技術的実現性**
   - RAG基盤の構築難易度
   - GitHub Actions環境での実行可能性
   - 各種APIの統合複雑度

2. **コスト対効果**
   - API利用料金の試算
   - 開発・運用コストの見積もり
   - 期待される効果（Issue品質向上）の定量化

3. **運用負荷**
   - Embedding更新の頻度・タイミング
   - エラー監視・アラート設計
   - プロンプト・パラメータのチューニング作業

4. **代替案検討**
   - ローカルLLM（Ollama等）の利用可能性
   - より簡易な実装（テンプレートチェックのみ）
   - 手動トリガーでの運用

## 関連ドキュメント

- GitHub Issue Template: `.github/ISSUE_TEMPLATE/feature_request.md`（要件）
- GitHub Issue Template: `.github/ISSUE_TEMPLATE/bug_report.md`（バグ報告）
- Qdrant Documentation: https://qdrant.tech/documentation/
- GitHub Actions Documentation: https://docs.github.com/actions

## 備考

- 本要件は技術的な実現可能性の議論を目的としている
- 詳細な技術設計は別途設計Issueで実施
- 各種APIの選定・評価は別途調査タスクとして実施
