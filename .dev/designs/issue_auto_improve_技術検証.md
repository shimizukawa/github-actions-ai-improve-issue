# 技術検証: Issue自動ブラッシュアップ機能

## 概要

Issue自動ブラッシュアップ機能の実現可能性を検証するため、Phase 0（技術検証・PoC）として以下の技術要素を調査・検証する。

## 検証目的

- GitHub Actions環境でのLLM API呼び出しの動作確認
- 実行時間・コストの実測
- 生成される例文の品質評価
- 技術選定の妥当性検証

## 検証項目

### 1. LLM API選定

#### 検証対象（2025年11月時点）

| サービス | モデル | 特徴 | 入力/100万T | 出力/100万T |
|---------|--------|------|------------|------------|
| **Google** | Gemini 2.0 Flash-Lite | 超低コスト、検証向け | $0.075 | $0.30 |
| **Google** | Gemini 2.5 Flash | コスパ良好、ハイブリッド推論 | $0.30 | $2.50 |
| **Anthropic** | Claude 3.7 Sonnet | 最新、推論強化、コーディング向け | $3.00 | $15.00 |
| **OpenAI** | GPT-4o-mini | バランス型、低コスト | $0.15 | $0.60 |

#### 検証内容

**プロンプト例:**
```
以下のIssue記述を、feature-1（機能要件）テンプレートに沿った具体的な内容に拡張してください。

【Issue記述】
想定運転データの一括登録機能を追加したい

【テンプレート】
（feature-1テンプレートの構造を提示）

【出力形式】
Markdown形式で、テンプレートの各項目を埋めてください。
```

**評価指標:**
- テンプレート項目の網羅率
- 生成内容の具体性（抽象的すぎないか）
- 生成内容の妥当性（明らかな誤りがないか）
- 生成時間
- API利用コスト（1リクエストあたり）

**想定結果（2025年11月更新）:**
- Gemini 2.0 Flash-Lite: 中品質、10秒、$0.002-0.003
- Gemini 2.5 Flash: 高品質、12秒、$0.015-0.020
- Claude 3.7 Sonnet: 最高品質、15秒、$0.024-0.030
- GPT-4o-mini: 高品質、10秒、$0.004-0.006

**参考: 2025年10月 Nejumi Leaderboard 日本語LLMランキング**
- Gemini 2.5 Pro: 1位（コーディング性能 0.6449）
- OpenAI o4-mini: 2位（0.6444）
- Claude 3.7 Sonnet: 8位（0.5940、実装タスク向け）
- Gemini 2.5 Flash: 30位（0.5004、コスパ重視）

#### 推奨選定（2025年11月更新）

**Phase 0-1**: Gemini 2.0 Flash-Lite（極低コスト、検証速度優先）
- 代替: Gemini 2.5 Flash（品質重視の場合）

**Phase 2以降**: Claude 3.7 Sonnet（最新版、推論能力強化）
- 拡張思考モード（Thinking）対応で判定精度向上の可能性
- コーディング関連タスクに強い

**Phase 2代替案**: Gemini 2.5 Flash（コスト重視の場合）
- Claude 3.7の約1/10のコストで中程度の品質

### 2. GitHub Actions Workflow

#### 検証内容

本番用 `issue_auto_improve.yml` は以下の構成で、`ai-processing` / `ai-processed` ラベルをもとに重複処理を防ぎつつ `uv run` スクリプトを実行し、成功時に `ai-processed` を付与する。失敗時には `ai-processing` を外して再試行できるようにする。

```yaml
name: Issue Auto Improve

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

#### 検証ポイント

- ラベル gating (`ai-processing` / `ai-processed`) が正しく動作するか
- `uv run .github/scripts/improve_issue.py` により例文が生成され、コメントが投稿されるか
- 失敗時には `ai-processing` を除去して再実行可能になるか


**想定課題と対策:**
| 課題 | 対策 |
|------|------|
| API呼び出しタイムアウト | リトライロジック追加、タイムアウト値調整 |
| Secrets管理 | GitHub Secrets使用、環境変数で受け渡し |
| 実行時間超過 | 処理の並列化、軽量モデル使用 |
| 重複コメント投稿 | Workflow実行履歴チェック、コメント存在確認 |

### 3. テンプレート判定ロジック

#### 検証内容

テンプレート判定は厳密な LLM 判定ではなく、キーワードベースで `feature_request` / `bug_report` を選定します。具体的には、タイトル/本文の小文字化された文字列に対して以下のキーワードの出現回数をスコアとして集計し、最大スコアのテンプレートを採用（0点の場合は `feature_request` をデフォルト）。

**キーワード例:**
- `feature_request`: 機能, 追加, 変更, 改善, したい, 欲しい, 必要
- `bug_report`: バグ, エラー, 不具合, 動かない, 失敗, 問題

**検証手順:**
1. テスト用 Issue を用意し、タイトルと本文に一致するキーワードを含める
2. `TemplateDetector.detect()` を実行し、期待されるテンプレート名が返るか確認
3. キーワードがないケースでは `feature_request` にフォールバックすることを確認

**評価指標:**
- 判定精度（10件程度のテストケースで実測）
- 判定に要する時間（関数実行 + ファイル読み込み）

#### 誤判定時の対策

- `KEYWORDS` の調整（重要キーワードの追加/削除）
- タイトル/本文の正規化（全角→半角など）
- 不明な場合は `feature_request` にフォールバックし、生成コンテンツで差分を補完

### 4. コスト試算

#### 想定利用量

- Issue作成頻度: 月20件
- 処理対象（簡易記述）: 月10件（50%）
- 再生成（Issue更新）: 月5件

**合計: 月15リクエスト**

#### コスト計算（2025年11月更新）

**Gemini 2.0 Flash-Lite:**
- 入力トークン: 500トークン/リクエスト × $0.075/1M = $0.0000375
- 出力トークン: 1500トークン/リクエスト × $0.30/1M = $0.00045
- 月間コスト: $0.00048 × 15 = **$0.007/月（約1円）**

**Gemini 2.5 Flash:**
- 入力トークン: 500トークン × $0.30/1M = $0.00015
- 出力トークン: 1500トークン × $2.50/1M = $0.00375
- 月間コスト: $0.00390 × 15 = **$0.059/月（約9円）**

**Claude 3.7 Sonnet:**
- 入力トークン: 500トークン × $3/1M = $0.0015
- 出力トークン: 1500トークン × $15/1M = $0.0225
- 月間コスト: $0.024 × 15 = **$0.36/月（約54円）**

**GPT-4o-mini:**
- 入力トークン: 500トークン × $0.15/1M = $0.000075
- 出力トークン: 1500トークン × $0.60/1M = $0.0009
- 月間コスト: $0.000975 × 15 = **$0.015/月（約2円）**

**GitHub Actions:**
- 実行時間: 2分/回 × 15回 = 30分/月
- コスト: 無料枠2000分内で収まる（**$0**）

**合計想定コスト: 月1-54円（モデル選択による）**

| モデル | 月額コスト | 用途 |
|--------|-----------|------|
| Gemini 2.0 Flash-Lite | 約1円 | Phase 0検証 |
| GPT-4o-mini | 約2円 | 低コスト運用 |
| Gemini 2.5 Flash | 約9円 | コスパ重視運用 |
| Claude 3.7 Sonnet | 約54円 | 品質重視運用 |

### 5. 実装言語・ライブラリ選定

#### Python実装

**推奨構成:**
```
.github/
├── workflows/
│   └── issue_auto_improve.yml
└── scripts/
    └── improve_issue.py          # 単一スクリプト（PEP-723対応）
```

**依存ライブラリ管理:**

PEP-723を使用し、スクリプト内で依存関係を宣言：
```python
# /// script
# dependencies = [
#   "google-generativeai>=0.8.3",
# ]
# ///
```

**ローカル検証方法:**

```bash
# 環境変数設定
export ISSUE_TITLE="Mermaid図の拡大縮小機能"
export ISSUE_BODY="ページ編集 でMermaid図の部分だけ別ウインドウで表示して拡大縮小したい。ページ全体の編集ではプレビューの縦位置がずれてしまうため使いづらい"
export ISSUE_NUMBER="123"
export LLM_API_KEY="your-api-key"

# --dry-run モードでローカル実行（コメント投稿なし）
uvx .github/scripts/improve_issue.py --dry-run
```


## 参考情報

- GitHub Actions Documentation: https://docs.github.com/actions
- Gemini API Documentation: https://ai.google.dev/docs
- Claude API Documentation: https://docs.anthropic.com/
- OpenAI API Documentation: https://platform.openai.com/docs
