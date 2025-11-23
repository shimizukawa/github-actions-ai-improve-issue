# Issue自動改善: テンプレート設定化・構成整理 設計メモ

## 背景と目的

- プログラムを GitHub Actions から、または PyPI 経由でインストールして利用することを前提とする。
- 現状の課題:
  - 使い方ドキュメントが不十分で、導入・設定方法が分かりにくい。
  - テンプレート名およびテンプレートごとのシステムプロンプトがソースコードにハードコードされており、設定ファイルから変更できない。
- 後方互換性やデフォルト動作は不要とし、新しい構成・設定方式に一本化する。

## 方針（概要）

1. **コード配置の整理**
   - 現状 `.github/scripts/improve_issue.py` に存在するロジックを `/src/github_actions_ai_improve_issue/` 配下に移動し、パッケージとして管理する。
   - `src/github_actions_ai_improve_issue/main.py` は古い実装のため削除し、新構成に統一する。
   - GitHub Actions からはパッケージのエントリポイントを実行する（例: `uv run -m github_actions_ai_improve_issue`）。

2. **設定ファイル必須化**
   - 設定ファイルは YAML 形式とし、基本ファイル名を `.improve_issue.yml` とする。
   - 設定ファイルの探索順:
     1. 環境変数 `IMPROVE_ISSUE_CONFIG` にパスが指定されていれば、そのファイルを読み込む。
     2. 指定がない場合は、リポジトリルート直下の `.improve_issue.yml` を読み込む。
   - 上記いずれにも設定ファイルが存在しない場合はエラーとして終了させる（フォールバック・暗黙のデフォルトは持たない）。

3. **テンプレート定義の設定ファイル化**
   - テンプレート名、対応する Issue テンプレートファイル、システムプロンプト、判定キーワードをすべて設定ファイルで定義する。
   - コード側にはテンプレート内容やシステムプロンプトを一切ハードコードしない。
   - テンプレート数や名称は設定ファイル次第で増減可能とする。

4. **テンプレート判定ロジックの設定依存化**
   - Issue タイトル・本文からテンプレートを自動判定するロジックは維持するが、テンプレートごとのキーワード一覧を設定ファイルに持たせる。
   - 判定ロジックは設定からロードしたテンプレート群を走査し、キーワードマッチ数に基づいてテンプレートを選択する実装とする。

5. **システムプロンプトの外部化**
   - 各テンプレートに対応するシステムプロンプトは、設定ファイルの `system_prompt` として定義する。
   - 既存の `ROLE_AND_INSTRUCTIONS` 定数など、ソースコード内のハードコードされたシステムプロンプトは削除する。

6. **モデルの扱い**
   - 将来テンプレートごとにモデルを変えることは考慮しない。
   - LLM クライアントはこれまでどおり単一のモデル設定（例: `gemini-2.5-flash`）を利用し、テンプレートとは独立して扱う。

## 設定ファイル設計

### ファイル配置と探索

- パッケージから見たリポジトリルート（例: `repo_root = Path(__file__).resolve().parents[2]` 程度）を基準とする。
- 設定ファイル探索順:
  1. 環境変数 `IMPROVE_ISSUE_CONFIG` があれば、そのパスを最優先で利用。
  2. なければ `repo_root / ".improve_issue.yml"` を参照。
- どちらにもファイルが存在しない場合:
  - 設定読み込み処理で例外を送出し、メイン処理でメッセージを出力して終了（`sys.exit(1)` 相当）。

### YAML 構造案

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

### 設定用データモデル

- 設定ファイルを読み込んで内部表現にマッピングするため、以下のデータクラスを想定する。

```python
@dataclasses.dataclass
class TemplateConfig:
    name: str
    issue_template_file: str
    system_prompt: str
    keywords: list[str]


@dataclasses.dataclass
class ImproveIssueSettings:
    templates: dict[str, TemplateConfig]
    default_template: str
```

- バリデーション方針:
  - `default_template` が `templates` に含まれていない場合はエラー。
  - `templates` が空の場合はエラー。
  - 各 `TemplateConfig` について、`issue_template_file` / `system_prompt` / `keywords` は必須。

### 設定ロード

- 関数イメージ: `load_settings() -> ImproveIssueSettings`
  - 設定ファイルパスの決定（環境変数 → `.improve_issue.yml`）。
  - YAML 読み込みと辞書への変換。
  - 上記データクラスへのマッピングとバリデーション。
  - 問題があれば例外を投げる。

## テンプレート判定ロジック

### 目的

- Issue タイトルと本文から、どのテンプレートを使うべきかを自動で選択する。
- テンプレートごとのキーワードは `.improve_issue.yml` 内で定義する。

### 動作イメージ

- 判定クラス:

```python
class TemplateDetector:
    def __init__(self, settings: ImproveIssueSettings):
        self.settings = settings

    def detect(self, issue_body: str, issue_title: str = "") -> str:
        text = f"{issue_title} {issue_body}".lower()

        best_template: str | None = None
        best_score = -1

        for name, tmpl in self.settings.templates.items():
            score = sum(1 for kw in tmpl.keywords if kw.lower() in text)
            if score > best_score:
                best_score = score
                best_template = name

        if best_template is None or best_score <= 0:
            return self.settings.default_template

        return best_template
```

- テンプレート名の型は、設定で増減できるように Python コード側では `str` として扱う。

## Issue テンプレートファイル読み込み

### 目的

- GitHub の Issue テンプレート（`.github/ISSUE_TEMPLATE/*.md`）からフロントマターを除いた本文を取得し、LLM の出力テンプレートとして利用する。

### 動作イメージ

```python
def load_template_content(template: TemplateConfig) -> str:
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    template_file = (
        repo_root / ".github" / "ISSUE_TEMPLATE" / f"{template.issue_template_file}.md"
    )

    if not template_file.exists():
        raise FileNotFoundError(f"Issueテンプレートファイルが見つかりません: {template_file}")

    content = template_file.read_text(encoding="utf-8")

    # frontmatter (--- で囲まれた部分) の除去
    lines = content.split("\n")
    if lines and lines[0] == "---":
        for i in range(1, len(lines)):
            if lines[i] == "---":
                content = "\n".join(lines[i + 1 :])
                break

    return content.strip()
```

## プロンプト生成

### 目的

- 設定されたシステムプロンプトと Issue テンプレート本文、Issue 内容、RAG で取得した類似 Issue などを組み合わせて、LLM に渡すプロンプト文字列を構築する。

### インターフェース例

```python
def get_improve_prompt(
    template_name: str,
    issue_body: str,
    issue_title: str,
    similar_issues: list[dict[str, Any]] | None,
    settings: ImproveIssueSettings,
) -> str:
    tmpl = settings.templates[template_name]
    template_content = load_template_content(tmpl)

    prompt = f"""{tmpl.system_prompt}

【Issue記述】
タイトル: {issue_title}
本文: {issue_body}

【出力テンプレート】
以下のテンプレートに沿って具体的に記述してください：

{template_content}
"""

    # RAG の検索結果があれば、参考情報として追記する
    if similar_issues:
        ...

    return prompt
```

- 既存の RAG 連携（類似 Issue 情報の追記）はそのまま利用する。
- これにより、`ROLE_AND_INSTRUCTIONS` などのハードコードは不要となる。

## LLM クライアントとの連携

- LLM クライアント自体は従来どおり単一モデルを利用する。
- テンプレートごとにモデルを変える要件は明示的に「考慮しない」ため、設定ファイル側にもモデル名は持たせない。
- メインフローでは:
  1. 設定をロードして `ImproveIssueSettings` を構築。
  2. `TemplateDetector` でテンプレート名を決定。
  3. `get_improve_prompt` でプロンプトを生成。
  4. LLM クライアントにプロンプトを渡して出力を得る。

## ドキュメント整備方針

- README などのドキュメントで、以下を明確にする。
  - インストール方法（PyPI / GitHub Actions いずれの場合も）。
  - `.improve_issue.yml` が必須であることと、その配置場所・環境変数 `IMPROVE_ISSUE_CONFIG` の意味。
  - YAML のスキーマとサンプル（本設計で示したもの）。
  - テンプレート追加・削除の手順（設定ファイル編集と `.github/ISSUE_TEMPLATE/*.md` の用意）。
  - GitHub Actions の最小サンプルワークフローと、RAG を有効にした高度なサンプル。

## 実装上のメモ

- 後方互換性・フォールバックは不要なため、旧 `main.py` は削除し、新方式のみに統一する。
- 設定が見つからない・不正といった場合は、早期に処理を中断して明示的にエラーを返す実装とする。
- 実装後は `uv format` でコード整形を行う。