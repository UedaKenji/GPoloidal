# Script Conventions

`src/gpoloidal/scripts/` は「再現可能な解析手順」を置く場所です。  
RT-1 固有の設定組み立ては script 側に書いてよく、汎用処理は `src/gpoloidal/` の共通モジュールに置きます。

## 役割分担

- `src/gpoloidal/experiment.py`
  - cache / manifest / run record の backend
  - hash 化、cache 再利用、run 記録保存
- `src/gpoloidal/tomography.py`
  - `GPT_lin_general`, `GPT_log_general` など solver 本体
- `src/gpoloidal/benchmark_utils.py`
  - metrics / chi2 / 集計など汎用処理
- `src/gpoloidal/scripts/*.py`
  - RT1 などドメイン固有の設定
  - 実験フロー（順次実行）
  - 可視化とレポート保存

## 保存先ルール（重要）

3種類の場所を分ける。

1. `cache`（グローバル・再利用）
- 例: `%LOCALAPPDATA%\\gpoloidal\\cache`
- `observation matrix`, `inducing points` など重い再計算可能データ
- script 名では分けない（同じ設定なら別 script / 別 workspace からも再利用する）

2. `run record`（グローバル・追跡）
- 例: `%LOCALAPPDATA%\\gpoloidal\\records`
- `run_*.json` を保存する正本

3. `analysis_runs`（ローカル作業ディレクトリ・人間向け）
- リポジトリ内 scripts の既定: `<PROJECT_ROOT>/analysis_runs/<experiment>/`
- `--output-dir` で任意の作業ディレクトリへ変更可能
- 構造:
  - `latest/`（上書き）
  - `archive/<timestamp[_name]>/`（蓄積）

## `analysis_runs` の構造

推奨:

- `analysis_runs/<experiment>/latest/...`
- `analysis_runs/<experiment>/archive/20260226_010234/...`
- `analysis_runs/<experiment>/archive/20260226_010234_caseA/...`

`latest/` は入口。  
履歴は `archive/` に残す。

- `run_ref.json` を `archive/...` と `latest/` に置く
  - グローバル `run_*.json`（正本）への参照をローカルから辿るため
- `run record` の保存モード
  - `--record-mode light`（既定）: 軽量 run record のみ
  - `--record-mode archive`: `strict_traceability=true` + dependency manifests 埋め込み + backend result artifacts 保存
  - `--record-mode none` / `--no-run-record`: run record を保存しない

## Script 実装ルール

- Jupyter 実行を考慮して `argparse.parse_known_args()` を使う
  - `ipykernel` の `--f=...` を無視するため
- `--mode dev|analysis` を持たせる
  - `dev`: 既定の `analysis_runs` を `<PROJECT_ROOT>/analysis_runs`
  - `analysis`: 既定の `analysis_runs` を `<cwd>/analysis_runs`
- 相対パスは `cwd` 依存にしすぎない
  - `PROJECT_ROOT = Path(__file__).resolve().parents[...]` で解決する
- cache hit は明示的に print する
- cache は `default_cache_root()` を共通利用する（script ごとの suffix を付けない）
- 重い cache を作る設定（camera/raytrace/inducing/lnum）と、
  推論設定（prior/noise/iters）を意識して分ける

## Config 運用

- script は `--config` を受け取る（JSON/TOML/YAML）
- YAML を使う場合は `PyYAML` が必要（未導入ならエラーで案内）
- 実行時に `config_resolved.json` を `archive/...` に保存する
- `latest/` へもコピーされるので、直近の条件確認が簡単

## 実行例

```powershell
uv run python -m gpoloidal.scripts.rt1_loggp_lingp_benchmark_seq `
  --config configs/rt1/rt1_loggp_lingp_benchmark_seq.example.yaml `
  --mode dev `
  --record-mode light `
  --run-name testA
```

## 運用メモ

- 探索時は `--no-run-record` を使ってもよい
- 本番比較では `run record` を残す
- backend の `results/manifests` を使わない軽量運用でも、`run_*.json` は残す価値が高い
- `src/gpoloidal/scripts/rt1_loggp_lingp_benchmark.py` は互換 wrapper（実装本体は `*_seq.py`）
