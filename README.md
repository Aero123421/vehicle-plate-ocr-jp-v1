# vehicle-plate-ocr-jp-v1

Japanese vehicle detection, tracking & terminal multi-frame plate OCR — **v0.6.5 / Apache-2.0**

日本の車両・ナンバープレートを検出し、複数フレームの光学的な特徴量から車番を読み出すモデルです。配布する **7 個の INT8 ONNX グラフと 3 個の校正 JSON は合計 10,643,833 bytes（10.15 MiB）**。`web/models/` に同梱し、実写評価データや学習チェックポイントを取得せずに利用できます。

## 推論の構成

1. **車両・プレート検出**：時間方向の特徴量を持つ D-FINE / HGNetV2 系検出器。車両の **64×64 シルエット cutout** も出力します。
2. **局在化・追跡**：射影 4 隅 **IC-STN** がプレートを正規化し、Python パイプラインの **PMBM 追跡**が車両とプレートの観測を関連付けます。
3. **終端 Bayesian OCR**：フレーム間で語彙に依存しない特徴量・精度・矛盾量を蓄積し、終端で日本のプレート文法と校正済み minimum Bayes risk（MBR）判定を適用します。

JP7029732B2 に対する patent-clean な設計回避の実装契約は **`COLLECTING -> SEALED -> FINALIZED`** です。収集中の文字読み出しは **0 回**。各 collector の終端読み出しは 1 回のみで、フレームごとの文字列多数決・文字列分散による再認識指示は使いません。公開 predictor はクリップ EOF まで collector を保持します。詳細は [アーキテクチャ](docs/architecture.md) と [セッション実装](src/vpocr_ocr/session_v2.py) を参照してください。

## Browser Demo

GitHub の **Settings → Pages → Source: GitHub Actions** を選ぶと、`main` への push または手動実行で同梱の Pages workflow が `web/` を配信します。公開先 URL は workflow の environment に表示されます。

ローカルでは次の 1 コマンドで起動できます。

```bash
python3 -m http.server -d web 8080
```

`http://localhost:8080` を開き、同梱の合成ゲート映像、ドラッグ＆ドロップした動画、またはカメラを選択します。「認識を開始」で特徴量を収集し、動画の終了または「終了して読み出す」で終端結果を表示します。カメラは HTTPS または localhost とブラウザーのアクセス許可が必要です。カメラは終了時に停止します。

**100% client-side local inference / 映像アップロード 0 bytes**。モデルと ONNX Runtime WASM も同じサイトから読み込み、外部 CDN・解析サービスは利用しません。初回にはモデルに加えて約 15 MiB の WASM ランタイムを読み込みます。ローカル動画はサーバーに送信しません。

ブラウザー版は動作を確認するための簡易 IoU 追跡です。Python 版の PMBM・Smart-Zoom・画素サンプリングとの全体的な精度同等性は未検証で、下記ベンチマークの対象には含みません。最大 12 FPS でサンプリングし、フレームごとに最大 16 プレート候補・12 車両、クリップ全体で最大 256 プレートトラックを扱います。処理速度は端末に依存します。MP4/H.264 など、そのブラウザーがデコード可能な動画を利用してください。

`grammar_tables.json` は Python の `PlateVocab` と `_terminal_tables(JPPlateGrammar(...))` から決定的に生成します。254 トークン、全地域・分類番号・かな、5 種別の有効性表を含みます。`valid_base64` は **zlib → little-endian packbits → C-order `(5, N_cls, N_kana)`** の順で復元し、末尾の padding bits は無視します。`web/terminal.mjs` は同じ有限因子の終端 MBR 計算を実装します。

同梱サンプルは実写を一切含まず、`Scene3DGPUEngine(domain_split="val_holdout")` で生成した 768×448・24 フレーム・12 FPS の H.264 / yuv420p MP4 です。生成条件は [sample provenance](web/samples/synthetic_gate_demo.json) に記録しています。

## Python Quickstart

Python 3.12 以上を使用してください。リポジトリを clone したディレクトリで実行します。

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
vehicle-plate-ocr-jp-v1 --version
vehicle-plate-ocr-jp-v1 info
vehicle-plate-ocr-jp-v1 predict video.mp4 --json
```

ONNX 推論は CPU の INT8 グラフのみを使用します。既存の研究・合成コード用に PyTorch と ONNX 関連依存もインストールします。`vpocr-synth`、`vpocr-ocr`、`vpocr-det` の既存 CLI も引き続き利用できます。

```python
from vehicle_plate_ocr_jp_v1 import create_predictor

predictor = create_predictor()
results = predictor.predict_clip("video.mp4")
print(results)  # [{"serial": "1234"}, {"serial": "5678", "full_text": "…"}, ...]
```

`predict_clip` は採用した終端結果を返し、全文字列を検証できた場合にだけ `full_text` を含めます。採用結果がない場合は `[]` です。公開クラス `VehiclePlateVideoPipeline`、`MultiRateEdgeCascadePipeline`、`MFPlateOCRSessionV2`、`SessionConfigV2`、`TerminalCalibrationV2`、`OCRDecisionV2` も新パッケージから import できます。

既存の `out/models/` を持つ研究 checkout では、無設定時に元の frozen adapter に委譲します。新規 clone と wheel では同梱の `web/models/` を使います。`create_predictor(root=...)` は指定したルートを使用し、モデルがない場合は失敗します。自動ダウンロードや FP32 fallback は行いません。

## v0.6.5 のベンチマーク概要と SLA 範囲

v0.6.5 は配布・パッケージ整備のリリースです。モデルと校正値は変更せず、以下は **既存の測定記録**を引用しています。このリリース作業では学習・校正・ベンチマークを再実行していません。

| 既存測定の条件 | 記録値 |
| --- | --- |
| Nano INT8 / ALPR60、60 枚の人手 GT | 生の 4 フィールド完全一致 70.0%、一連指定番号の正解率 95.0% |
| 同条件、終端採用 | 57/60 採用、strict emitted precision 92.98% |
| 同条件、`FULL_4FIELD` | 20 件採用、完全一致 20/20 |
| 同条件、`CORE_VERIFIED` | 37 件、一連指定番号の正解 37/37 |
| 街路 hard negatives 400 枚 | full-scene false accepts 0/400 |
| Wave-5 sealed、未見動画 8 本・読み取り可能 serial 6 件 | serial recall 2/6（33.3%）、余剰予測 1 件、full plate 正解 0/2 |

出典：[Gen-6 実測記録（記録内の版は v0.6.0）](docs/benchmarks/v6_gen6_realworld_benchmark.json)、[Wave-5 sealed 記録](docs/benchmarks/v6_wave5_video_eval.json)。静止画の切り出し評価と動画の検出・追跡を含む評価では条件が異なります。20 件・37 件での観測精度は運用時の保証値ではありません。

**遠距離・低解像度で 4 フィールド完全認識を保証する SLA はありません。** `CORE_VERIFIED` では不確かな地域・分類番号・かなを `'?'` でマスクします。`OCRDecisionV2.format_verified_text()` またはブラウザーの終端表示を使用し、推測した上段文字を確定値として扱わないでください。公開 predictor の JSON では `CORE_VERIFIED` の全文字列を省略します。視認性や identity が不足する場合は `UNREADABLE` / `TRACK_MIXED` として棄却します。実環境の速度・精度は端末、撮影距離、照明、画素数に依存します。

## 配布と開発

`web/`、`web/models/`、`web/samples/` は公開対象です。`out/`（実写評価データ、旧チェックポイント、監査出力）、`docs/refs/`（外部資料キャッシュ）、`web/node_modules/` は公開対象から除外します。wheel には同じ INT8 バンドルと語彙データを含めます。

```bash
# 文法表のみ再生成（モデル・校正値は変更しません）
PYTHONPATH=src python tools/export_release_grammar.py

# 合成デモ再生成（Scene3DGPUEngine + ffmpeg/libx264）
# PATH に ffmpeg がない場合のみ pip install imageio-ffmpeg を使用できます。
PYTHONPATH=src python tools/render_release_demo.py

# 公開パッケージの短いテストのみ
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_vehicle_plate_ocr_jp_v1_package.py -p no:cacheprovider -q
```

CI は公開パッケージ、Gen-6 契約、地域一覧と文字集合を検証します。`out/` を持たない clone では非配布の旧アーカイブのチェックだけが skip され、公開契約のチェックは実行されます。

### 再学習と再校正（純粋合成データのみ）

全ステージの学習・検証・校正入力は手続き生成した合成データです。検出・4 隅推定は `train` / `val_holdout` / `cal_holdout` を分離し、INT8 化後の Stage-1 / Stage-2 校正を `stage1_int8_cal.json` / `stage2_int8_cal.json` に記録します。OCR は乱数初期化した Base から Nano へ合成データだけで蒸留し、4 個の ONNX グラフと provenance を出力します。検出器の固定 D-FINE バックボーンには公開 Apache-2.0 重みを使い、初回取得にネットワーク接続が必要です。学習用画像や評価映像の取得は不要です。

上記の Python 環境を有効にしてリポジトリ直下で実行します。検出・4 隅推定の学習は **CUDA / bfloat16 対応 GPU** が必要です。OCR は CPU でも実行できますが、全学習には GPU を推奨します。`--help` はモデルや学習依存を読み込まず、ファイルを作成しません。

```bash
python tools/train_det_quad.py --help
python tools/train_ocr.py --help
python tools/calibrate_policy.py --help

# 短縮した学習・エクスポート経路の確認（隔離出力必須、GPU と実行時間が必要）
python tools/train_det_quad.py --dry-run --out-dir out/dry-det/models --cal-dir out/dry-det/calibration
python tools/train_ocr.py --dry-run --out-dir out/dry-ocr/models

# 全学習：既定出力は gitignore 対象の out/models と out/calibration
python tools/train_det_quad.py
python tools/train_ocr.py

# 学習済みの入れ子構成を再校正し、スキーマ・サイズ・合成 MP4 推論を確認
python tools/calibrate_policy.py --models-dir out/models --calibration-out out/calibration/gen6.json
python tools/calibrate_policy.py --models-dir out/models --calibration-out out/calibration/gen6.json --audit-only

# 新規 clone の同梱バンドルを再学習せず監査（7 グラフ + 3 JSON、上限 15.000 MiB）
python tools/calibrate_policy.py --audit-only

# 同梱バンドルを合成 cal_holdout だけで再校正する場合
# 既定では web/models/gen6.json を更新するため、公開バンドル更新時だけ実行
python tools/calibrate_policy.py
```

学習スクリプトはチェックポイント、FP32 / INT8 ONNX、数値 parity 検証、ソース・フォント・教師モデルのハッシュ記録まで実行します。短縮実行は性能検証の代わりにはなりません。`calibrate_policy.py` は CPU で終端判定と追跡方策を校正し、公開 `create_predictor()` による短い手続き生成 MP4 の smoke test と `__pycache__` の掃除を行います。`--audit-only` は指定した校正 JSON を再利用し、校正値・モデルを更新しません。配布バンドルの既存合計は **10.15 MiB** です。

再学習の成果物は `web/models/` に自動コピーしません。ブラウザー用に配布するときは、Nano の検出・cutout（`out/models/onnx_temporal_det_nano/`）、`stage2.int8.onnx`（`out/models/onnx_det_v4/`）、OCR の 4 個の INT8 グラフ（`out/models/onnx_v2_nano/`）、3 個の校正 JSON（`out/calibration/`）を明示的に `web/models/` へコピーし、`--audit-only` で再確認してください。文法表は上記の再生成コマンドで更新できます。

## ライセンス

プロジェクトと配布モデルは **Apache-2.0** です。[LICENSE](LICENSE)、[NOTICE](NOTICE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。Apache ライセンス本文は [Apache Software Foundation の公式原文](https://www.apache.org/licenses/LICENSE-2.0.txt) を収録しています。

D-FINE、Hugging Face Transformers の HGNetV2 / D-FINE modeling port、MIT ライセンスの ONNX Runtime / onnxruntime-web に謝辞を示します。`data/fonts/` の BIZ UDPGothic、M PLUS Rounded 1c、Zen Maru Gothic は **SIL Open Font License 1.1** で、それぞれの `*-OFL.txt` が適用されます。ブラウザー配布の runtime ライセンス・第三者通知は `web/vendor/` に同梱しています。

名称と版番号は公開パッケージに統一しています。追加の謝辞は `NOTICE` に記載しています。ベンチマーク JSON は集計値のみを収録し、個別予測・評価画像・ローカルパスを含みません。
