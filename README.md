# SER フォーマット処理ツール集

天体撮影などで使用されるSER形式の動画ファイルと、FITS形式の画像ファイルを処理するためのPythonスクリプト集です。

## 概要

このプロジェクトは、SER (Simple Efficient Recording) フォーマットのファイルを解析・表示・処理するためのツールを提供します。SERフォーマットは天体撮影において、高フレームレートでの撮影データを効率的に記録するために使用される形式です。

## 必要なライブラリ

```bash
pip install numpy opencv-python astropy Pillow matplotlib
```

- `numpy`: 数値計算
- `opencv-python`: 画像処理・表示
- `astropy`: FITS ファイル処理
- `Pillow`: 画像処理
- `matplotlib`: ヒストグラム表示・グラフ描画
- `tkinter`: GUI（通常Pythonに付属）

## ファイル構成

### SER フォーマット関連

- **[SER_Info.py](SER_Info.py)** - SERファイルのヘッダー情報とタイムスタンプを表示
- **[SER_Frame_Viewer.py](SER_Frame_Viewer.py)** - 指定フレームの画像を表示
- **[SER_Interactive_Viewer.py](SER_Interactive_Viewer.py)** - インタラクティブなフレーム閲覧
- **[SER_Stacker.py](SER_Stacker.py)** - 高機能スタッカー（開発中）

### FITS フォーマット関連

- **[FITS_Viewer.py](FITS_Viewer.py)** - FITSファイルの表示・操作
- **[FITS_Calc.py](FITS_Calc.py)** - FITS画像間での演算処理

### ドキュメント

- **[仕様.md](仕様.md)** - 詳細な仕様書・使用方法
- **[SER_Stacker.md](SER_Stacker.md)** - SER_Stacker.py の開発記録

## 主要機能

### 1. SER ファイル情報表示 ([SER_Info.py](SER_Info.py))

```bash
python SER_Info.py <ser_file_path> [-a] [-o]
```

- ヘッダー情報（解像度、フレーム数、カラー形式など）を表示
- タイムスタンプ情報の表示
- `-a`: 全タイムスタンプ表示
- `-o`: フレームオフセット表示

### 2. フレーム表示 ([SER_Frame_Viewer.py](SER_Frame_Viewer.py))

```bash
python SER_Frame_Viewer.py <ser_file_path> <frame_number> [options]
```

- 指定したフレームの画像を表示
- ヒストグラム表示（`-hist`オプション）
- 表示範囲調整（`--clip-low`, `--clip-high`）
- エンディアン強制指定（`--force-endian`）

### 3. インタラクティブ閲覧 ([SER_Interactive_Viewer.py](SER_Interactive_Viewer.py))

```bash
python SER_Interactive_Viewer.py <ser_file_path> [options]
```

**キーボード操作:**
- `←/→`: フレーム移動
- `Shift/Ctrl/Alt + ←/→`: 高速移動
- `↑/↓`: ビットシフト調整（16bitデータ）
- `q/Esc`: 終了

### 4. 高機能スタッカー ([SER_Stacker.py](SER_Stacker.py)) - 開発中

```bash
python SER_Stacker.py <ser_file_path> [options]
```

**主な機能:**
- **ダークフレーム補正**: FITS形式のダークフレームを適用
- **スタック処理**: 複数フレームの合成（平均、合計、最大、最小）
- **位置合わせ**: テンプレートマッチングによる自動アライメント
- **移動平均スタック**: リアルタイム処理風のスタック
- **保存機能**: FITS/SER/PNG形式での出力

**モード切替（Tabキー）:**
- `View`: フレーム表示・移動
- `Dark Select`: ダークフレーム選択
- `Stack Config`: スタック設定
- `Align Config`: 位置合わせ設定

### 5. FITS ファイル表示 ([FITS_Viewer.py](FITS_Viewer.py))

```bash
python FITS_Viewer.py <fits_file_path> [options]
```

- ZScale/パーセンタイル正規化
- Bayerパターンデベイヤー処理
- HDU選択・切替
- ビットシフト表示

### 6. FITS 画像演算 ([FITS_Calc.py](FITS_Calc.py))

```bash
python FITS_Calc.py <input1.fits> <operator> <input2.fits> = <output.fits> [options]
```

**対応演算:**
- `+`, `-`, `*`, `/`: 基本演算
- `avg`, `max`, `min`: 統計演算

## 特殊機能

### Bayerパターン対応
- RGGB, GRBG, GBRG, BGGR パターンの自動認識
- デベイヤー処理によるカラー画像生成

### 16bit データ対応
- ビットシフト表示（上位8bitの範囲を調整）
- エンディアン（Big/Little）の自動判定・強制指定

### 位置合わせ機能（SER_Stacker.py）
- マウスドラッグによるROI選択
- テンプレートマッチングによる自動位置合わせ
- Bayerデータでの色情報保持（偶数ピクセル丸め）

## ファイル形式

### SER フォーマット仕様
- **ヘッダー**: 178バイト（メタデータ）
- **画像データ**: フレーム数分のraw画像データ
- **トレーラー**: タイムスタンプ（オプション）

詳細な仕様については [仕様.md](仕様.md) を参照してください。

### FITS フォーマット対応
- Primary HDU および Extension HDU
- 8bit/16bit/32bit整数、float/double対応
- FITSヘッダー情報の読み取り・編集

## 使用例

### 基本的なワークフロー

1. **ファイル情報確認**
   ```bash
   python SER_Info.py sample.ser
   ```

2. **フレーム確認**
   ```bash
   python SER_Frame_Viewer.py sample.ser 100 -hist
   ```

3. **スタック処理**
   ```bash
   python SER_Stacker.py sample.ser --start-frame 50
   ```

4. **FITS演算**
   ```bash
   python FITS_Calc.py light.fits - dark.fits = result.fits
   ```

## 注意事項

- 16bitデータの場合、エンディアンの設定が重要です
- スタック処理では大量のメモリを使用する場合があります
- 位置合わせ機能は平行移動のみ対応（回転・スケール変更は未対応）

## 開発情報

- **言語**: Python 3.x
- **主要ライブラリ**: OpenCV, NumPy, Astropy
- **対応プラットフォーム**: Windows, macOS, Linux

詳細な開発記録は [SER_Stacker.md](SER_Stacker.md) を参照してください。

## ライセンス

このプロジェクトは学習・研究目的で作成されました。

## 参考資料

- [SER Format Description Version 3](https://free-astro.org/images/5/51/SER_Doc_V3b.pdf)
- [FITS Standard](https://fits.gsfc.nasa.gov/) 
