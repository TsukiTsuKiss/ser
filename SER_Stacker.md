# SER_Stacker.py 開発の記録

このドキュメントは、`SER_Stacker.py` スクリプトの機能追加と改善の過程を記録したものです。

## ベースとなったプログラム

`SER_Stacker.py` は、以下の既存スクリプトの機能やロジックを参考にしたり、取り込んだりしながら開発されました。

*   **`SER_Info.py`**: SERファイルのヘッダー読み込み、ファイル構造の基本。
*   **`SER_Frame_Viewer.py` / `SER_Interactive_Viewer.py`**: SERフレームデータの抽出・表示、ビットシフト処理、インタラクティブ操作の基礎。
*   **`FITS_Viewer.py`**: FITSファイルの読み込み（ダーク用）、表示正規化（ZScale等）、デベイヤー処理。
*   **`FITS_Calc.py`**: FITSファイルのHDU探索ロジック。

これらのスクリプトで実装された機能が、`SER_Stacker.py` の基盤となり、ダーク補正、スタック、アライメントといった高度な機能が追加されました。

## 1. 基本機能の実装

*   **初期段階:** SER ファイルを読み込み、指定されたフレームを OpenCV ウィンドウに表示する基本的なビューア機能から開発を開始しました。
*   **フレームナビゲーション:** キーボード操作 (`j`, `k` キーなど) によるフレーム間の移動機能を追加しました。

## 2. ダークフレーム処理機能

*   **ダークフレーム選択:** カレントディレクトリにある FITS 形式のダークフレームファイルをリスト表示し、選択できる機能を追加しました (`dark_select` モード)。
*   **ダークフレーム適用:** 選択したダークフレームを `Enter` キーで適用し、表示フレームから減算する処理を実装しました。画像サイズが一致しない場合は適用しないチェックも追加しました。
*   **ダークフレーム解除:** 適用中のダークフレームを `d` キーで解除する機能を追加しました。
*   **計算用ビットシフト:** ダークフレーム減算時の精度を調整するため、ダークフレームデータに適用するビットシフト量を `[` および `]` キーで変更できる機能を追加しました (16bit データの場合に有効)。

## 3. ウィンドウレイアウトとUI改善

*   **複数ウィンドウ化:** 処理結果を分かりやすく表示するため、単一ウィンドウから複数ウィンドウ構成に変更しました。
    *   上段: 元フレーム(ダーク減算後)、ログ/ダーク選択/プレビュー、デベイヤーフレーム
    *   下段: スタック結果(Raw)、スタック結果(Dark Sub)、デベイヤー後スタック結果
*   **ダークプレビュー:** `dark_select` モードで、ファイルリスト表示 (`j` キー) と選択中ダークファイルのプレビュー表示 (`k` キー) を切り替えられるようにしました。
*   **ログ表示:** `view` モードや `stack_config` モードでは、中央ウィンドウに操作ログを表示するようにしました。

## 4. スタック機能

*   **スタック設定モード:** スタック処理の設定を行うための `stack_config` モードを追加しました。
*   **スタックパラメータ設定:** スタックするフレーム数と、スタック演算方法 (`avg`, `sum`, `max`, `min`) を `j`/`k`/`i`/`m` キーで設定できるようにしました。
*   **スタック実行:** `Enter` キーで設定に基づいてスタック処理を実行し、下段のウィンドウに結果を表示する機能を追加しました。ダーク減算前の Raw データと減算後のデータの両方をスタックします。
*   **スタック表示正規化:** スタック結果ウィンドウの表示コントラストを調整するため、正規化モード (`Bitshift`, `Auto`, `Fixed`) を `b` キーで切り替えられる機能を追加しました。
*   **スタック結果保存 (キー入れ替え後):**
    *   `s` キー: **移動平均スタック**の結果を複数フレームの SER ファイル (`.ser`、ファイル名に `_running` が付く) として保存します。Raw データ (`_raw_running`) と Dark Subtracted データ (`_darks_running`、ダーク適用時のみ) の両方が保存されます。出力されるフレーム数は `入力総フレーム数 - スタックフレーム数 + 1` となります。エンディアンはデフォルトで Little Endian で保存されますが、`--ser-player-compat` オプション指定時はヘッダーフラグのみ `0` (Big Endian) になります。
    *   `w` キー: 現在の**単一フレーム**スタック結果を FITS 形式 (`.fts`) および PNG 形式 (表示プレビュー、ファイル名に `_preview` が付く) で保存します。Raw データと Dark Subtracted データの両方が保存されます。
    *   (旧 `v` キー: AVI 保存機能は現在削除されています。)

## 5. アライメント (位置合わせ) 機能

*   **アライメント設定モード:** 位置合わせの設定を行う `align_config` モードを追加しました。
*   **基本操作:** アライメント機能の有効/無効 (`e` キー)、基準フレームの設定 (`Enter` キー)、マウスドラッグによる ROI 選択、設定リセット (`d` キー) を実装しました。
*   **モード内フレーム移動:** `align_config` モード内でも `j`/`k` キー等でフレームを移動し、基準フレームを選択できるように改善しました。
*   **アルゴリズム試行錯誤:**
    *   当初、OpenCV の `phaseCorrelate` や `findTransformECC` を試しましたが、低コントラストなデータや Bayer データに対して十分な精度が得られず、スタック結果がブレる問題が発生しました。
    *   原因として、元データのコントラスト不足や、Bayer パターンへの単純なサブピクセルシフト適用の問題が考えられました。
    *   計算前の ROI データ正規化、表示ビットシフトの計算への適用などを試みましたが、決定的な改善には至りませんでした。
*   **`matchTemplate` の採用:** ユーザーの C++ アプリケーションでの成功事例を参考に、位置合わせアルゴリズムを `cv2.matchTemplate` に変更しました。
    *   **前処理:** 基準 ROI (テンプレート) と検索領域の両方を、現在の表示用ビットシフトを適用して 8bit グレースケールに変換してからマッチングを行うようにしました。
    *   **ダーク減算処理の追加 (精度向上):** さらに精度を向上させるため、上記の前処理の前に、適用されているダークフレームを減算する処理を追加しました。これにより、固定パターンノイズ等の影響を受けにくくなり、より安定したマッチングが期待できます。
    *   **検索領域:** ROI の周囲 +/- 256 ピクセルを検索範囲としました。
    *   **結果:** この方法により、低コントラストなデータでも特徴を捉えやすくなり、位置合わせの精度が大幅に向上し、スタックのブレが解消されました。
*   **Bayer データ対応:** Bayer パターンの SER ファイルの場合、色情報を維持するため、`matchTemplate` で計算されたシフト量を適用する際に、**最も近い偶数ピクセルに丸める**処理を維持しました。
*   **互換性オプション:** 特定のビューア（例: SER Player）がヘッダーフラグを正しく解釈しない場合があるため、コマンドラインオプション `--ser-player-compat` を追加しました。このオプションを指定すると、`w` キーでの保存時にヘッダーの `LittleEndian` フラグのみ `0` (Big Endian を示す) が書き込まれます（データ自体は Little Endian のままです）。

## 6. UI/表示の調整

*   **ROI 表示:** ROI 選択中 (黄色枠) と確定後 (白色枠) の表示について、線の太さ、影の有無、リサイズ時の位置ずれ、Bayer データでの色表示などをデバッグ・修正し、最終的に線太さ 1、影なし、確定枠は元座標から再計算する方式に落ち着きました。

## 7. ドキュメント

*   開発の区切りごとに `仕様.md` を更新し、機能、操作方法、および位置合わせのノウハウなどを記録しました。

## 8. エンディアン処理の修正

*   移動平均スタック SER 保存機能 (`w` キー) において、当初、出力 SER ファイルのエンディアンが入力ファイルに依存していましたが、これによりヘッダー情報と実際のデータエンディアンが不一致になる問題が発生しました。
*   この問題を解決するため、`w` キーで保存される SER ファイルは、**常に Little Endian** で書き込まれるように仕様を変更しました。具体的には、ヘッダーの `LittleEndian` フラグには常に `1` を設定し、画像データは必要に応じてバイトスワップを行ってから書き込むように修正しました。

このように、対話的な開発プロセスを通じて、多くの機能追加と試行錯誤、特にアライメント機能におけるアルゴリズム選択の重要性を経験しながら `SER_Stacker.py` は進化しました。 