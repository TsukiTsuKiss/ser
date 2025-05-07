import argparse
import sys
import numpy as np
import cv2
from astropy.io import fits
from astropy.visualization import ZScaleInterval # パーセンタイルより良い表示のためZScaleを試す

# tkinter はオプション (画面サイズ取得用)
try:
    import tkinter as tk
    _tkinter_available = True
except ImportError:
    _tkinter_available = False

# --- Key Code Constants ---
KEY_LEFT_ARROW = 2424832
KEY_RIGHT_ARROW = 2555904
KEY_UP_ARROW = 2490368
KEY_DOWN_ARROW = 2621440
KEY_ESC = 27
KEY_D = ord('d')
KEY_Q = ord('q')

def get_bayer_pattern_info(header):
    """FITSヘッダーからベイヤーパターンとオフセットを読み取り、OpenCVの定数を返す"""
    bayer_pattern = header.get('BAYERPAT', '').upper()
    # FITSのオフセット(1-based)からOpenCVのパターンを決定
    # OpenCV Bayerパターンは左上のピクセルを基準にする
    # XBAYROFF, YBAYROFF (FITS standard, 1-based)
    # x_offset = header.get('XBAYROFF', 1) - 1 # 0-basedに変換
    # y_offset = header.get('YBAYROFF', 1) - 1 # 0-basedに変換

    # ★★★ BAYERPAT がない場合にオフセットから推測するロジックを削除 ★★★
    # # 一般的なBAYERPATキーワードがないか、オフセットから判断する
    # if not bayer_pattern and 'CTYPE3' not in header: # データキューブでないことを確認
    #      # オフセットに基づいてパターンを推測 (よくあるパターン)
    #     if (x_offset % 2 == 0) and (y_offset % 2 == 0):
    #          bayer_pattern = 'RGGB' # 仮定
    #          print("警告: BAYERPATが見つかりません。オフセット(0,0)からRGGBと仮定します。", file=sys.stderr)
    #     elif (x_offset % 2 == 1) and (y_offset % 2 == 0):
    #          bayer_pattern = 'GRBG' # 仮定
    #          print("警告: BAYERPATが見つかりません。オフセット(1,0)からGRBGと仮定します。", file=sys.stderr)
    #     elif (x_offset % 2 == 0) and (y_offset % 2 == 1):
    #          bayer_pattern = 'GBRG' # 仮定
    #          print("警告: BAYERPATが見つかりません。オフセット(0,1)からGBRGと仮定します。", file=sys.stderr)
    #     elif (x_offset % 2 == 1) and (y_offset % 2 == 1):
    #          bayer_pattern = 'BGGR' # 仮定
    #          print("警告: BAYERPATが見つかりません。オフセット(1,1)からBGGRと仮定します。", file=sys.stderr)
    #     else:
    #         bayer_pattern = None # 不明

    # OpenCVのデコード定数を返す (BAYERPATがヘッダーにあればそれを使う)
    if bayer_pattern == 'RGGB':
        return cv2.COLOR_BayerRG2BGR, 'RGGB'
    elif bayer_pattern == 'GRBG':
        return cv2.COLOR_BayerGR2BGR, 'GRBG'
    elif bayer_pattern == 'GBRG':
        return cv2.COLOR_BayerGB2BGR, 'GBRG'
    elif bayer_pattern == 'BGGR':
        return cv2.COLOR_BayerBG2BGR, 'BGGR'
    else:
        if bayer_pattern: # 未知のパターンが指定されていた場合
            print(f"警告: 未知のベイヤーパターン '{bayer_pattern}' が指定されています。", file=sys.stderr)
        return None, None

def normalize_image(image_data, original_bitpix, bit_shift=0, clip_low_percent=0.5, clip_high_percent=99.5, use_zscale=True):
    """画像データをビットシフト後に0-255のuint8に正規化する"""

    if image_data.size == 0:
        print("警告: 画像データが空です。", file=sys.stderr)
        return np.zeros((100, 100), dtype=np.uint8), 0, 255 # ダミーの範囲を返す

    # --- ビットシフト処理 (16bit以上の整数のみ) ---
    is_integer = np.issubdtype(image_data.dtype, np.integer)
    effective_data = image_data # デフォルトは元データ
    is_shifted_view = False # ビットシフトされた8bit表示かどうかのフラグ

    if is_integer and original_bitpix >= 16:
        max_shift = max(0, original_bitpix - 8)
        if bit_shift > max_shift:
            print(f"警告: bit_shift({bit_shift})が最大値({max_shift})を超えています。調整します。", file=sys.stderr)
            bit_shift = max_shift
        elif bit_shift < 0:
             bit_shift = 0

        print(f"情報: ビットシフト: {bit_shift} (表示ビット範囲: {7+bit_shift} - {bit_shift})")
        # シフトして下位8bitを抽出 (& 0xFF)
        # NumPyの右シフト >> は符号ビットを維持する場合があるので注意が必要だが、
        # FITSの整数データは通常符号なしなので、この方法で問題ないことが多い。
        # 厳密には fits.open(..., uint=True) で符号なしとして読むか、
        # astropy.io.fits.BSCALE/BZERO を考慮すべきだが、ここでは単純化。
        shifted_data_raw = (image_data >> bit_shift)
        # 8bitマスクを適用
        # shifted_8bit_data = (shifted_data_raw & 0xFF).astype(np.uint8)

        # ★★★ オーバーフロー処理を追加 ★★★
        # シフト後の値が255を超えるピクセルを255にクリップする
        # np.where を使用して 256 以上の値を 255 に置き換え
        clipped_shifted_data = np.where(shifted_data_raw >= 256, 255, shifted_data_raw)
        # 必要であれば、負の値もクリップ (FITSでBITPIX<0の場合など)
        # clipped_shifted_data = np.where(clipped_shifted_data < 0, 0, clipped_shifted_data)

        # このデータを表示用とする (uint8に変換)
        scaled_data = clipped_shifted_data.astype(np.uint8)
        # scaled_data = shifted_8bit_data
        vmin, vmax = 0, 255 # 範囲は固定
        is_shifted_view = True # 8bitスライス表示モード
        # print(f"正規化範囲 (Fixed for Shifted 8bit): vmin=0, vmax=255") # メッセージ変更
        print(f"正規化範囲 (Shifted 8bit with Overflow Clip): vmin=0, vmax=255")

    else:
        # 浮動小数点 or 8bit整数の場合 or 16bitだがシフトしない場合(bit_shift=0)
        # この場合はZScale/Percentileで正規化する
        if bit_shift > 0:
             print(f"情報: BITPIX={original_bitpix}, dtype={image_data.dtype} のためビットシフト表示は無効です。通常の正規化を行います。", file=sys.stderr)
             bit_shift = 0 # 内部状態をリセット

        # NaNやInfを有限な値（例：0）に置き換える
        effective_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 表示範囲の決定 --- (ZScale or Percentile)
        if use_zscale:
            try:
                interval = ZScaleInterval(contrast=0.25)
                vmin, vmax = interval.get_limits(effective_data.astype(np.float32))
                print(f"正規化範囲 (ZScale): vmin={vmin:.3g}, vmax={vmax:.3g}")
            except Exception as e:
                print(f"警告: ZScaleでの範囲計算に失敗しました ({e})。パーセンタイルを使用します。", file=sys.stderr)
                vmin = np.percentile(effective_data, clip_low_percent)
                vmax = np.percentile(effective_data, clip_high_percent)
                print(f"正規化範囲 (Percentile): vmin={vmin:.3g}, vmax={vmax:.3g}")
        else:
            vmin = np.percentile(effective_data, clip_low_percent)
            vmax = np.percentile(effective_data, clip_high_percent)
            print(f"正規化範囲 (Percentile): vmin={vmin:.3g}, vmax={vmax:.3g}")

        # vminとvmaxが不正な場合の処理
        if vmin >= vmax:
            print(f"警告: 計算された vmin ({vmin}) >= vmax ({vmax})。表示範囲を調整します。", file=sys.stderr)
            min_val = np.min(effective_data)
            max_val = np.max(effective_data)
            if min_val == max_val:
                 vmin = min_val - 0.5
                 vmax = max_val + 0.5
            else:
                 vmin = min_val
                 vmax = max_val
                 if vmin >= vmax: vmax = vmin + 1

        # --- 正規化とスケーリング --- (0-255のuint8へ)
        effective_data = effective_data.astype(np.float32)
        clipped_data = np.clip(effective_data, vmin, vmax)
        if vmax - vmin > 1e-6:
            normalized_data = (clipped_data - vmin) / (vmax - vmin)
        else:
            normalized_data = np.zeros_like(clipped_data)
        scaled_data = (normalized_data * 255).astype(np.uint8)

    # 最終的な uint8 画像と、表示に使われた範囲 (vmin, vmax) を返す
    return scaled_data, vmin, vmax

def find_first_image_hdu(hdul):
    """最初の画像データを持つHDUを探す"""
    primary_hdu = hdul[0]
    if primary_hdu.is_image and primary_hdu.data is not None and primary_hdu.data.ndim >= 2:
        print("プライマリHDUに画像データが見つかりました。")
        return primary_hdu, 0
    for i, hdu in enumerate(hdul[1:], 1):
        if hdu.is_image and hdu.data is not None and hdu.data.ndim >= 2:
            print(f"拡張HDU {i} に画像データが見つかりました。")
            return hdu, i
    return None, -1 # 画像HDUが見つからない場合

def main():
    parser = argparse.ArgumentParser(description='FITSファイルをOpenCVで表示します。')
    parser.add_argument('fits_path', help='表示するFITSファイルのパス')
    parser.add_argument('--clip-low', type=float, default=0.5, help='表示範囲の下限パーセンタイル (デフォルト: 0.5, ZScale時/ビットシフト表示時無視)')
    parser.add_argument('--clip-high', type=float, default=99.5, help='表示範囲の上限パーセンタイル (デフォルト: 99.5, ZScale時/ビットシフト表示時無視)')
    parser.add_argument('--use-percentile', action='store_true', help='パーセンタイルクリップを強制的に使用します (デフォルトはZScaleを試行, ビットシフト表示時無視)')
    parser.add_argument('--hdu', type=int, default=None, help='表示するHDUのインデックス (0始まり) を指定します。デフォルトは最初の画像HDU。')
    parser.add_argument('--debayer', action='store_true', help='デベイヤー処理を有効にします。')
    parser.add_argument('--bayer-pattern', type=str, default=None, help='デベイヤー用のベイヤーパターンを強制指定します (例: RGGB, GRBG, GBRG, BGGR)。指定がない場合はヘッダーから自動検出。')
    parser.add_argument('--show-header', action='store_true', help='選択されたHDUのヘッダー情報をコンソールに表示します。')

    args = parser.parse_args()

    # --- 状態変数 --- 
    display_mode = 'mono' # 'mono' or 'color'
    bit_shift = 0         # 右シフト量 (0 to original_bitpix - 8)
    max_shift = 0         # 計算される最大シフト量
    needs_update = True   # 最初に画像処理・表示が必要
    current_bayer_index = 0     # Index for the cycle list
    normalized_mono = None
    debayered_color = None
    can_debayer = False
    can_shift_bits = False
    original_bitpix = None
    image_data = None
    window_title_base = f"FITS Viewer: {args.fits_path}"
    window_title = window_title_base
    debayer_window_name = f"Debayered: {args.fits_path}"
    vmin_disp, vmax_disp = 0, 255

    # --- Bayer Pattern Cycle Definition --- 
    bayer_patterns_cycle = [
        ('BGGR', cv2.COLOR_BayerBG2BGR),
        ('GBRG', cv2.COLOR_BayerGB2BGR),
        ('GRBG', cv2.COLOR_BayerGR2BGR),
        ('RGGB', cv2.COLOR_BayerRG2BGR),
    ]

    try:
        # --- Load Data --- 
        print(f"FITSファイルを開いています: {args.fits_path}")
        with fits.open(args.fits_path) as hdul:
            hdul.info() # ファイルのHDU情報を表示
            if args.hdu is not None:
                if 0 <= args.hdu < len(hdul):
                    image_hdu = hdul[args.hdu]
                    hdu_index = args.hdu
                    if not (image_hdu.is_image and image_hdu.data is not None and image_hdu.data.ndim >=2):
                         print(f"エラー: 指定されたHDU {args.hdu} は有効な画像データを含んでいません。", file=sys.stderr)
                         sys.exit(1)
                    print(f"指定されたHDU {hdu_index} を使用します。")
                else:
                    print(f"エラー: 指定されたHDUインデックス {args.hdu} が範囲外です (0-{len(hdul)-1})。", file=sys.stderr)
                    sys.exit(1)
            else:
                image_hdu, hdu_index = find_first_image_hdu(hdul)
                if image_hdu is None:
                    print(f"エラー: ファイル内に表示可能な画像データが見つかりませんでした。", file=sys.stderr)
                    sys.exit(1)

            image_data = image_hdu.data
            header = image_hdu.header
            original_bitpix = header.get('BITPIX', 0)
            print(f"画像データを読み込みました。HDU: {hdu_index}, Shape: {image_data.shape}, dtype: {image_data.dtype}, BITPIX: {original_bitpix}")

            # --- Restore Original Debayer Initialization Logic --- 
            if args.debayer:
                if args.bayer_pattern:
                    # Manual specification
                    forced_pattern = args.bayer_pattern.upper()
                    bayer_map = {'RGGB': cv2.COLOR_BayerRG2BGR, 'GRBG': cv2.COLOR_BayerGR2BGR,
                                 'GBRG': cv2.COLOR_BayerGB2BGR, 'BGGR': cv2.COLOR_BayerBG2BGR}
                    initial_debayer_code = bayer_map.get(forced_pattern)
                    if initial_debayer_code:
                        initial_bayer_name = forced_pattern
                        print(f"情報: 手動指定されたベイヤーパターン '{initial_bayer_name}' を使用します。")
                    else:
                        print(f"警告: 不明なベイヤーパターン '{args.bayer_pattern}' が指定されました。デベイヤーは無効です。", file=sys.stderr)
                        initial_debayer_code = None
                        initial_bayer_name = None
                else:
                    # Autodetection from header
                    initial_debayer_code, initial_bayer_name = get_bayer_pattern_info(header)
                    if initial_debayer_code:
                        print(f"情報: ヘッダーからベイヤーパターン '{initial_bayer_name}' を検出しました。")
                    else:
                        # Autodetection failed, use BGGR default
                        print(f"情報: ヘッダーからベイヤーパターンを特定/推測できませんでした。デフォルトの 'BGGR' を使用します。", file=sys.stderr)
                        initial_debayer_code = cv2.COLOR_BayerBG2BGR
                        initial_bayer_name = 'BGGR'

                # Set can_debayer based on whether we have a valid code
                can_debayer = initial_debayer_code is not None
                if not can_debayer:
                     display_mode = 'mono'
                     print("警告: デベイヤーが有効化されましたが、最終的に有効なパターンが特定/設定されませんでした。", file=sys.stderr)

            else: # args.debayer is False
                display_mode = 'mono'
                can_debayer = False
                initial_debayer_code = None
                initial_bayer_name = None

            # --- Set initial bayer index based on the determined pattern --- 
            if initial_bayer_name:
                try:
                    current_bayer_index = [name for name, code in bayer_patterns_cycle].index(initial_bayer_name)
                except ValueError:
                    print(f"警告: 初期パターン '{initial_bayer_name}' がサイクルリストにありません。BGGRから開始します。", file=sys.stderr)
                    current_bayer_index = 0 # Default to BGGR index
            else:
                 current_bayer_index = 0 # Default if cannot debayer initially (index 0 is BGGR)

            if can_debayer:
                 print(f"初期 Bayer パターン: {bayer_patterns_cycle[current_bayer_index][0]}")
                 print("情報: ←/→ キーでプレビュー中の Bayer パターンを切り替えられます。")
            # else: No message if debayering is completely disabled

            # --- Bit Shift Check (unchanged) --- 
            if np.issubdtype(image_data.dtype, np.integer) and original_bitpix >= 16:
                can_shift_bits = True; max_shift = original_bitpix - 8; bit_shift = max_shift
                print(f"情報: 上下キーでビットシフトが可能です (BITPIX={original_bitpix}, 初期Shift={bit_shift}, Max Shift={max_shift})。")
            else: max_shift = 0; bit_shift = 0

            # --- Multidimensional Data Handling (unchanged) --- 
            if image_data.ndim > 2: image_data = image_data[0]
            elif image_data.ndim < 2: print("エラー: 画像次元不正", file=sys.stderr); sys.exit(1)

            # --- Header Display (unchanged) --- 
            if args.show_header:
                print("--- Header Information (HDU: {}) ---".format(hdu_index))
                for card in header.cards:
                    if card.keyword in ['COMMENT', 'HISTORY']: print(card.image)
                    else: print(f"{card.keyword:<8} = {repr(card.value):<20}{f' / {card.comment}' if card.comment else ''}")
                print("--- End Header ---")

            # --- Window Creation & Layout (unchanged) --- 
            window_title = f"{window_title_base} (HDU: {hdu_index})" # この時点でのタイトル
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            if can_debayer: cv2.namedWindow(debayer_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            initial_width, initial_height = 640, 480
            if _tkinter_available: # Calculate size based on screen
                try:
                   root=tk.Tk(); screen_width=root.winfo_screenwidth(); screen_height=root.winfo_screenheight(); root.destroy()
                   temp_normalized, _, _ = normalize_image(image_data, original_bitpix, 0, args.clip_low, args.clip_high, not args.use_percentile)
                   img_h, img_w = temp_normalized.shape[:2]; aspect_ratio = img_w / img_h if img_h > 0 else 1
                   target_w=screen_width*0.5; target_h=screen_height*0.5; potential_h=target_w/aspect_ratio; potential_w=target_h*aspect_ratio
                   if potential_h<=target_h: initial_width=int(target_w); initial_height=int(potential_h)
                   else: initial_height=int(target_h); initial_width=int(potential_w)
                   initial_width=max(initial_width, 320); initial_height=max(initial_height, 240)
                except Exception as e: print(f"警告: ウィンドウサイズ初期化エラー: {e}", file=sys.stderr)
            cv2.resizeWindow(window_title, initial_width, initial_height)
            if can_debayer: cv2.resizeWindow(debayer_window_name, initial_width, initial_height)
            try: # Position windows
                 cv2.moveWindow(window_title, 0, 0)
                 if can_debayer: cv2.moveWindow(debayer_window_name, initial_width, 0)
            except cv2.error as e: print(f"警告: ウィンドウ配置エラー: {e}", file=sys.stderr)

            # --- Print Controls (Update message) --- 
            print("\n操作方法:")
            print("  Q または Esc: 終了")
            if can_debayer:
                 print("  D: モノクロ/カラー表示切り替え (メインウィンドウ)")
                 print("  ←/→: デベイヤーパターン切り替え (BGGR<->GBRG<->GRBG<->RGGB)") # Add arrow key info
            if can_shift_bits:
                 print(f"  ↑/↓: ビットシフト変更 (Max={max_shift}, Min=0)")
            print("--------------------")

            # --- Main Loop --- 
            while True:
                if needs_update:
                    print("表示更新中...")
                    # Get current Bayer pattern based on index
                    current_bayer_name, current_debayer_code = bayer_patterns_cycle[current_bayer_index]
                    # Normalize mono image
                    normalized_mono, vmin_disp, vmax_disp = normalize_image(
                        image_data, original_bitpix, bit_shift,
                        args.clip_low, args.clip_high, not args.use_percentile
                    )
                    # Debayer using current pattern
                    debayered_color = None
                    if can_debayer:
                        try:
                            debayered_color = cv2.cvtColor(normalized_mono, current_debayer_code)
                        except cv2.error as e:
                             print(f"警告: デベイヤー処理失敗 ({current_bayer_name}): {e}", file=sys.stderr)
                    # Update window titles
                    mode_str = "Color" if display_mode == 'color' else "Mono"
                    bit_str = f" (Bits: {7+bit_shift}-{bit_shift}, Shift: {bit_shift})" if can_shift_bits else ""
                    range_str = f" [Range: {vmin_disp:.3g}-{vmax_disp:.3g}]" if not (can_shift_bits and bit_shift > 0) else ""
                    main_title = f"{window_title_base} (HDU: {hdu_index}) - {mode_str}{bit_str}{range_str}"
                    if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) != -1: cv2.setWindowTitle(window_title, main_title)
                    else: break
                    if can_debayer and cv2.getWindowProperty(debayer_window_name, cv2.WND_PROP_AUTOSIZE) != -1:
                         debayer_title = f"Debayered: {current_bayer_name}" # Show current pattern
                         cv2.setWindowTitle(debayer_window_name, debayer_title)
                    needs_update = False

                # Display images
                display_image_main = debayered_color if (display_mode == 'color' and debayered_color is not None) else normalized_mono
                if display_image_main is not None: cv2.imshow(window_title, display_image_main)
                if can_debayer and cv2.getWindowProperty(debayer_window_name, cv2.WND_PROP_VISIBLE) >= 1:
                     display_image_debayer = debayered_color if debayered_color is not None else np.zeros_like(normalized_mono)
                     cv2.imshow(debayer_window_name, display_image_debayer)

                # --- Key Handling (Remove Debug Print) --- 
                key = cv2.waitKeyEx(30) & 0xFFFFFF
                # if key != -1 and key != 255 and key != 0:
                #     print(f"DEBUG Key Code: {key}") # Remove debug print

                if key == KEY_Q or key == KEY_ESC: break
                elif key == KEY_D and can_debayer:
                    display_mode = 'color' if display_mode == 'mono' else 'mono'
                    print(f"メインウィンドウ表示モード: {display_mode}")
                    needs_update = True
                elif key == KEY_UP_ARROW and can_shift_bits: # Up arrow
                    if bit_shift < max_shift: bit_shift += 1; needs_update = True
                    else: print("ビットシフト最大", file=sys.stderr)
                elif key == KEY_DOWN_ARROW and can_shift_bits: # Down arrow
                    if bit_shift > 0: bit_shift -= 1; needs_update = True
                    else: print("ビットシフト最小", file=sys.stderr)
                elif key == KEY_LEFT_ARROW and can_debayer: # Left arrow
                    current_bayer_index = (current_bayer_index - 1 + len(bayer_patterns_cycle)) % len(bayer_patterns_cycle)
                    print(f"Bayerパターン変更: {bayer_patterns_cycle[current_bayer_index][0]}")
                    needs_update = True
                elif key == KEY_RIGHT_ARROW and can_debayer: # Right arrow
                    current_bayer_index = (current_bayer_index + 1) % len(bayer_patterns_cycle)
                    print(f"Bayerパターン変更: {bayer_patterns_cycle[current_bayer_index][0]}")
                    needs_update = True

                # Window closing check (unchanged)
                try:
                     main_visible = cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) >= 1
                     debayer_visible = True
                     if can_debayer:
                         # Check if debayer window handle is valid before getting property
                         if cv2.getWindowProperty(debayer_window_name, cv2.WND_PROP_AUTOSIZE) != -1:
                             debayer_visible = cv2.getWindowProperty(debayer_window_name, cv2.WND_PROP_VISIBLE) >= 1
                         else:
                             debayer_visible = False # Window doesn't exist

                     if not main_visible or (can_debayer and not debayer_visible):
                          print("ウィンドウが閉じられました。"); break
                except cv2.error: print("ウィンドウアクセスエラー。終了します。"); break

            cv2.destroyAllWindows()
            print("ビューアーを終了しました。")

    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {args.fits_path}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"エラー: ファイルを開けません。FITS形式でないか、破損している可能性があります。 ({e})", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
         missing_lib = str(e).split("'")[-2]
         install_cmd = "pip install numpy opencv-python astropy" + (" tkinter" if not _tkinter_available else "")
         print(f"エラー: 必要なライブラリ '{missing_lib}' が見つかりません。`{install_cmd}` を実行してください。", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 