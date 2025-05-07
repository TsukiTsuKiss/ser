import argparse
import sys
import numpy as np
import cv2
from astropy.io import fits
from astropy.visualization import ZScaleInterval # パーセンタイルより良い表示のためZScaleを試す
import datetime # Added for history timestamp

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
# KEY_D = ord('d') # Not used in Calc
# KEY_Q = ord('q') # Replaced by Y/N
KEY_Y = ord('y')
KEY_N = ord('n')

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

def normalize_image(image_data, original_bitpix_hint, bit_shift=0, clip_low_percent=0.5, clip_high_percent=99.5, use_zscale=True):
    """画像データをビットシフト後に0-255のuint8に正規化する (float結果にもシフト範囲適用)"""
    if image_data.size == 0:
        return np.zeros((100, 100), dtype=np.uint8), 0, 255

    is_integer = np.issubdtype(image_data.dtype, np.integer)
    can_potentially_shift = original_bitpix_hint >= 16 # Use the hint

    # --- Path 1: Actual Bit Shift for Integers --- 
    if is_integer and can_potentially_shift and bit_shift > 0:
        max_shift = max(0, original_bitpix_hint - 8)
        bit_shift = min(max(0, bit_shift), max_shift)
        shifted_data_raw = (image_data >> bit_shift)
        clipped_shifted_data = np.where(shifted_data_raw >= 256, 255, shifted_data_raw)
        # Optional negative clip
        scaled_data = clipped_shifted_data.astype(np.uint8)
        return scaled_data, 0, 255 # Return actual shifted data

    # --- Path 2: Normalization using calculated or ZScale/Percentile range --- 
    vmin, vmax = None, None # Reset

    # Determine vmin, vmax based on shift state OR ZScale/Percentile
    if can_potentially_shift and bit_shift > 0:
        # If shifting would be active, calculate the equivalent range
        # This applies even to float data (like result_data)
        vmin = 0.0
        vmax = float(255 << bit_shift) # Scale the 0-255 range
        # print(f"DEBUG: Applying calculated shift range: vmin={vmin}, vmax={vmax}")
    else:
        # Use ZScale/Percentile for non-shiftable data or shift=0
        effective_data_norm = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        if use_zscale:
            try:
                interval = ZScaleInterval(contrast=0.25)
                vmin, vmax = interval.get_limits(effective_data_norm.astype(np.float32))
            except Exception:
                vmin = np.percentile(effective_data_norm, clip_low_percent)
                vmax = np.percentile(effective_data_norm, clip_high_percent)
        else:
            vmin = np.percentile(effective_data_norm, clip_low_percent)
            vmax = np.percentile(effective_data_norm, clip_high_percent)
        # Handle vmin >= vmax case
        if vmin >= vmax:
             min_val = np.min(effective_data_norm); max_val = np.max(effective_data_norm)
             if min_val == max_val: vmin = min_val - 0.5; vmax = max_val + 0.5
             else: vmin = min_val; vmax = max_val
             if vmin >= vmax: vmax = vmin + 1
        # print(f"DEBUG: Using ZScale/Percentile range: vmin={vmin}, vmax={vmax}")

    # --- Apply normalization using the determined vmin, vmax --- 
    # Ensure data is float for scaling, handle potential NaN/Inf
    effective_data_display = image_data.astype(np.float32)
    # Use calculated vmax/vmin if they exist from ZScale/Percentile, otherwise use reasonable fallbacks
    fallback_vmax = np.max(effective_data_display) if np.isfinite(np.max(effective_data_display)) else 1.0
    fallback_vmin = np.min(effective_data_display) if np.isfinite(np.min(effective_data_display)) else 0.0
    effective_data_display = np.nan_to_num(effective_data_display, nan=0.0, posinf=vmax if vmax is not None else fallback_vmax, neginf=vmin if vmin is not None else fallback_vmin)

    # Ensure vmin/vmax are set if calculation above failed somehow (should not happen)
    if vmin is None: vmin = 0.0
    if vmax is None: vmax = 255.0
    if vmin >= vmax: vmax = vmin + 1e-6 # Prevent division by zero

    clipped_data = np.clip(effective_data_display, vmin, vmax)
    if vmax - vmin > 1e-6:
        normalized_data = (clipped_data - vmin) / (vmax - vmin)
    else:
        normalized_data = np.zeros_like(clipped_data)
    scaled_data = (normalized_data * 255).astype(np.uint8)

    return scaled_data, vmin, vmax

def find_first_image_hdu(hdul):
    """最初の画像データを持つHDUを探す"""
    primary_hdu = hdul[0]
    if primary_hdu.is_image and primary_hdu.data is not None and primary_hdu.data.ndim >= 2:
        # print("プライマリHDUに画像データが見つかりました。") # Calcでは抑制
        return primary_hdu, 0
    for i, hdu in enumerate(hdul[1:], 1):
        if hdu.is_image and hdu.data is not None and hdu.data.ndim >= 2:
            # print(f"拡張HDU {i} に画像データが見つかりました。") # Calcでは抑制
            return hdu, i
    return None, -1 # 画像HDUが見つからない場合

# --- Calculation Specific Function ---
def perform_calculation(data1, data2, operation):
    """NumPy配列データに対して指定された演算を実行する"""
    print(f"演算 '{operation}' を実行中...", end='')
    data1 = data1.astype(np.float32); data2 = data2.astype(np.float32)
    result_data = None
    with np.errstate(divide='ignore', invalid='ignore'):
        if operation == '+': result_data = data1 + data2
        elif operation == '-': result_data = data1 - data2
        elif operation == '*': result_data = data1 * data2
        elif operation == '/':
            result_data = np.divide(data1, data2, out=np.zeros_like(data1), where=data2!=0)
            if np.any(data2 == 0): print(" [警告: ゼロ除算発生->0]", end='')
        elif operation.lower() == 'avg': result_data = (data1 + data2) / 2.0
        elif operation.lower() == 'max': result_data = np.maximum(data1, data2)
        elif operation.lower() == 'min': result_data = np.minimum(data1, data2)
        else: raise ValueError(f"未対応の演算子です: {operation}")
    print(" 完了")
    return result_data

def main():
    # --- Argument Parsing (Using sys.argv) ---
    if not (len(sys.argv) == 6 or len(sys.argv) == 8) or sys.argv[4] != '=':
        print("使用法: python FITS_Calc.py <入力1> <演算子> <入力2> = <出力> [--bayer-pattern PATTERN]", file=sys.stderr)
        print("対応演算子: +, -, *, /, avg, max, min", file=sys.stderr)
        print("PATTERN: RGGB, GRBG, GBRG, BGGR", file=sys.stderr)
        sys.exit(1)

    input_file1 = sys.argv[1]
    operation = sys.argv[2]
    input_file2 = sys.argv[3]
    output_file = sys.argv[5]
    forced_bayer_pattern = None

    if len(sys.argv) == 8:
        if sys.argv[6].lower() == '--bayer-pattern':
            forced_bayer_pattern = sys.argv[7].upper() # Store uppercase
        else:
            print("エラー: 不明なオプション引数です。--bayer-pattern のみ対応しています。", file=sys.stderr)
            sys.exit(1)

    print(f"入力1: {input_file1}"); print(f"演算:  {operation}"); print(f"入力2: {input_file2}"); print(f"出力:  {output_file}")
    if forced_bayer_pattern: print(f"強制 Bayer パターン指定: {forced_bayer_pattern}")

    # --- State Variables ---
    bit_shift = 0
    max_shift = 0
    needs_update = True
    current_bayer_index = 0
    can_debayer_initially1 = False
    can_debayer_initially2 = False # For input 2
    can_shift_bits = False
    original_bitpix1 = 0
    original_bitpix2 = 0 # For input 2
    save_confirmed = False # For save confirmation
    # Image data
    data1, data2, result_data = None, None, None
    header1, header2 = None, None
    norm_data1, norm_data2, norm_result = None, None, None
    debayered1, debayered2, debayered_result = None, None, None

    # Bayer Pattern Cycle (same as viewer)
    bayer_patterns_cycle = [ ('BGGR', cv2.COLOR_BayerBG2BGR), ('GBRG', cv2.COLOR_BayerGB2BGR), ('GRBG', cv2.COLOR_BayerGR2BGR), ('RGGB', cv2.COLOR_BayerRG2BGR) ]

    # Window Names
    norm_window_names = [f"Input 1 ({input_file1})", f"Input 2 ({input_file2})", f"Output Preview ({output_file})"]
    debayer_window_names = [f"Input 1 Debayered", f"Input 2 Debayered", f"Output Debayered (In1 Pattern)"]

    try:
        # --- Load Input File 1 ---
        print(f"入力ファイル1を読み込み中: {input_file1}")
        with fits.open(input_file1) as hdul1:
            image_hdu1, hdu_index1 = find_first_image_hdu(hdul1)
            if image_hdu1 is None: raise ValueError(f"{input_file1} 内に画像データが見つかりません。")
            data1 = image_hdu1.data
            header1 = image_hdu1.header
            original_bitpix1 = header1.get('BITPIX', 0)
            print(f"  -> HDU:{hdu_index1}, Shape:{data1.shape}, BITPIX:{original_bitpix1}")

            # Get initial debayer info for Input 1
            initial_debayer_code1, initial_bayer_name1 = None, None
            if forced_bayer_pattern:
                bayer_map = {'RGGB': cv2.COLOR_BayerRG2BGR, 'GRBG': cv2.COLOR_BayerGR2BGR,
                             'GBRG': cv2.COLOR_BayerGB2BGR, 'BGGR': cv2.COLOR_BayerBG2BGR}
                initial_debayer_code1 = bayer_map.get(forced_bayer_pattern)
                if initial_debayer_code1: initial_bayer_name1 = forced_bayer_pattern
                else: print(f"警告: --bayer-pattern '{forced_bayer_pattern}'は不明。デベイヤー無効化。", file=sys.stderr)
            else:
                initial_debayer_code1, initial_bayer_name1 = get_bayer_pattern_info(header1)
                if not initial_bayer_name1:
                    initial_debayer_code1 = cv2.COLOR_BayerBG2BGR; initial_bayer_name1 = 'BGGR'
                    print(f"  -> Bayer情報なし、デフォルト '{initial_bayer_name1}' を使用")
                else: print(f"  -> Bayerパターン '{initial_bayer_name1}' 検出")
            can_debayer_initially1 = initial_debayer_code1 is not None

        # --- Load Input File 2 ---
        print(f"入力ファイル2を読み込み中: {input_file2}")
        with fits.open(input_file2) as hdul2:
            image_hdu2, hdu_index2 = find_first_image_hdu(hdul2)
            if image_hdu2 is None: raise ValueError(f"{input_file2} 内に画像データが見つかりません。")
            data2 = image_hdu2.data
            header2 = image_hdu2.header # Store header2 as well
            original_bitpix2 = header2.get('BITPIX', 0)
            print(f"  -> HDU:{hdu_index2}, Shape:{data2.shape}, BITPIX:{original_bitpix2}")

            # Get initial debayer info for Input 2
            initial_debayer_code2, initial_bayer_name2 = None, None
            if forced_bayer_pattern:
                # Use the same forced pattern if provided
                bayer_map = {'RGGB': cv2.COLOR_BayerRG2BGR, 'GRBG': cv2.COLOR_BayerGR2BGR,
                             'GBRG': cv2.COLOR_BayerGB2BGR, 'BGGR': cv2.COLOR_BayerBG2BGR}
                initial_debayer_code2 = bayer_map.get(forced_bayer_pattern)
                if initial_debayer_code2: initial_bayer_name2 = forced_bayer_pattern
                # else: code remains None
            else:
                initial_debayer_code2, initial_bayer_name2 = get_bayer_pattern_info(header2)
                if not initial_bayer_name2:
                    # ★★★ Apply default BGGR to input 2 as well ★★★
                    print(f"  -> Bayer情報なし、デフォルト 'BGGR' を使用 (Input 2)")
                    initial_debayer_code2 = cv2.COLOR_BayerBG2BGR
                    initial_bayer_name2 = 'BGGR'
                else: print(f"  -> Bayerパターン '{initial_bayer_name2}' 検出 (Input 2)")
            can_debayer_initially2 = initial_debayer_code2 is not None

        # --- Dimension Check ---
        if data1.shape != data2.shape:
            raise ValueError(f"エラー: 入力ファイルの画像次元が一致しません ({data1.shape} vs {data2.shape})")
        print(f"画像次元: {data1.shape}")

        # --- Determine Bit Shift Capability (Based on Input 1) ---
        if np.issubdtype(data1.dtype, np.integer) and original_bitpix1 >= 16:
            can_shift_bits = True
            max_shift = original_bitpix1 - 8
            bit_shift = max_shift # Default to highest bits
            print(f"情報: 入力1は{original_bitpix1}bit整数。上下キーでビットシフト可能 (初期Shift={bit_shift}, Max={max_shift})。")
        else: bit_shift = 0 # Ensure shift is 0 if not applicable

        # --- Set Initial Bayer Index (Based on Input 1) ---
        if initial_bayer_name1:
            try: current_bayer_index = [name for name, code in bayer_patterns_cycle].index(initial_bayer_name1)
            except ValueError: current_bayer_index = 0 # Default to BGGR index
        else: current_bayer_index = 0
        print(f"初期 Bayer パターン (Input1基準): {bayer_patterns_cycle[current_bayer_index][0]}")

        # --- Perform Calculation ---
        result_data = perform_calculation(data1, data2, operation)
        # ★★★ Debug print for result_data ★★★
        if result_data is not None:
            print(f"DEBUG: result_data shape: {result_data.shape}, dtype: {result_data.dtype}")
        else:
            print("DEBUG: result_data is None after calculation!")

        # --- Create Windows & Layout (3x2 grid) ---
        win_w, win_h = 480, 360 # Default window size
        if _tkinter_available:
            try:
                root = tk.Tk(); screen_width = root.winfo_screenwidth(); screen_height = root.winfo_screenheight(); root.destroy()
                win_w = int(screen_width / 3.1); win_h = int(screen_height / 2.1)
                win_w = max(win_w, 320); win_h = max(win_h, 200)
            except Exception: pass

        try:
            # Create and position windows
            for i, name in enumerate(norm_window_names):
                cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(name, win_w, win_h); cv2.moveWindow(name, i * win_w, 0)
            if can_debayer_initially1:
                cv2.namedWindow(debayer_window_names[0], cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(debayer_window_names[0], win_w, win_h); cv2.moveWindow(debayer_window_names[0], 0 * win_w, win_h)
            if can_debayer_initially2:
                cv2.namedWindow(debayer_window_names[1], cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(debayer_window_names[1], win_w, win_h); cv2.moveWindow(debayer_window_names[1], 1 * win_w, win_h)
            if can_debayer_initially1: # Output debayer window only if Input 1 allows
                cv2.namedWindow(debayer_window_names[2], cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(debayer_window_names[2], win_w, win_h); cv2.moveWindow(debayer_window_names[2], 2 * win_w, win_h)
        except cv2.error as e: print(f"警告: ウィンドウ配置/リサイズエラー: {e}", file=sys.stderr)

        # --- Main Display Loop ---
        print("---------------------------------")
        print(f"結果を '{output_file}' に保存しますか？")
        controls = "  Y: 保存 | N/Esc: キャンセル | ←/→: Bayer切替"
        if can_shift_bits: controls += f" | ↑/↓: BitShift(Max={max_shift})"
        print(controls)
        print("---------------------------------")

        while True:
            if needs_update:
                print(f"表示更新中... Bayer: {bayer_patterns_cycle[current_bayer_index][0]}, BitShift: {bit_shift if can_shift_bits else 'N/A'}")
                current_bayer_name, current_debayer_code = bayer_patterns_cycle[current_bayer_index]

                # Normalize Input 1
                norm_data1, vmin1, vmax1 = normalize_image(data1, original_bitpix1, bit_shift=bit_shift if can_shift_bits else 0)
                # Normalize Input 2
                bit_shift_in2 = 0
                if can_shift_bits and np.issubdtype(data2.dtype, np.integer) and original_bitpix2 >= 16:
                     bit_shift_in2 = bit_shift
                norm_data2, vmin2, vmax2 = normalize_image(data2, original_bitpix2, bit_shift=bit_shift_in2)
                # Normalize Result
                norm_result, vmin3, vmax3 = normalize_image(result_data, original_bitpix1, bit_shift=bit_shift if can_shift_bits else 0)

                # Re-Debayer
                debayered1, debayered2, debayered_result = None, None, None
                if can_debayer_initially1: # Use initial capability flags
                    try: debayered1 = cv2.cvtColor(norm_data1, current_debayer_code)
                    except cv2.error: debayered1 = None # Ignore error
                if can_debayer_initially2:
                    try: debayered2 = cv2.cvtColor(norm_data2, current_debayer_code)
                    except cv2.error: debayered2 = None # Ignore error
                if can_debayer_initially1: # Output debayer depends on input 1 capability
                    try: debayered_result = cv2.cvtColor(norm_result, current_debayer_code)
                    except cv2.error: debayered_result = None # Ignore error

                # --- Update Window Titles ---
                title1 = f"Input 1 ({input_file1})"
                if can_shift_bits and bit_shift > 0: title1 += f" (Shift:{bit_shift} Bits:{7+bit_shift}-{bit_shift})"
                else: title1 += f" [Range:{vmin1:.3g}-{vmax1:.3g}]"
                if cv2.getWindowProperty(norm_window_names[0], cv2.WND_PROP_AUTOSIZE)!=-1: cv2.setWindowTitle(norm_window_names[0], title1)

                title2 = f"Input 2 ({input_file2})"
                if can_shift_bits and bit_shift_in2 > 0: title2 += f" (Shift:{bit_shift} Bits:{7+bit_shift}-{bit_shift})"
                else: title2 += f" [Range:{vmin2:.3g}-{vmax2:.3g}]"
                if cv2.getWindowProperty(norm_window_names[1], cv2.WND_PROP_AUTOSIZE)!=-1: cv2.setWindowTitle(norm_window_names[1], title2)

                # Output Title (no shift info)
                # title3 = f"Output Preview ({output_file}) [Range:{vmin3:.3g}-{vmax3:.3g}]"
                # ★★★ Link Output title to bit shift keys ★★★
                title3 = f"Output Preview ({output_file})"
                # Add shift info IF bit shift is active, even though it doesn't apply to float data itself
                if can_shift_bits and bit_shift > 0:
                     title3 += f" (Shift:{bit_shift} Bits:{7+bit_shift}-{bit_shift})" # Mimic input titles
                else: # Otherwise show the actual range used for normalization
                     title3 += f" [Range:{vmin3:.3g}-{vmax3:.3g}]"
                if cv2.getWindowProperty(norm_window_names[2], cv2.WND_PROP_AUTOSIZE)!=-1: cv2.setWindowTitle(norm_window_names[2], title3)

                title_suffix = f" | Debayer: {current_bayer_name}"
                if debayered1 is not None and cv2.getWindowProperty(debayer_window_names[0], cv2.WND_PROP_AUTOSIZE)!=-1: cv2.setWindowTitle(debayer_window_names[0], f"Input 1 Debayered{title_suffix}")
                if debayered2 is not None and cv2.getWindowProperty(debayer_window_names[1], cv2.WND_PROP_AUTOSIZE)!=-1: cv2.setWindowTitle(debayer_window_names[1], f"Input 2 Debayered{title_suffix}")
                if debayered_result is not None and cv2.getWindowProperty(debayer_window_names[2], cv2.WND_PROP_AUTOSIZE)!=-1: cv2.setWindowTitle(debayer_window_names[2], f"Output Debayered (In1 Pattern){title_suffix}")

                needs_update = False

            # --- Display Images ---
            if norm_data1 is not None: cv2.imshow(norm_window_names[0], norm_data1)
            if norm_data2 is not None: cv2.imshow(norm_window_names[1], norm_data2)
            if norm_result is not None: cv2.imshow(norm_window_names[2], norm_result)

            # Show debayered images, checking window existence first
            if can_debayer_initially1 and cv2.getWindowProperty(debayer_window_names[0], cv2.WND_PROP_AUTOSIZE)!=-1:
                if debayered1 is not None: cv2.imshow(debayer_window_names[0], debayered1)
                else: cv2.imshow(debayer_window_names[0], np.zeros((win_h, win_w, 3), dtype=np.uint8)) # Show black if failed

            if can_debayer_initially2 and cv2.getWindowProperty(debayer_window_names[1], cv2.WND_PROP_AUTOSIZE)!=-1:
                if debayered2 is not None: cv2.imshow(debayer_window_names[1], debayered2)
                else: cv2.imshow(debayer_window_names[1], np.zeros((win_h, win_w, 3), dtype=np.uint8))

            if can_debayer_initially1 and cv2.getWindowProperty(debayer_window_names[2], cv2.WND_PROP_AUTOSIZE)!=-1: # Output debayer depends on input 1
                if debayered_result is not None: cv2.imshow(debayer_window_names[2], debayered_result)
                else: cv2.imshow(debayer_window_names[2], np.zeros((win_h, win_w, 3), dtype=np.uint8))

            # --- Key Handling ---
            key = cv2.waitKeyEx(30) & 0xFFFFFF

            if key == KEY_Y:
                save_confirmed = True; print("'Y' 保存します。"); break
            elif key == KEY_N or key == KEY_ESC:
                save_confirmed = False; print("'N'/Esc 保存キャンセル。"); break
            elif key == KEY_LEFT_ARROW and can_debayer_initially1: # Bayer switch tied to input 1 capability
                current_bayer_index = (current_bayer_index - 1 + len(bayer_patterns_cycle)) % len(bayer_patterns_cycle)
                needs_update = True
            elif key == KEY_RIGHT_ARROW and can_debayer_initially1:
                current_bayer_index = (current_bayer_index + 1) % len(bayer_patterns_cycle)
                needs_update = True
            elif key == KEY_UP_ARROW and can_shift_bits:
                if bit_shift < max_shift: bit_shift += 1; needs_update = True
                # else: print("ビットシフト最大", file=sys.stderr)
            elif key == KEY_DOWN_ARROW and can_shift_bits:
                if bit_shift > 0: bit_shift -= 1; needs_update = True
                # else: print("ビットシフト最小", file=sys.stderr)

            # --- Window Closing Check ---
            active_windows = norm_window_names[:]
            if can_debayer_initially1: active_windows.extend([debayer_window_names[0], debayer_window_names[2]])
            if can_debayer_initially2: active_windows.append(debayer_window_names[1])
            any_window_open = False
            try:
                for name in active_windows:
                     if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) != -1 and cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
                         any_window_open = True; break
            except cv2.error: pass
            if not any_window_open: save_confirmed = False; print("ウィンドウが閉じられました。保存キャンセル。"); break

        # End Loop
        cv2.destroyAllWindows()

        # --- Save Result ---
        if save_confirmed:
            output_header = header1.copy(); output_header['BITPIX'] = -32 # Save as float32
            if 'BSCALE' in output_header: del output_header['BSCALE']
            if 'BZERO' in output_header: del output_header['BZERO']

            # ★★★ Explicitly set BAYERPAT in output header ★★★
            final_bayer_name = bayer_patterns_cycle[current_bayer_index][0]
            output_header['BAYERPAT'] = final_bayer_name
            # Remove potentially confusing offset keywords if they exist
            if 'XBAYROFF' in output_header: del output_header['XBAYROFF']
            if 'YBAYROFF' in output_header: del output_header['YBAYROFF']

            dt_now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            output_header.add_history(f"Calculated on {dt_now} by FITS_Calc.py")
            output_header.add_history(f"Operation: {input_file1} {operation} {input_file2}")
            output_header.add_history(f"Debayer preview pattern used (Input 1 basis): {final_bayer_name}")
            if can_shift_bits: output_header.add_history(f"Preview bit shift used (Input 1 basis): {bit_shift}")
            # ★★★ Debug print before saving ★★★
            if result_data is not None:
                print(f"DEBUG: Saving data - min={np.min(result_data)}, max={np.max(result_data)}")
            else:
                print("DEBUG: Cannot save, result_data is None!")
            # Create HDU and write file
            output_hdu = fits.PrimaryHDU(data=result_data.astype(np.float32), header=output_header)
            hdul = fits.HDUList([output_hdu]); print(f"{output_file} に結果を書き込み中...", end=''); hdul.writeto(output_file, overwrite=True); print(" 完了"); hdul.close()

            # ★★★ Verification Step ★★★
            try:
                with fits.open(output_file) as hdul_verify:
                    verify_data = hdul_verify[0].data
                    if verify_data is not None:
                        print(f"VERIFY: Read back data - shape={verify_data.shape}, dtype={verify_data.dtype}, min={np.min(verify_data)}, max={np.max(verify_data)}")
                    else:
                        print("VERIFY: Read back data is None!")
            except Exception as e:
                print(f"VERIFY: Failed to read back file: {e}")
            # ★★★ End Verification ★★★

        else: sys.exit(0)

    # Exception handling
    except FileNotFoundError as e: print(f"エラー: {e}", file=sys.stderr); sys.exit(1)
    except (ValueError, OSError) as e: print(f"エラー: {e}", file=sys.stderr); sys.exit(1)
    except ImportError as e:
         missing_lib = str(e).split("'")[-2]; install_cmd = "pip install numpy opencv-python astropy" + (" tkinter" if not _tkinter_available else "")
         print(f"エラー: 必要なライブラリ '{missing_lib}' が見つかりません。`{install_cmd}` を実行してください。", file=sys.stderr); sys.exit(1)
    except Exception as e: print(f"\n予期せぬエラーが発生しました: {e}", file=sys.stderr); import traceback; traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    main() 