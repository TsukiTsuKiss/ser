import struct
import sys
import argparse
import time # 時間計測用 (オプション)
import numpy as np
import cv2 # OpenCV をインポート
import tkinter as tk # 画面サイズ取得のためにインポート
import os # For listing directory
from astropy.io import fits # For reading FITS dark frames
from astropy.visualization import ZScaleInterval, PercentileInterval # For dark preview normalization

# --- Key Code Constants ---
KEY_TAB = 9
KEY_ENTER = 13
KEY_ESC = 27
KEY_B = ord('b') # For bitshift
KEY_D = ord('d') # For Dark frame clearing
KEY_E = ord('e') # For alignment
KEY_LEFT_BRACKET = ord('[')
KEY_RIGHT_BRACKET = ord(']')
KEY_W = ord('w') # For saving stack as FITS
KEY_S = ord('s') # For saving stack as SER
# KEY_V = ord('v') # Reverted: For saving stack as AVI
# --- New ASCII Key Bindings ---
KEY_J = ord('j') # Left (Was Right)
KEY_K = ord('k') # Right (New)
KEY_I = ord('i') # Up (New)
KEY_M = ord('m') # Down (New)
KEY_Q = ord('q') # New Quit Key
KEY_H = 104 # New key for 1000 step
KEY_L = 108 # New key for 1000 step

# --- Modifier Key Offset Constants (Platform Dependent!) ---
# These values are common on Windows, but may differ on other OS
# MOD_SHIFT = 2**16 # 65536 (REMOVED - Using specific codes based on user feedback)
# MOD_CTRL = 2**17  # 131072 (REMOVED)
# MOD_ALT = 2**18   # 262144 (REMOVED)

# --- Observed Key Codes (User Environment Specific!) ---
# Use these codes carefully, they might not work elsewhere.
KEY_SHIFT_J = 74
KEY_CTRL_J = 10
KEY_SHIFT_K = 75
KEY_CTRL_K = 11

# --- SERファイル読み込み関連 (SER_Frame_Viewer.pyから流用) ---
def read_ser_header(f):
    """SERファイルのヘッダーを読み込み、辞書として返す"""
    header_data = f.read(178)
    if len(header_data) < 178:
        raise ValueError("ヘッダーの読み込みに失敗しました。ファイルサイズが不足しています。")
    endian_prefix = '<'
    header_format = endian_prefix + (
        '14s '   # FileID
        'i '     # LuID
        'i '     # ColorID
        'i '     # LittleEndian
        'i '     # 幅 (Width)
        'i '     # 高さ (Height)
        'i '     # ビット数 (PixelDepthPerPlane)
        'i '     # フレーム数 (FrameCount)
        '40s '   # Observer
        '40s '   # Instrument
        '40s '   # Telescope
        'q '     # DateTime
        'q '     # DateTimeUTC
    )
    try:
        unpacked_header = struct.unpack(header_format, header_data)
    except struct.error as e:
        raise ValueError(f"ヘッダーの unpack に失敗しました: {e}")
    header = {
        "FileID": unpacked_header[0].decode('latin-1').strip('\x00'),
        "LuID": unpacked_header[1],
        "ColorID": unpacked_header[2],
        "LittleEndian": unpacked_header[3],
        "Width": unpacked_header[4],
        "Height": unpacked_header[5],
        "PixelDepthPerPlane": unpacked_header[6],
        "FrameCount": unpacked_header[7],
        "Observer": unpacked_header[8].decode('latin-1').strip('\x00'),
        "Instrument": unpacked_header[9].decode('latin-1').strip('\x00'),
        "Telescope": unpacked_header[10].decode('latin-1').strip('\x00'),
        "DateTime_Ticks": unpacked_header[11],
        "DateTimeUTC_Ticks": unpacked_header[12],
    }
    return header

def calculate_bytes_per_frame(header):
    """ヘッダー情報から1フレームあたりのバイト数を計算する"""
    width = header['Width']
    height = header['Height']
    pixel_depth = header['PixelDepthPerPlane']
    color_id = header['ColorID']
    num_planes = 1
    if color_id == 100 or color_id == 101: # RGB or BGR
        num_planes = 3
    bytes_per_plane_pixel = 1
    if pixel_depth > 8:
        bytes_per_plane_pixel = 2
    elif pixel_depth <= 0:
         raise ValueError(f"PixelDepthPerPlane が不正な値です: {pixel_depth}")
    if pixel_depth > 16:
         print(f"警告: PixelDepthPerPlane ({pixel_depth}) が16を超えています。計算は16bitとして扱います。", file=sys.stderr)
         bytes_per_plane_pixel = 2
    bytes_per_pixel_total = bytes_per_plane_pixel * num_planes
    frame_size = width * height * bytes_per_pixel_total
    return frame_size
# --- ここまで流用 ---

# --- FITS ファイル読み込みヘルパー (FITS_Calc.pyから流用) ---
def find_first_image_hdu(hdul):
    """最初の画像データを持つHDUを探す"""
    primary_hdu = hdul[0]
    if primary_hdu.is_image and primary_hdu.data is not None and primary_hdu.data.ndim >= 2:
        return primary_hdu, 0
    for i, hdu in enumerate(hdul[1:], 1):
        if hdu.is_image and hdu.data is not None and hdu.data.ndim >= 2:
            return hdu, i
    return None, -1 # 画像HDUが見つからない場合

class InteractiveViewer:
    def __init__(self, ser_file_path, start_frame=1, clip_low=0.0, clip_high=100.0, force_endian=None, ser_player_compat=False):
        self.ser_file_path = ser_file_path
        self.clip_low_percent = clip_low
        self.clip_high_percent = clip_high
        self.force_endian = force_endian
        self.ser_player_compat_mode = ser_player_compat # Store the compatibility mode flag
        self.header = None
        self.image_frame_size = 0
        self.pixels_per_frame = 0
        self.current_frame_index = 0 # 内部では0始まり
        self.total_frames = 0
        self.num_planes = 1
        self.dtype = np.uint8
        self.is_16bit = False
        self.bit_shift = 8
        self.dark_bit_shift = 0 # New: Dedicated bit shift for dark frame calculation
        self.color_id = 0
        self.all_frames_data = None
        self.window_name = f"SER Viewer: {ser_file_path}" # メインウィンドウ名
        self.debayer_window_name = f"Debayered: {ser_file_path}" # デベイヤーウィンドウ名
        self.dark_preview_window_name = f"Dark Preview / Select: {ser_file_path}" # ダークプレビュー/選択ウィンドウ名 (変更)
        self.raw_stack_window_name = "Stack (Raw)" # New window for raw stack
        self.dark_sub_stack_window_name = "Stack (Dark Subtracted)" # New window for dark subtracted stack
        self.debayer_stack_window_name = "Debayered Stack" # Window for debayered dark-sub stack

        # --- Log Messages ---
        self.log_messages = []
        self.max_log_lines = 15 # Max number of log lines to keep/display

        # --- Mode and State Variables ---
        self.current_mode = 'view' # 'view', 'dark_select', 'stack_config'
        # Dark Frame State
        self.dark_file_path = None
        self.dark_frame_data = None
        self.dark_file_list = []
        self.selected_dark_index = -1
        self.is_showing_dark_preview = False # Dark select mode state
        # Stack State
        self.stack_frame_count = 10 # Default number of frames to stack
        self.stack_operations = ['avg', 'sum', 'max', 'min']
        self.stack_operation_index = 0 # Default operation index
        self.raw_stacked_image_data = None # Holds the result of raw stack
        self.dark_subtracted_stacked_image_data = None # Holds the result of dark subtracted stack
        # Stack internal state for saving filenames correctly
        self.last_stack_start_index = -1
        self.last_stack_end_index = -1

        # Add attribute for stack display mode toggle: 'auto', 'fixed'
        # Options: 'auto', 'fixed', 'bitshift'
        self.stack_display_mode = 'bitshift' # Default to bitshift
        # Define the cycle order: bitshift -> auto -> fixed
        self.stack_display_modes = ['bitshift', 'auto', 'fixed']

        # Cache for stack previews to avoid re-normalization every frame
        self.cached_raw_stack_preview = None

        # --- Alignment State ---
        self.alignment_enabled = False
        self.align_reference_data = None # To store the reference frame data (copy)
        self.align_roi = None            # To store the selected ROI (x, y, w, h)
        self.align_roi_selecting = False # Flag for mouse ROI selection state
        self.align_roi_start_point = None # Start point for mouse drag
        self.current_mouse_roi_endpoint = None # For drawing temp ROI

        # --- Caching for Performance ---
        self.cached_dark_preview_img = None
        self.cached_dark_preview_index = -1 # Index of the cached dark preview
        self.cached_dark_sub_stack_preview = None
        self.cached_debayer_stack_preview = None
        self.needs_stack_redraw = True # Force initial draw

        self._initialize(start_frame)

    def _initialize(self, start_frame):
        print(f"ファイル {self.ser_file_path} を読み込んでいます...")
        start_time = time.time()
        try:
            with open(self.ser_file_path, 'rb') as f:
                self.header = read_ser_header(f)
                header_size = 178

                self.total_frames = self.header['FrameCount']
                if self.total_frames <= 0:
                    raise ValueError("ヘッダーから読み取ったフレーム数が0以下です。")

                # start_frame は 1 始まりで受け取るが、内部インデックスは 0 始まり
                if not (1 <= start_frame <= self.total_frames):
                    print(f"警告: 指定された開始フレーム番号 {start_frame} は無効です (1-{self.total_frames})。最初のフレーム (1) から開始します。", file=sys.stderr)
                    self.current_frame_index = 0
                else:
                    self.current_frame_index = start_frame - 1

                self.image_frame_size = calculate_bytes_per_frame(self.header)
                self.color_id = self.header['ColorID']

                if self.color_id == 100 or self.color_id == 101:
                    self.num_planes = 3

                if self.header['PixelDepthPerPlane'] > 8:
                    self.dtype = np.uint16
                    self.is_16bit = True
                elif self.header['PixelDepthPerPlane'] <= 0:
                     raise ValueError(f"ヘッダーの PixelDepthPerPlane が不正な値です: {self.header['PixelDepthPerPlane']}")

                self.pixels_per_frame = (self.header['Width'] *
                                         self.header['Height'] *
                                         self.num_planes)

                # --- 全画像データをメモリに読み込む --- 
                expected_data_bytes = self.total_frames * self.image_frame_size
                print(f"全 {self.total_frames} フレーム ({expected_data_bytes / (1024*1024):.2f} MB) をメモリに読み込み中...")
                f.seek(header_size)
                all_image_data_raw = f.read(expected_data_bytes)
                if len(all_image_data_raw) < expected_data_bytes:
                    actual_frames = len(all_image_data_raw) // self.image_frame_size
                    print(f"警告: ファイルサイズがヘッダー情報 ({self.total_frames} フレーム) と一致しません。実際に読み込めたのは {actual_frames} フレーム分のデータです。", file=sys.stderr)
                    if actual_frames <= 0:
                         raise IOError("画像データを読み込めませんでした。")
                    self.total_frames = actual_frames
                    # current_frame_index が範囲外になっていないか確認
                    self.current_frame_index = min(self.current_frame_index, self.total_frames - 1)
                    expected_data_bytes = self.total_frames * self.image_frame_size
                    all_image_data_raw = all_image_data_raw[:expected_data_bytes] # 読み込めた分だけ使う
                # ファイルはここで閉じる

            load_end_time = time.time()
            print(f"読み込み完了 ({load_end_time - start_time:.2f}秒)")

            # --- エンディアン決定とバイトスワップ (必要な場合) --- 
            needs_byteswap = False
            if self.is_16bit:
                is_system_little_endian = sys.byteorder == 'little'
                # ヘッダーの LittleEndian フィールド: 0=Big, 1=Little
                is_data_little_endian_in_header = (self.header['LittleEndian'] != 0)
                effective_data_endian = 'little' if is_data_little_endian_in_header else 'big'

                print("--- エンディアン情報 ---")
                print(f"  ヘッダー LittleEndian Flag: {self.header['LittleEndian']} ({effective_data_endian} endian)")
                print(f"  システムエンディアン: {sys.byteorder}")

                if self.force_endian:
                     print(f"  [情報] --force-endian オプションによりエンディアンを '{self.force_endian}' に強制します。")
                     effective_data_endian = self.force_endian
                     print(f"  エンディアン決定元: 強制({self.force_endian})")
                else:
                     print(f"  エンディアン決定元: ヘッダー")

                if (effective_data_endian == 'little' and not is_system_little_endian) or \
                   (effective_data_endian == 'big' and is_system_little_endian):
                    needs_byteswap = True
                print(f"  バイトスワップが必要か: {needs_byteswap}")
                print("--------------------	")

            # --- NumPy配列に変換 & バイトスワップ --- 
            self.all_frames_data = np.frombuffer(all_image_data_raw, dtype=self.dtype)
            if needs_byteswap:
                print("バイトスワップを実行中...")
                swap_start_time = time.time()
                # inplace=False (デフォルト) を使用して read-only エラーを回避
                self.all_frames_data = self.all_frames_data.byteswap()
                swap_end_time = time.time()
                print(f"バイトスワップ完了 ({swap_end_time - swap_start_time:.2f}秒)")

            expected_total_elements = self.total_frames * self.pixels_per_frame
            if self.all_frames_data.size != expected_total_elements:
                 # ファイルサイズ警告ですでに対処済みの場合もあるが、念のためチェック
                 raise ValueError(f"最終的なデータサイズ ({self.all_frames_data.size} 要素) が期待値 ({expected_total_elements} 要素) と一致しません。")

        except FileNotFoundError:
            print(f"エラー: ファイル '{self.ser_file_path}' が見つかりません。", file=sys.stderr)
            sys.exit(1)
        except (ValueError, IOError) as e:
            print(f"エラー: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"初期化中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
            sys.exit(1)

    def get_frame_data(self, frame_index):
        """メモリから指定フレームのデータを抽出し、reshapeする"""
        if not (0 <= frame_index < self.total_frames):
             raise IndexError(f"指定されたフレームインデックス {frame_index} が範囲外です (0-{self.total_frames-1})。")
        start_idx = frame_index * self.pixels_per_frame
        end_idx = start_idx + self.pixels_per_frame
        frame_data_1d = self.all_frames_data[start_idx:end_idx]

        if frame_data_1d.size != self.pixels_per_frame:
            # ここに来る場合はロジックエラーの可能性が高い
            raise RuntimeError(f"フレーム {frame_index + 1} のデータ抽出中のサイズ不一致 ({frame_data_1d.size} vs {self.pixels_per_frame})。")

        shape = (self.header['Height'], self.header['Width'])
        if self.num_planes == 3:
             shape += (self.num_planes,)

        try:
            image_array = frame_data_1d.reshape(shape)
        except ValueError as reshape_error:
             raise ValueError(f"フレーム {frame_index + 1} の画像データの形状変更 ({shape}) に失敗しました: {reshape_error}。データサイズとヘッダー情報が一致しない可能性があります。")
        return image_array

    def _apply_bit_shift_and_clip(self, image_array_in, is_16bit, bit_shift, clip_low_p, clip_high_p):
        """ビットシフトとパーセンタイルクリップを適用して uint8 画像を返すヘルパー関数"""
        # --- 16bit -> 8bit 変換 (クリッピング含む) ---
        if is_16bit:
            display_max_16bit = (255 << bit_shift)
            display_min_16bit = (0 << bit_shift)
            clipped_16bit = np.clip(image_array_in, display_min_16bit, display_max_16bit)
            image_array_8bit = np.right_shift(clipped_16bit, bit_shift).astype(np.uint8)
        else:
            # 元がuint8ならそのまま (入力はコピーされる保証はないので注意が必要だが、ここでは問題ないと仮定)
            image_array_8bit = image_array_in

        # --- オプションによるパーセンタイルクリップ (8bitデータに対して) ---
        if clip_low_p > 0.0 or clip_high_p < 100.0:
            min_val_8bit = np.min(image_array_8bit)
            max_val_8bit = np.max(image_array_8bit)
            # percentile計算のためにfloat32に変換
            # 多チャンネルの場合、全チャンネルまとめて計算される
            image_array_8bit_float = image_array_8bit.astype(np.float32)
            display_min = np.percentile(image_array_8bit_float, clip_low_p)
            display_max = np.percentile(image_array_8bit_float, clip_high_p)

            if display_max <= display_min:
                display_max = display_min + 1

            clipped_array_float = np.clip(image_array_8bit_float, display_min, display_max)
            scale_factor = display_max - display_min
            if scale_factor > 0:
                normalized_array_float = (clipped_array_float - display_min) / scale_factor
            else:
                normalized_array_float = np.zeros_like(clipped_array_float)
            final_image_8bit = (normalized_array_float * 255).astype(np.uint8)
        else:
            final_image_8bit = image_array_8bit

        return final_image_8bit

    def prepare_frames_for_display(self, frame_index):
        """指定フレームを取得し、(調整された)ダーク減算を行い、メイン表示用とデベイヤー表示用の uint8 画像を返す"""
        image_array = self.get_frame_data(frame_index)

        # --- ダークフレーム減算 (適用されている場合、専用シフト適用) ---
        processed_image_array = image_array.astype(np.float32) # Start with float version
        if self.dark_frame_data is not None:
            try:
                # Apply dedicated dark bit shift before subtraction
                # Avoid division by zero explicitly, though 2**0 is 1
                dark_divisor = 2.0 ** self.dark_bit_shift
                if dark_divisor <= 0: dark_divisor = 1.0 # Safety check

                adjusted_dark = self.dark_frame_data / dark_divisor

                # Subtract the adjusted dark frame
                processed_image_array = processed_image_array - adjusted_dark
                # print(f"DEBUG: Dark subtracted with shift {self.dark_bit_shift}") # Debug
            except ValueError as e:
                # 次元不一致などでエラーになる可能性は低いが念のため
                print(f"警告: ダークフレームの減算に失敗しました ({e})。ダークを無視します。", file=sys.stderr)
                self.dark_frame_data = None # 問題が起きたダークは一旦解除
                processed_image_array = image_array # 元データに戻す
            except Exception as e:
                 print(f"警告: ダークフレーム減算中に予期せぬエラー ({e})。ダークを無視します。", file=sys.stderr)
                 self.dark_frame_data = None
                 processed_image_array = image_array

        # --- メイン表示用フレームの準備 --- 
        # Now, processed_image_array is the float32 result AFTER dark subtraction
        # The _apply_bit_shift_and_clip expects uint8/16 input for *display* shift.
        # So, we need to convert the float result back to the original dtype range
        # before applying the *display* bit shift.

        # Clip the float result to the valid range of the original data type
        # Use 0 as min, max depends on original bit depth
        display_dtype_max = 65535 if self.is_16bit else 255
        clipped_float_result = np.clip(processed_image_array, 0, display_dtype_max)

        # Convert back to original dtype for the display shift function
        main_display_frame_source = clipped_float_result.astype(self.dtype)

        # Apply the *display* bit shift and percentile clip
        main_display_frame = self._apply_bit_shift_and_clip(
            main_display_frame_source, self.is_16bit, self.bit_shift, self.clip_low_percent, self.clip_high_percent
        )

        # メイン表示用の色空間調整 (OpenCVはBGR)
        if self.num_planes == 3:
            if self.color_id == 100: # SERがRGB
                main_display_frame = cv2.cvtColor(main_display_frame, cv2.COLOR_RGB2BGR)
            # SERがBGRならそのまま
        # モノクロはそのまま

        # --- デベイヤー表示用フレームの準備 --- 
        debayer_display_frame_bgr = None
        bayer_code = None
        # 出力色空間を BGR から RGB に戻す (これが正しかったはず)
        if self.color_id == 8: bayer_code = cv2.COLOR_BAYER_RG2RGB # RGGB -> RGB
        elif self.color_id == 9: bayer_code = cv2.COLOR_BAYER_GR2RGB # GRBG -> RGB
        elif self.color_id == 10: bayer_code = cv2.COLOR_BAYER_GB2RGB # GBRG -> RGB
        elif self.color_id == 11: bayer_code = cv2.COLOR_BAYER_BG2RGB # BGGR -> RGB

        if bayer_code is not None:
            try:
                # Debayering should happen on data as close to original as possible,
                # but AFTER dark subtraction. So we use the clipped float result converted back.
                debayered_rgb = cv2.cvtColor(main_display_frame_source, bayer_code)

                # Apply *display* bit shift and clip to the debayered result
                debayer_display_frame_rgb = self._apply_bit_shift_and_clip(
                    debayered_rgb, self.is_16bit, self.bit_shift,
                    self.clip_low_percent, self.clip_high_percent
                )
                debayer_display_frame_bgr = debayer_display_frame_rgb

            except cv2.error as e:
                print(f"\n警告: フレーム {frame_index + 1} のデベイヤー処理中にエラーが発生しました: {e}", file=sys.stderr)
                debayer_display_frame_bgr = None

        return main_display_frame, debayer_display_frame_bgr

    def _update_dark_file_list(self):
        """カレントディレクトリのFITSファイルをリストアップする"""
        try:
            print("Updating dark file list...")
            current_dir = "."
            # os.listdirの結果をソートして順序を安定させる
            all_files = sorted(os.listdir(current_dir))
            self.dark_file_list = [f for f in all_files if f.lower().endswith(('.fits', '.fts'))]
            print(f"Found {len(self.dark_file_list)} FITS files.")

            # インデックス調整
            if not self.dark_file_list:
                self.selected_dark_index = -1
            else:
                # 以前選択していたファイルがリストに残っているか確認
                current_selection_name = os.path.basename(self.dark_file_path) if self.dark_file_path else None
                if current_selection_name and current_selection_name in self.dark_file_list:
                    try:
                        self.selected_dark_index = self.dark_file_list.index(current_selection_name)
                    except ValueError:
                         # リストにはあるはずだが念のため
                         self.selected_dark_index = max(0, min(self.selected_dark_index, len(self.dark_file_list) - 1))
                else:
                    # 選択ファイルがない、またはリストから消えた場合は範囲内に収める
                    self.selected_dark_index = max(0, min(self.selected_dark_index, len(self.dark_file_list) - 1))
        except Exception as e:
            print(f"Error updating dark file list: {e}", file=sys.stderr)
            self.dark_file_list = []
            self.selected_dark_index = -1

    def _draw_dark_window(self, width, height):
        """ダーク選択ウィンドウの内容を描画する"""
        dark_img = np.zeros((height, width, 3), dtype=np.uint8)
        line_height = 20
        y_pos = line_height
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color_normal = (200, 200, 200)
        color_selected = (255, 255, 0) # Yellow for selected
        color_applied = (0, 255, 0) # Green for applied

        # 適用中のダークファイル表示
        applied_text = f"Applied: {os.path.basename(self.dark_file_path) if self.dark_file_path else 'None'}"
        cv2.putText(dark_img, applied_text, (10, y_pos), font, font_scale, color_applied, 1)
        y_pos += line_height * 2

        # ファイルリスト表示
        if not self.dark_file_list:
            cv2.putText(dark_img, "No FITS files found in current dir.", (10, y_pos), font, font_scale, color_normal, 1)
        else:
            # 表示範囲を調整 (多すぎる場合)
            max_items_display = height // line_height - 3 # 上下のマージン分減らす
            start_idx = 0
            if len(self.dark_file_list) > max_items_display:
                # 選択項目が中央あたりに来るように調整
                start_idx = max(0, self.selected_dark_index - max_items_display // 2)
                start_idx = min(start_idx, len(self.dark_file_list) - max_items_display)

            for i, filename in enumerate(self.dark_file_list[start_idx : start_idx + max_items_display]):
                actual_index = start_idx + i
                prefix = "> " if actual_index == self.selected_dark_index else "  "
                color = color_selected if actual_index == self.selected_dark_index else color_normal
                cv2.putText(dark_img, f"{prefix}{filename}", (10, y_pos), font, font_scale, color, 1)
                y_pos += line_height
            # スクロール可能であることを示唆 (もしあれば)
            if start_idx > 0:
                cv2.putText(dark_img, "... ^ ...", (10, y_pos - line_height * max_items_display -5), font, 0.4, color_normal, 1)
            if start_idx + max_items_display < len(self.dark_file_list):
                 cv2.putText(dark_img, "... v ...", (10, y_pos + 5), font, 0.4, color_normal, 1)

        if cv2.getWindowProperty(self.dark_preview_window_name, cv2.WND_PROP_AUTOSIZE) != -1:
            cv2.imshow(self.dark_preview_window_name, dark_img)

    def _normalize_dark_for_display(self, dark_data, bit_shift=0):
        """ダークフレームデータ(float32)をプレビュー用に正規化する (ビットシフト考慮)"""
        if dark_data is None or dark_data.size == 0:
            return None

        vmin, vmax = None, None

        # Determine vmin, vmax based on shift state OR ZScale/Percentile
        # Check if the *original* SER data was shiftable (implicitly via bit_shift > 0 check)
        if bit_shift > 0:
            # Apply equivalent range scaling based on bit shift
            vmin = 0.0
            vmax = float(255 << bit_shift)
            # print(f"DEBUG Dark Norm (Shift={bit_shift}): vmin={vmin}, vmax={vmax}") # Debug
        else:
            # Use ZScale/Percentile if bit_shift is 0
            try:
                interval = ZScaleInterval(contrast=0.25)
                vmin, vmax = interval.get_limits(dark_data)
                # print(f"DEBUG Dark Norm (ZScale): {vmin}, {vmax}") # Debug
            except Exception:
                try:
                    interval = PercentileInterval(99.5)
                    vmin, vmax = interval.get_limits(dark_data)
                    # print(f"DEBUG Dark Norm (Percentile): {vmin}, {vmax}") # Debug
                except Exception as e:
                    print(f"Warning: Could not normalize dark frame preview: {e}", file=sys.stderr)
                    return None

        if vmin is None or vmax is None or vmin >= vmax:
            # Fallback if calculation failed or resulted in invalid range
            min_val = np.min(dark_data); max_val = np.max(dark_data)
            if min_val == max_val: vmin = min_val - 0.5; vmax = max_val + 0.5
            else: vmin = min_val; vmax = max_val
            if vmin >= vmax: vmax = vmin + 1e-6
            # print(f"DEBUG Dark Norm (Fallback): vmin={vmin}, vmax={vmax}") # Debug

        # Normalize to 0-1 float using determined vmin, vmax
        # Handle potential NaN/Inf in dark_data if necessary (though unlikely for darks)
        safe_dark_data = np.nan_to_num(dark_data, nan=vmin, posinf=vmax, neginf=vmin)
        normalized_data = (safe_dark_data - vmin) / (vmax - vmin)
        # Clip to 0-1 and scale to 0-255 uint8
        scaled_data = (np.clip(normalized_data, 0, 1) * 255).astype(np.uint8)

        # Convert to 3-channel BGR for display
        if scaled_data.ndim == 2:
            return cv2.cvtColor(scaled_data, cv2.COLOR_GRAY2BGR)
        elif scaled_data.ndim == 3:
             # Assume it's already BGR or similar 3-channel
             return scaled_data
        else:
             print("Warning: Unexpected dark frame dimension for display.", file=sys.stderr)
             return None

    def _draw_dark_preview(self, width, height):
        """適用中のダークフレームのプレビューを描画する (ビットシフト連動)"""
        dark_img = np.zeros((height, width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (200, 200, 200)

        if self.dark_frame_data is None:
            cv2.putText(dark_img, "No Dark Applied", (10, height // 2), font, font_scale, color, 1)
        else:
            # Pass the current *dark* bit_shift value for normalization
            # Only apply if the original SER was 16bit, otherwise it doesn't make sense
            shift_to_pass = self.dark_bit_shift if self.is_16bit else 0
            normalized_dark = self._normalize_dark_for_display(self.dark_frame_data, shift_to_pass)
            if normalized_dark is not None:
                # Resize normalized dark to fit window (maintaining aspect ratio?)
                # For simplicity, just resize directly for now
                try:
                    preview_resized = cv2.resize(normalized_dark, (width, height), interpolation=cv2.INTER_NEAREST)
                    dark_img = preview_resized
                    # Add overlay text for filename?
                    applied_text = f"Dark: {os.path.basename(self.dark_file_path)}"
                    cv2.putText(dark_img, applied_text, (10, 20), font, 0.5, (0, 255, 0), 1)
                except cv2.error as resize_error:
                     print(f"Warning: Could not resize dark preview: {resize_error}", file=sys.stderr)
                     cv2.putText(dark_img, "Error displaying dark", (10, height // 2), font, font_scale, (0,0,255), 1)
            else:
                cv2.putText(dark_img, "Error normalizing dark", (10, height // 2), font, font_scale, (0,0,255), 1)

        if cv2.getWindowProperty(self.dark_preview_window_name, cv2.WND_PROP_AUTOSIZE) != -1:
            cv2.imshow(self.dark_preview_window_name, dark_img)

    def _draw_selected_dark_preview(self, width, height):
        """現在選択中のダークファイルのプレビューを画像バッファに描画して返す (キャッシュ利用)"""
        # If the selected index hasn't changed and we have a cached image, return it
        if self.selected_dark_index == self.cached_dark_preview_index and self.cached_dark_preview_img is not None:
            return self.cached_dark_preview_img

        # Otherwise, generate the preview
        preview_img = np.zeros((height, width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color_error = (0, 0, 255)
        color_info = (200, 200, 200)

        if not self.dark_file_list or not (0 <= self.selected_dark_index < len(self.dark_file_list)):
            cv2.putText(preview_img, "No file selected", (10, height // 2), font, font_scale, color_info, 1)
            # Update cache even if no file selected
            self.cached_dark_preview_img = preview_img
            self.cached_dark_preview_index = self.selected_dark_index
            return preview_img

        selected_filename = self.dark_file_list[self.selected_dark_index]
        filepath = os.path.join(".", selected_filename)

        try:
            # print(f"DEBUG: Reading preview for {selected_filename}") # Debug - Removed
            with fits.open(filepath) as hdul:
                image_hdu, _ = find_first_image_hdu(hdul)
                if image_hdu is None:
                    cv2.putText(preview_img, f"No image in: {selected_filename}", (10, height // 2), font, font_scale, color_error, 1)
                    # Update cache on error
                    self.cached_dark_preview_img = preview_img
                    self.cached_dark_preview_index = self.selected_dark_index
                    return preview_img

                dark_data_raw = image_hdu.data
                normalized_preview = self._normalize_dark_for_display(dark_data_raw.astype(np.float32))

                if normalized_preview is not None:
                    try:
                        preview_resized = cv2.resize(normalized_preview, (width, height), interpolation=cv2.INTER_NEAREST)
                        # Overlay filename
                        cv2.putText(preview_resized, f"Preview: {selected_filename}", (10, 20), font, 0.5, (255, 255, 0), 1)
                        # Cache the successful preview
                        self.cached_dark_preview_img = preview_resized
                        self.cached_dark_preview_index = self.selected_dark_index
                        return preview_resized # Return the new preview image
                    except cv2.error as resize_error:
                        print(f"Warning: Could not resize dark preview: {resize_error}", file=sys.stderr)
                        cv2.putText(preview_img, "Error resizing preview", (10, height // 2), font, font_scale, color_error, 1)
                        # Update cache on error
                        self.cached_dark_preview_img = preview_img
                        self.cached_dark_preview_index = self.selected_dark_index
                        return preview_img
                else:
                    cv2.putText(preview_img, f"Cannot normalize: {selected_filename}", (10, height // 2), font, font_scale, color_error, 1)
                    # Update cache on error
                    self.cached_dark_preview_img = preview_img
                    self.cached_dark_preview_index = self.selected_dark_index
                    return preview_img

        except FileNotFoundError:
            cv2.putText(preview_img, f"Not found: {selected_filename}", (10, height // 2), font, font_scale, color_error, 1)
            # Update cache on error
            self.cached_dark_preview_img = preview_img
            self.cached_dark_preview_index = self.selected_dark_index
            return preview_img
        except Exception as e:
            print(f"Error reading preview {selected_filename}: {e}", file=sys.stderr)
            cv2.putText(preview_img, f"Error reading: {selected_filename}", (10, height // 2), font, font_scale, color_error, 1)
            # Update cache on error
            self.cached_dark_preview_img = preview_img
            self.cached_dark_preview_index = self.selected_dark_index
            return preview_img

    # Modified: Accept display_mode argument to switch normalization method
    def _normalize_stack_for_display(self, stack_data, display_mode='auto'):
        """スタック結果データ(float32)をプレビュー用に正規化する (モード切替対応)"""
        if stack_data is None or stack_data.size == 0:
            return None

        vmin, vmax = None, None

        if display_mode == 'fixed':
            # Use fixed range [0, 65535]
            vmin = 0.0
            vmax = 65535.0
            # print(f"  DEBUG Norm Using FIXED range: vmin={vmin:.2f}, vmax={vmax:.2f}") # Debug - Removed
        else: # Default to 'auto' (ZScale/Percentile)
            try:
                interval = ZScaleInterval(contrast=0.25)
                vmin, vmax = interval.get_limits(stack_data)
                # print(f"DEBUG Stack Norm (ZScale): {vmin}, {vmax}") # Debug - Removed
            except Exception:
                try:
                    interval = PercentileInterval(99.5)
                    vmin, vmax = interval.get_limits(stack_data)
                    # print(f"DEBUG Stack Norm (Percentile): {vmin}, {vmax}") # Debug - Removed
                except Exception as e:
                    print(f"Warning: Could not auto-normalize stack preview: {e}", file=sys.stderr)
                    # Fallback needed if auto-norm fails
                    vmin, vmax = None, None # Force fallback calculation below

            # Fallback calculation if auto-norm failed or resulted in invalid range
            if vmin is None or vmax is None or vmin >= vmax:
                min_val = np.min(stack_data); max_val = np.max(stack_data)
                if min_val == max_val: vmin = min_val - 0.5; vmax = max_val + 0.5
                else: vmin = min_val; vmax = max_val
                if vmin >= vmax: vmax = vmin + 1e-6
                # print(f"  DEBUG Norm Using Fallback range: vmin={vmin:.2f}, vmax={vmax:.2f}") # Debug - Removed

        # Add detailed debug prints for normalization ranges (keep these for now)
        # print(f"  DEBUG Norm Input stack_data: min={np.min(stack_data):.2f}, max={np.max(stack_data):.2f}, mean={np.mean(stack_data):.2f}") # Debug - Removed
        # print(f"  DEBUG Norm Applied Method='{display_mode}': vmin={vmin:.2f}, vmax={vmax:.2f}") # Debug - Removed

        # --- Normalization logic based on mode ---
        if display_mode == 'bitshift':
            # Apply bit shift similar to _apply_bit_shift_and_clip
            # Note: stack_data is float32 here, but the logic should adapt
            # The _apply_bit_shift_and_clip expects uint or int type usually...
            # Let's try applying the shift logic carefully here.
            if self.is_16bit:
                # Shift float data first
                shifted_data = stack_data / (2.0 ** self.bit_shift)
                # Clip to 0-255 equivalent range (for 8-bit display)
                # We assume the *original* data was 16bit, so we clip after shift
                scaled_data = np.clip(shifted_data, 0, 255).astype(np.uint8)
            else:
                # For 8-bit original data, just ensure it's in display range
                scaled_data = np.clip(stack_data, 0, 255).astype(np.uint8)
            # print(f"  DEBUG Norm Using BITSHIFT (shift={self.bit_shift}): output range {np.min(scaled_data)}-{np.max(scaled_data)}")

        else: # Handle 'auto' and 'fixed' using vmin/vmax calculated earlier
            safe_stack_data = np.nan_to_num(stack_data, nan=vmin, posinf=vmax, neginf=vmin)
            if vmax - vmin == 0:
                # Avoid division by zero if vmax equals vmin
                # print("  DEBUG Norm Warning: vmax == vmin, setting normalized_data to 0.5") # Debug - Removed
                normalized_data = np.full_like(safe_stack_data, 0.5, dtype=np.float64)
            else:
                normalized_data = (safe_stack_data - vmin) / (vmax - vmin)

            # scaled_data calculation remains the same for auto/fixed
            scaled_data = (np.clip(normalized_data, 0, 1) * 255).astype(np.uint8)

        # print output range remains the same
        # print(f"  DEBUG Norm Output scaled_data: min={np.min(scaled_data)}, max={np.max(scaled_data)}") # Debug - Removed

        # BGR conversion restored
        if scaled_data.ndim == 2:
            return cv2.cvtColor(scaled_data, cv2.COLOR_GRAY2BGR)
        elif scaled_data.ndim == 3:
             # If it's already 3 channels, assume it's BGR (or was handled upstream)
             # This might happen if debayered data is passed in
             return scaled_data
        else:
             print("Warning: Unexpected stack result dimension for display.", file=sys.stderr)
             return None

    def _stack_frames(self):
        """現在の設定に基づいてフレームをスタックし、結果を self.raw_stacked_image_data と self.dark_subtracted_stacked_image_data に保存する"""
        operation = self.stack_operations[self.stack_operation_index]
        requested_frames = self.stack_frame_count

        # Determine actual frame range to stack
        end_index = self.current_frame_index
        start_index = max(0, end_index - requested_frames + 1)
        actual_frames = end_index - start_index + 1

        print(f"\n--- Starting Stack --- ")
        print(f"  Operation: {operation.upper()}")
        print(f"  Requested Frames: {requested_frames}")
        print(f"  Current Frame Index: {end_index}")
        print(f"  Stacking Frames: {actual_frames} (Indices {start_index} to {end_index}) ")
        if self.dark_frame_data is not None:
            print(f"  Applying Dark Frame: {os.path.basename(self.dark_file_path)} (Shift: {self.dark_bit_shift})")
        else:
            print("  No Dark Frame applied.")
        print("----------------------")

        if actual_frames <= 0:
            print("Error: Cannot stack 0 frames.", file=sys.stderr)
            self.raw_stacked_image_data = None
            self.dark_subtracted_stacked_image_data = None
            self.needs_stack_redraw = True # Need redraw to show N/A
            return False

        try:
            # Get the shape and dtype of the first frame to initialize accumulators
            first_frame_data = self.get_frame_data(start_index)
            frame_shape = first_frame_data.shape
            # Use float64 for accumulation
            raw_accumulator = np.zeros(frame_shape, dtype=np.float64)
            dark_sub_accumulator = np.zeros(frame_shape, dtype=np.float64)

            adjusted_dark = None
            has_dark = self.dark_frame_data is not None
            if has_dark:
                dark_divisor = 2.0 ** self.dark_bit_shift
                if dark_divisor <= 0: dark_divisor = 1.0
                adjusted_dark = self.dark_frame_data / dark_divisor # Still float32 is likely fine here

            # Process the first frame
            frame_data_raw_float = first_frame_data.astype(np.float32)
            frame_data_dark_sub_float = frame_data_raw_float.copy() # Start with copy
            if has_dark:
                frame_data_dark_sub_float -= adjusted_dark

            # --- Apply Alignment Shift (if enabled) ---
            dx, dy = self._calculate_alignment_shift(first_frame_data)
            if dx != 0.0 or dy != 0.0:
                 rows, cols = frame_shape[:2] # Get height, width
                 M = np.float32([[1, 0, dx], [0, 1, dy]])
                 frame_data_raw_float = cv2.warpAffine(frame_data_raw_float, M, (cols, rows))
                 frame_data_dark_sub_float = cv2.warpAffine(frame_data_dark_sub_float, M, (cols, rows))
                 # Log shift for the first frame
                 if start_index == self.current_frame_index: # Check if it's the reference frame itself being processed first
                     self._add_log(f"Align Frame {start_index+1}: Ref frame, no shift applied.")
                 else:
                     self._add_log(f"Align Frame {start_index+1}: Shift ({dx:.2f}, {dy:.2f})")
            # --- End Alignment --- #

            # Apply to accumulators based on operation
            if operation == 'avg' or operation == 'sum':
                raw_accumulator += frame_data_raw_float.astype(np.float64)
                dark_sub_accumulator += frame_data_dark_sub_float.astype(np.float64)
            elif operation == 'max':
                raw_accumulator = frame_data_raw_float.astype(np.float64)
                dark_sub_accumulator = frame_data_dark_sub_float.astype(np.float64)
            elif operation == 'min':
                raw_accumulator = frame_data_raw_float.astype(np.float64)
                dark_sub_accumulator = frame_data_dark_sub_float.astype(np.float64)

            # Loop through the rest of the frames
            for i in range(start_index + 1, end_index + 1):
                frame_data = self.get_frame_data(i)
                frame_data_raw_float = frame_data.astype(np.float32)
                frame_data_dark_sub_float = frame_data_raw_float.copy()
                if has_dark:
                    frame_data_dark_sub_float -= adjusted_dark

                # --- Apply Alignment Shift (if enabled) ---
                dx, dy = self._calculate_alignment_shift(frame_data)
                if dx != 0.0 or dy != 0.0:
                     # --- Re-add Bayer Alignment: Round shift to nearest even integer --- 
                     if self.color_id in [8, 9, 10, 11]: # Check if it's Bayer data
                         dx_apply = round(dx / 2) * 2
                         dy_apply = round(dy / 2) * 2
                         # Optional logging if rounding changes the value significantly
                         # if abs(dx - dx_apply) > 0.1 or abs(dy - dy_apply) > 0.1:
                         #    print(f"DEBUG Align Frame {i+1}: Rounded Bayer shift ({dx:.2f}, {dy:.2f}) -> ({dx_apply}, {dy_apply})")
                     else:
                         dx_apply = dx # Use original shift for non-Bayer
                         dy_apply = dy
                     # --- End Bayer Rounding --- 

                     rows, cols = frame_shape[:2] # Get height, width
                     M = np.float32([[1, 0, dx_apply], [0, 1, dy_apply]]) # Use potentially rounded shift
                     frame_data_raw_float = cv2.warpAffine(frame_data_raw_float, M, (cols, rows))
                     frame_data_dark_sub_float = cv2.warpAffine(frame_data_dark_sub_float, M, (cols, rows))
                     # Log the original calculated shift
                     self._add_log(f"Align Frame {i+1}: Shift ({dx:.2f}, {dy:.2f})")
                # --- End Alignment --- #

                # Apply to accumulators
                if operation == 'avg' or operation == 'sum':
                    raw_accumulator += frame_data_raw_float.astype(np.float64)
                    dark_sub_accumulator += frame_data_dark_sub_float.astype(np.float64)
                elif operation == 'max':
                    raw_accumulator = np.maximum(raw_accumulator, frame_data_raw_float.astype(np.float64))
                    dark_sub_accumulator = np.maximum(dark_sub_accumulator, frame_data_dark_sub_float.astype(np.float64))
                elif operation == 'min':
                    raw_accumulator = np.minimum(raw_accumulator, frame_data_raw_float.astype(np.float64))
                    dark_sub_accumulator = np.minimum(dark_sub_accumulator, frame_data_dark_sub_float.astype(np.float64))

                # Optional: Add progress indicator? e.g., print every 10 frames
                if (i - start_index + 1) % 10 == 0:
                    print(f"  Processed frame {i+1}/{self.total_frames} ({i - start_index + 1}/{actual_frames} stacked)...", end='\r')
            print("\n  Processing complete.") # Newline after progress

            # Final calculation for average
            if operation == 'avg' and actual_frames > 0:
                raw_accumulator /= actual_frames
                dark_sub_accumulator /= actual_frames

            # Add debug print for final accumulator values before storing
            # print(f"DEBUG Stack Final raw_accumulator: min={np.min(raw_accumulator):.2f}, max={np.max(raw_accumulator):.2f}, mean={np.mean(raw_accumulator):.2f}") # Debug - Removed
            # if has_dark: # Only print dark_sub if relevant
            #     print(f"DEBUG Stack Final dark_sub_accumulator: min={np.min(dark_sub_accumulator):.2f}, max={np.max(dark_sub_accumulator):.2f}, mean={np.mean(dark_sub_accumulator):.2f}") # Debug - Removed

            # Store results as float32
            self.raw_stacked_image_data = raw_accumulator.astype(np.float32)
            self.dark_subtracted_stacked_image_data = dark_sub_accumulator.astype(np.float32)
            # Store the actual indices used for saving later
            self.last_stack_start_index = start_index
            self.last_stack_end_index = end_index
            print(f"--- Stack Complete. Results stored (Raw/DarkSub). ---")
            self.needs_stack_redraw = True # Trigger redraw of stack windows
            return True

        except MemoryError:
            print("\nError: Insufficient memory to perform stack operation.", file=sys.stderr)
            self.raw_stacked_image_data = None
            self.dark_subtracted_stacked_image_data = None
            self.needs_stack_redraw = True # Need redraw to show N/A
            return False
        except Exception as e:
            print(f"\nError during stacking: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            self.raw_stacked_image_data = None
            self.dark_subtracted_stacked_image_data = None
            self.needs_stack_redraw = True # Need redraw to show N/A
            return False

    def _add_log(self, message):
        """Adds a message to the log buffer, keeping only the most recent lines."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        # Insert at the beginning for chronological order (newest first)
        self.log_messages.insert(0, log_entry)
        # Limit the number of messages
        if len(self.log_messages) > self.max_log_lines:
            self.log_messages = self.log_messages[:self.max_log_lines]

    def _draw_log_window(self, width, height):
        """Draws the log messages onto an image buffer."""
        log_img = np.zeros((height, width, 3), dtype=np.uint8)
        line_height = 18
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (200, 200, 200)

        # Draw lines from bottom up (newest at bottom)
        y_pos = height - line_height // 2 # Start near bottom
        for i in range(len(self.log_messages)):
            if y_pos < line_height: # Stop if we run out of space
                break
            cv2.putText(log_img, self.log_messages[i], (10, y_pos), font, font_scale, color, 1)
            y_pos -= line_height

        return log_img

    # Modified: Accept extension argument
    def _generate_output_filename(self, base_filename, start_frame_idx, end_frame_idx, operation, suffix, extension=".fts"):
        """Generates a filename for the output file using frame range."""
        name_part = base_filename
        if name_part.lower().endswith('.ser'):
            name_part = name_part[:-4]
        # Use 1-based indexing for filename
        output_filename = f"{name_part}_stack{start_frame_idx+1}-{end_frame_idx+1}f_{operation}_{suffix}{extension}" # Use extension arg
        return output_filename

    # Modified: Use start/end frame indices for filename
    def _generate_output_filename_png(self, base_filename, start_frame_idx, end_frame_idx, operation, suffix):
        """Generates a filename for the output PNG preview file using frame range."""
        name_part = base_filename
        if name_part.lower().endswith('.ser'):
            name_part = name_part[:-4]
        # Use 1-based indexing for filename, add _preview suffix
        output_filename = f"{name_part}_stack{start_frame_idx+1}-{end_frame_idx+1}f_{operation}_{suffix}_preview.png"
        return output_filename

    # Modified: Use last_stack indices for filenames
    def _save_stacked_images(self, raw_content, darks_content, debayer_content):
        """Saves the current stacked images (FITS) and their previews (PNG)."""
        fits_saved = False
        png_saved = False

        if self.raw_stacked_image_data is None and self.dark_subtracted_stacked_image_data is None:
            self._add_log("No stacked data available to save.")
            return

        base_name = os.path.basename(self.ser_file_path)
        stack_op = self.stack_operations[self.stack_operation_index]
        start_idx = self.last_stack_start_index
        end_idx = self.last_stack_end_index

        # --- Save FITS --- 
        # Save Raw Stacked Image (FITS) if available
        if self.raw_stacked_image_data is not None:
            raw_filename_fits = self._generate_output_filename(base_name, start_idx, end_idx, stack_op, "raw")
            self._add_log(f"Saving Raw stack FITS to: {raw_filename_fits}")
            try:
                hdu = fits.PrimaryHDU(self.raw_stacked_image_data.astype(np.float32)) # Ensure float32
                # Add some basic header info?
                hdu.header['HISTORY'] = f"Stacked from {base_name}"
                hdu.header['SB_COUNT'] = (self.stack_frame_count, 'Requested stack count')
                # TODO: Add actual stacked frames count?
                hdu.header['SB_OPER'] = (stack_op, 'Stacking operation')
                hdul = fits.HDUList([hdu])
                hdul.writeto(raw_filename_fits, overwrite=True)
                self._add_log(f"Successfully saved {raw_filename_fits}")
                fits_saved = True
            except Exception as e:
                self._add_log(f"ERROR saving {raw_filename_fits}: {e}")
                print(f"Error saving {raw_filename_fits}: {e}", file=sys.stderr)

        # Save Dark Subtracted Stacked Image (FITS) if available
        if self.dark_subtracted_stacked_image_data is not None:
            darks_filename_fits = self._generate_output_filename(base_name, start_idx, end_idx, stack_op, "darks")
            self._add_log(f"Saving Dark Sub stack FITS to: {darks_filename_fits}")
            try:
                hdu = fits.PrimaryHDU(self.dark_subtracted_stacked_image_data.astype(np.float32))
                hdu.header['HISTORY'] = f"Stacked from {base_name} (dark subtracted)"
                hdu.header['SB_COUNT'] = (self.stack_frame_count, 'Requested stack count')
                hdu.header['SB_OPER'] = (stack_op, 'Stacking operation')
                if self.dark_file_path:
                     hdu.header['SB_DARKF'] = (os.path.basename(self.dark_file_path), 'Dark frame used')
                     hdu.header['SB_DKSFT'] = (self.dark_bit_shift, 'Dark frame calculation shift')
                hdul = fits.HDUList([hdu])
                hdul.writeto(darks_filename_fits, overwrite=True)
                self._add_log(f"Successfully saved {darks_filename_fits}")
                fits_saved = True # Mark as saved even if only one FITS is saved
            except Exception as e:
                self._add_log(f"ERROR saving {darks_filename_fits}: {e}")
                print(f"Error saving {darks_filename_fits}: {e}", file=sys.stderr)

        # --- Save PNG Previews (using provided content) --- 
        if raw_content is None and darks_content is None and debayer_content is None:
            self._add_log("No stack previews generated to save as PNG.")
            # Don't return here if FITS might have saved
        else:
            # Save Raw Stack Preview (PNG) if available
            if raw_content is not None:
                raw_filename_png = self._generate_output_filename_png(base_name, start_idx, end_idx, stack_op, "raw")
                self._add_log(f"Saving Raw preview PNG to: {raw_filename_png}")
                try:
                    cv2.imwrite(raw_filename_png, raw_content)
                    self._add_log(f"Successfully saved {raw_filename_png}")
                    png_saved = True
                except Exception as e:
                    self._add_log(f"ERROR saving {raw_filename_png}: {e}")
                    print(f"Error saving {raw_filename_png}: {e}", file=sys.stderr)

            # Save Dark Subtracted Stack Preview (PNG) if available
            if darks_content is not None:
                darks_filename_png = self._generate_output_filename_png(base_name, start_idx, end_idx, stack_op, "darks")
                self._add_log(f"Saving Dark Sub preview PNG to: {darks_filename_png}")
                try:
                    cv2.imwrite(darks_filename_png, darks_content)
                    self._add_log(f"Successfully saved {darks_filename_png}")
                    png_saved = True
                except Exception as e:
                    self._add_log(f"ERROR saving {darks_filename_png}: {e}")
                    print(f"Error saving {darks_filename_png}: {e}", file=sys.stderr)

            # Save Debayered Stack Preview (PNG) if available
            if debayer_content is not None:
                debayer_filename_png = self._generate_output_filename_png(base_name, start_idx, end_idx, stack_op, "debayer")
                self._add_log(f"Saving Debayer preview PNG to: {debayer_filename_png}")
                try:
                    cv2.imwrite(debayer_filename_png, debayer_content)
                    self._add_log(f"Successfully saved {debayer_filename_png}")
                    png_saved = True
                except Exception as e:
                    self._add_log(f"ERROR saving {debayer_filename_png}: {e}")
                    print(f"Error saving {debayer_filename_png}: {e}", file=sys.stderr)

        if not fits_saved and not png_saved:
             self._add_log("Save failed: Neither FITS nor PNG data was saved.")
        elif not fits_saved:
             self._add_log("Save warning: PNG previews saved, but FITS data failed to save.")
        elif not png_saved:
             self._add_log("Save warning: FITS data saved, but PNG previews failed to save.")
        else:
             self._add_log("Save completed for FITS and PNG previews.")

    # --- Remove single frame save function --- #
    # def _save_stacked_to_ser(self, stack_data, suffix):
    #     ...
    # --- End Remove ---

    # --- Re-add function for running stack SER save --- #
    def _save_running_stack_to_ser(self, suffix):
        """Performs a running stack and saves the result as a multi-frame SER file."""
        stack_op = self.stack_operations[self.stack_operation_index]
        stack_size = self.stack_frame_count

        if stack_size <= 0:
            self._add_log("Error: Stack size must be greater than 0 for running stack.")
            return False
        if stack_size > self.total_frames:
            self._add_log(f"Warning: Stack size ({stack_size}) is larger than total frames ({self.total_frames}). Adjusting to {self.total_frames}.")
            stack_size = self.total_frames

        num_output_frames = self.total_frames - stack_size + 1
        if num_output_frames <= 0:
            self._add_log(f"Error: Not enough frames ({self.total_frames}) to create a running stack of size {stack_size}.")
            return False

        # Prepare output filename
        base_name = os.path.basename(self.ser_file_path)
        output_filename = self._generate_output_filename(base_name, 0, self.total_frames - 1, f"{stack_op}{stack_size}f", f"{suffix}_running", extension=".ser")

        self._add_log(f"Starting Running Stack ({suffix}, {stack_op.upper()}, {stack_size} frames) -> {output_filename} ({num_output_frames} frames output)")
        print(f"\n--- Starting Running Stack ({suffix}) --- ")
        print(f"  Operation: {stack_op.upper()}")
        print(f"  Stack Window Size: {stack_size} frames")
        print(f"  Input Frames: {self.total_frames}")
        print(f"  Output Frames: {num_output_frames}")
        print(f"  Output File: {output_filename}")
        if self.dark_frame_data is not None: print(f"  Applying Dark Frame: {os.path.basename(self.dark_file_path)}")
        if self.alignment_enabled: print(f"  Alignment: ENABLED (ROI: {'Set' if self.align_roi else 'Not Set'}, Ref Frame: {self.align_reference_data is not None})")
        else: print("  Alignment: DISABLED")
        print("----------------------------------")

        all_output_frames_data = []
        start_time_total = time.time()

        try:
            adjusted_dark = None
            has_dark = self.dark_frame_data is not None
            if has_dark:
                dark_divisor = 2.0 ** self.dark_bit_shift
                if dark_divisor <= 0: dark_divisor = 1.0
                adjusted_dark = self.dark_frame_data / dark_divisor

            for out_frame_idx in range(num_output_frames):
                start_frame_idx = out_frame_idx
                end_frame_idx = out_frame_idx + stack_size - 1
                actual_frames_in_window = stack_size

                first_frame_for_window = self.get_frame_data(start_frame_idx)
                frame_shape = first_frame_for_window.shape
                accumulator = np.zeros(frame_shape, dtype=np.float64)
                dark_sub_accumulator = np.zeros(frame_shape, dtype=np.float64) if suffix == "darks" else None

                for i in range(start_frame_idx, end_frame_idx + 1):
                    frame_data = self.get_frame_data(i)
                    frame_data_raw_float = frame_data.astype(np.float32)
                    frame_data_dark_sub_float = None
                    if suffix == "darks":
                         frame_data_dark_sub_float = frame_data_raw_float.copy()
                         if has_dark:
                             frame_data_dark_sub_float -= adjusted_dark

                    dx, dy = self._calculate_alignment_shift(frame_data)
                    dx_apply = round(dx / 2) * 2 if self.color_id in [8, 9, 10, 11] else dx
                    dy_apply = round(dy / 2) * 2 if self.color_id in [8, 9, 10, 11] else dy
                    if dx_apply != 0.0 or dy_apply != 0.0:
                         rows, cols = frame_shape[:2]
                         M = np.float32([[1, 0, dx_apply], [0, 1, dy_apply]])
                         frame_data_raw_float = cv2.warpAffine(frame_data_raw_float, M, (cols, rows))
                         if suffix == "darks" and frame_data_dark_sub_float is not None:
                              frame_data_dark_sub_float = cv2.warpAffine(frame_data_dark_sub_float, M, (cols, rows))

                    target_accumulator = dark_sub_accumulator if suffix == "darks" else accumulator
                    source_data_float = frame_data_dark_sub_float if suffix == "darks" else frame_data_raw_float
                    if source_data_float is None: continue

                    if i == start_frame_idx:
                        if stack_op == 'avg' or stack_op == 'sum': target_accumulator += source_data_float.astype(np.float64)
                        elif stack_op == 'max' or stack_op == 'min': target_accumulator = source_data_float.astype(np.float64)
                    else:
                        if stack_op == 'avg' or stack_op == 'sum': target_accumulator += source_data_float.astype(np.float64)
                        elif stack_op == 'max': target_accumulator = np.maximum(target_accumulator, source_data_float.astype(np.float64))
                        elif stack_op == 'min': target_accumulator = np.minimum(target_accumulator, source_data_float.astype(np.float64))

                target_accumulator_final = dark_sub_accumulator if suffix == "darks" else accumulator
                if stack_op == 'avg': target_accumulator_final /= actual_frames_in_window

                max_val = 65535 if self.is_16bit else 255
                clipped_data = np.clip(target_accumulator_final, 0, max_val)
                output_frame_native_dtype = clipped_data.astype(self.dtype)
                all_output_frames_data.append(output_frame_native_dtype)

                if (out_frame_idx + 1) % 10 == 0 or out_frame_idx == num_output_frames - 1:
                    elapsed_time = time.time() - start_time_total
                    print(f"  Processed output frame {out_frame_idx + 1}/{num_output_frames}... ({elapsed_time:.2f}s elapsed)", end='\r')
            print("\n  All frames processed.")

            # --- Prepare Header for Multi-Frame SER --- #
            final_header = self.header.copy()
            final_header['FrameCount'] = num_output_frames

            # --- Set LittleEndian flag based on compatibility mode --- #
            if self.ser_player_compat_mode:
                final_header['LittleEndian'] = 0 # SER Player Compat: Set flag to 0 (Big)
                self._add_log(f"SER Player Compat Mode: Writing SER with Big Endian flag (0).")
            else:
                final_header['LittleEndian'] = 1 # Default: Set flag to 1 (Little)
                self._add_log(f"Writing SER with Little Endian flag (1).")

            # --- Data Endianness (Always prepare as Little Endian) --- #
            is_system_little_endian = sys.byteorder == 'little'
            header_pack_endian_prefix = '<'

            # Pack Header
            header_format = header_pack_endian_prefix + ('14s i i i i i i i 40s 40s 40s q q')
            header_values = (
                final_header['FileID'].encode('latin-1').ljust(14, b'\x00'), final_header['LuID'],
                final_header['ColorID'], final_header['LittleEndian'], final_header['Width'],
                final_header['Height'], final_header['PixelDepthPerPlane'], final_header['FrameCount'],
                final_header['Observer'].encode('latin-1').ljust(40, b'\x00'),
                final_header['Instrument'].encode('latin-1').ljust(40, b'\x00'),
                final_header['Telescope'].encode('latin-1').ljust(40, b'\x00'),
                final_header['DateTime_Ticks'], final_header['DateTimeUTC_Ticks']
            )
            packed_header = struct.pack(header_format, *header_values)
            if len(packed_header) != 178: raise RuntimeError("Packed header size incorrect")

            # --- Combine all frame data and handle endianness --- #
            print("Combining frame data...")
            bytes_per_frame = final_header['Width'] * final_header['Height'] * (2 if self.is_16bit else 1) * (3 if self.num_planes==3 else 1)
            expected_total_elements = num_output_frames * self.pixels_per_frame
            try:
                combined_data_array_stacked = np.stack(all_output_frames_data, axis=0)
                if combined_data_array_stacked.size != expected_total_elements: raise ValueError(f"Combined data size mismatch: expected {expected_total_elements}, got {combined_data_array_stacked.size}")
                combined_data_array_flat = combined_data_array_stacked.flatten()
            except ValueError as e:
                self._add_log(f"Error concatenating/flattening frame data: {e}")
                print(f"Error concatenating/flattening frame data: {e}", file=sys.stderr)
                return False

            # --- Adjust data endianness for Little Endian output --- #
            needs_byteswap_for_ser = False
            # if self.is_16bit and (is_ser_file_little_endian != is_system_little_endian): # Removed condition based on original header
            if self.is_16bit and not is_system_little_endian: # Swap if system is Big Endian
                needs_byteswap_for_ser = True

            if needs_byteswap_for_ser:
                print("Performing byte swap for Little Endian SER data writing...") # Updated message
                swapped_data = combined_data_array_flat.byteswap() # inplace=False by default
                output_data_bytes = swapped_data.tobytes()
                # self._add_log(f"Performed byte swap for {suffix} SER data writing (original SER was {'Little' if is_ser_file_little_endian else 'Big'} Endian).") # Removed
                self._add_log(f"Performed byte swap for {suffix} SER data writing (System is Big Endian).") # Updated log
            else:
                output_data_bytes = combined_data_array_flat.tobytes()
                if self.is_16bit:
                   self._add_log(f"No byte swap needed for {suffix} SER data writing (System is Little Endian).")

            # --- Write to File --- #
            print(f"Writing to {output_filename}...")
            expected_data_bytes = num_output_frames * bytes_per_frame
            actual_data_bytes = len(output_data_bytes)
            if actual_data_bytes != expected_data_bytes: raise RuntimeError(f"CRITICAL ERROR: Data size mismatch before writing! Expected {expected_data_bytes}, got {actual_data_bytes}.")

            with open(output_filename, 'wb') as f:
                f.write(packed_header)
                f.write(output_data_bytes)

            total_time = time.time() - start_time_total
            self._add_log(f"Running Stack ({suffix}) saved to {output_filename} ({total_time:.2f}s)")
            print(f"--- Running Stack ({suffix}) Complete ({total_time:.2f}s) ---")
            return True

        except MemoryError:
            error_msg = f"ERROR: Insufficient memory during running stack ({suffix}) for {output_filename}"
            self._add_log(error_msg)
            print(f"\n{error_msg}", file=sys.stderr)
            return False
        except Exception as e:
            error_msg = f"ERROR during running stack ({suffix}) for {output_filename}: {e}"
            self._add_log(error_msg)
            print(f"\n{error_msg}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return False
    # --- End of Running Stack SER Save Function ---

    # --- Reverted Running Stack AVI Save Function (Placeholder/Removed) ---
    # def _save_running_stack_to_avi(self, fps):
    #     pass # This functionality was removed during revert

    def run(self):
        """インタラクティブ表示を開始する (OpenCVバージョン)"""
        print("--- 操作方法 ---")
        print("j : 前のフレーム") # Changed h to j
        print("k : 次のフレーム") # Changed j to k
        print("i / m : (16bit画像のみ) 表示ビット範囲を変更") # Changed u/n to i/m
        print("Tab : モード切替 (View -> Dark Select -> Stack Config)")
        print("[ / ] : (Dark適用中) ダークフレーム計算用ビットシフト変更")
        print("b : スタック表示の正規化方法切替 (Auto/Fixed)")
        print("s : スタック結果をFITS/PNGで保存")
        print("w : (単一フレーム)スタック結果をSERで保存") # Re-add w key description
        # print("v : 移動平均スタック結果をAVIで保存") # Reverted
        print("e : (Align Configモード) 位置合わせの有効/無効切替") # Align 機能キー追加
        print("Enter : (Align Configモード) 基準フレーム設定 / (Stack Config モード) スタック実行") # Align 機能キー追加, Stack実行追加
        print("Q / Esc : 終了")
        print("---------------")

        # --- 画面サイズ取得 --- 
        try:
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            print(f"画面サイズ: {screen_width}x{screen_height}")
        except Exception as e:
            print(f"警告: 画面サイズの取得に失敗しました ({e})。デフォルトサイズを使用します。", file=sys.stderr)
            screen_width = 1280 # デフォルト値
            screen_height = 720 # デフォルト値

        # --- 目標初期ウィンドウサイズ計算 (画面の50%) --- 
        target_w = screen_width * 0.5
        target_h = screen_height * 0.5

        # --- 元画像の縦横比を保ちつつ、目標サイズに合わせる --- 
        initial_width = 640 # デフォルト
        initial_height = 480 # デフォルト
        if self.header['Width'] > 0 and self.header['Height'] > 0:
            img_w = self.header['Width']
            img_h = self.header['Height']
            aspect_ratio = img_w / img_h

            potential_h = target_w / aspect_ratio
            potential_w = target_h * aspect_ratio

            if potential_h <= target_h:
                initial_width = int(target_w)
                initial_height = int(potential_h)
            else:
                initial_height = int(target_h)
                initial_width = int(potential_w)

            initial_width = max(initial_width, 320)
            initial_height = max(initial_height, 240)

            print(f"計算された初期ウィンドウサイズ: {initial_width}x{initial_height}")
        else:
             print("警告: ヘッダーから有効な画像サイズを取得できませんでした。デフォルトサイズを使用します。", file=sys.stderr)

        # OpenCVウィンドウ作成
        win_w = initial_width
        win_h = initial_height

        # Top Row Windows
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window_name, win_w, win_h)
        cv2.moveWindow(self.window_name, 0, 0) # Top-Left

        cv2.namedWindow(self.dark_preview_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.dark_preview_window_name, win_w, win_h)
        cv2.moveWindow(self.dark_preview_window_name, win_w + 5, 0) # Top-Center

        can_debayer_initial = self.color_id in [8, 9, 10, 11]
        if can_debayer_initial:
            cv2.namedWindow(self.debayer_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.debayer_window_name, win_w, win_h)
            cv2.moveWindow(self.debayer_window_name, win_w * 2 + 10, 0) # Top-Right

        # Bottom Row Windows (Raw Stack, Dark Sub Stack, Debayered Dark Sub Stack)
        cv2.namedWindow(self.raw_stack_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.raw_stack_window_name, win_w, win_h)
        cv2.moveWindow(self.raw_stack_window_name, 0, win_h + 40) # Bottom-Left

        cv2.namedWindow(self.dark_sub_stack_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.dark_sub_stack_window_name, win_w, win_h)
        cv2.moveWindow(self.dark_sub_stack_window_name, win_w + 5, win_h + 40) # Bottom-Center

        cv2.namedWindow(self.debayer_stack_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.debayer_stack_window_name, win_w, win_h)
        cv2.moveWindow(self.debayer_stack_window_name, win_w * 2 + 10, win_h + 40) # Bottom-Right

        # Register mouse callback for the main window (for ROI selection)
        # Pass window dimensions as param for coordinate scaling in callback
        callback_param = {'win_w': win_w, 'win_h': win_h}
        cv2.setMouseCallback(self.window_name, self._mouse_callback_align_roi, param=callback_param)

        self._add_log("Viewer started.") # Initial log message

        while True:
            # Check if stack results exist
            raw_stack_available = self.raw_stacked_image_data is not None
            dark_sub_stack_available = self.dark_subtracted_stacked_image_data is not None

            # --- Prepare current frame's display data ALWAYS ---
            main_frame, debayer_frame_bgr = None, None
            try:
                main_frame, debayer_frame_bgr = self.prepare_frames_for_display(self.current_frame_index)
            except IndexError as e:
                 print(f"\nエラー: {e} runループのインデックスがおかしい可能性があります。終了します。", file=sys.stderr)
                 break
            except (ValueError, RuntimeError, IOError) as e:
                print(f"\nエラー: フレーム {self.current_frame_index + 1} の表示準備に失敗しました: {e}。このフレームをスキップします。", file=sys.stderr)
                self.current_frame_index = (self.current_frame_index + 1) % self.total_frames
                time.sleep(0.5)
                continue
            except Exception as e:
                 print(f"\n表示準備中に予期せぬエラーが発生しました: {e}。終了します。", file=sys.stderr)
                 break

            # --- Determine Bayer code (needed for current frame and stack) ---
            bayer_code = None
            if self.color_id == 8: bayer_code = cv2.COLOR_BAYER_RG2RGB
            elif self.color_id == 9: bayer_code = cv2.COLOR_BAYER_GR2RGB
            elif self.color_id == 10: bayer_code = cv2.COLOR_BAYER_GB2RGB
            elif self.color_id == 11: bayer_code = cv2.COLOR_BAYER_BG2RGB
            can_debayer = bayer_code is not None

            # --- メインウィンドウ表示 (左上: Current Frame ALWAYS) ---
            if main_frame is not None:
                # --- Convert to BGR for display if it's grayscale (e.g., Bayer data) ---
                if main_frame.ndim == 2:
                    display_frame_to_process = cv2.cvtColor(main_frame, cv2.COLOR_GRAY2BGR)
                elif main_frame.ndim == 3:
                    display_frame_to_process = main_frame # Already BGR or RGB (handled later if needed)
                else:
                    print(f"Warning: Unexpected frame dimension ({main_frame.ndim}) for main display.", file=sys.stderr)
                    display_frame_to_process = main_frame # Try to proceed

                # --- Resize frame to window size FIRST ---
                # Get current actual window size (image area)
                win_rect = cv2.getWindowImageRect(self.window_name)
                current_win_w = win_rect[2]
                current_win_h = win_rect[3]
                # Fallback if rect is invalid (e.g., window minimized)
                if current_win_w <= 0 or current_win_h <= 0:
                    current_win_w = win_w # Use initial size as fallback
                    current_win_h = win_h

                try:
                    display_frame_resized = cv2.resize(display_frame_to_process, (current_win_w, current_win_h), interpolation=cv2.INTER_NEAREST)
                except cv2.error as resize_error:
                    print(f"Warning: Could not resize main frame for display: {resize_error}", file=sys.stderr)
                    # If resize fails, show the original frame without ROI drawing?
                    display_frame_resized = display_frame_to_process.copy() # Fallback to original
                    # Update current dimensions to match fallback
                    current_win_w = display_frame_resized.shape[1]
                    current_win_h = display_frame_resized.shape[0]


                # --- Draw ROI and Temp ROI on the RESIZED frame ---
                display_frame_resized_with_roi = display_frame_resized.copy()
                # Define colors
                color_roi_main = (255, 255, 255) # White
                color_roi_shadow = (0, 0, 0)     # Black
                color_temp_roi_main = (0, 255, 255) # Yellow
                color_temp_roi_shadow = color_roi_shadow # Use black shadow for temp too
                line_thickness = 1 # Changed from 2 to 1
                shadow_offset = line_thickness # Keep this for potential future use, though unused now

                img_w_orig = self.header['Width']
                img_h_orig = self.header['Height']

                # Calculate scaling factors (original to window) only if needed and valid
                scale_x = 1.0
                scale_y = 1.0
                if img_w_orig > 0 and img_h_orig > 0:
                     scale_x = current_win_w / img_w_orig
                     scale_y = current_win_h / img_h_orig

                # Draw permanent ROI (Convert original coords to window coords)
                if self.alignment_enabled and self.align_roi is not None:
                    # Always calculate from original coords stored in self.align_roi
                    orig_x, orig_y, orig_w, orig_h = self.align_roi
                    # Convert to window coordinates using current scaling factors
                    win_x = int(orig_x * scale_x)
                    win_y = int(orig_y * scale_y)
                    win_w_roi = int(orig_w * scale_x)
                    win_h_roi = int(orig_h * scale_y)

                    # Ensure width/height are at least 1 after scaling for drawing
                    win_w_roi = max(1, win_w_roi)
                    win_h_roi = max(1, win_h_roi)

                    # Draw main rectangle (using calculated window coordinates)
                    cv2.rectangle(display_frame_resized_with_roi,
                                  (win_x, win_y),
                                  (win_x + win_w_roi, win_y + win_h_roi),
                                  color_roi_main, line_thickness)

                # Draw temporary ROI during selection (using window coordinates directly)
                if self.current_mode == 'align_config' and self.align_roi_selecting and self.align_roi_start_point and self.current_mouse_roi_endpoint:
                    start_pt_win = self.align_roi_start_point # Already window coords
                    end_pt_win = self.current_mouse_roi_endpoint   # Already window coords
                    # Calculate corners for main (window coords)
                    x1_win, y1_win = start_pt_win
                    x2_win, y2_win = end_pt_win
                    # Ensure rect coords are ordered for drawing rect correctly
                    rect_x1 = min(x1_win, x2_win)
                    rect_y1 = min(y1_win, y2_win)
                    rect_x2 = max(x1_win, x2_win)
                    rect_y2 = max(y1_win, y2_win)

                    # --- REMOVED SHADOW DRAWING ---
                    # Draw main rectangle
                    cv2.rectangle(display_frame_resized_with_roi, (rect_x1, rect_y1), (rect_x2, rect_y2), color_temp_roi_main, line_thickness)
                # --- End ROI Drawing ---

                # Update title with current frame info, dark, stack, align config etc.
                mode_str = f"[{self.current_mode.replace('_', ' ').title()}]"
                title_main = f"{mode_str} Frame {self.current_frame_index + 1}/{self.total_frames}"
                if self.is_16bit:
                    upper_bit = 7 + self.bit_shift
                    lower_bit = 0 + self.bit_shift
                    title_main += f" | Bits: {upper_bit}-{lower_bit} (Shift: {self.bit_shift})"
                dark_status = self.dark_file_path if self.dark_file_path else "None"
                title_main += f" | Dark: {os.path.basename(dark_status) if dark_status != 'None' else 'None'}"
                if self.dark_frame_data is not None: title_main += f" (Shift: {self.dark_bit_shift})"
                # Calculate actual stack frames possible at current index
                actual_stack_frames_at_current = self.current_frame_index - max(0, self.current_frame_index - self.stack_frame_count + 1) + 1
                current_stack_op = self.stack_operations[self.stack_operation_index]
                # Update title to show actual/configured stack frames
                title_main += f" | Stack: {actual_stack_frames_at_current}f/{self.stack_frame_count}f {current_stack_op}"
                # Add Alignment Status to Title
                align_status = "ON" if self.alignment_enabled else "OFF"
                roi_status = "Set" if self.align_roi else "Not Set"
                title_main += f" | Align: {align_status} (ROI: {roi_status})"

                # Show the RESIZED frame with ROI drawing
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE) != -1:
                     cv2.setWindowTitle(self.window_name, title_main)
                     cv2.imshow(self.window_name, display_frame_resized_with_roi) # Show the modified resized frame
            # else: Frame prep failed, handled earlier

            # --- 中央ウィンドウ表示 (Top-Center: Log / Dark Select List / Dark Preview) ---
            # Cache dark preview only when needed
            if self.current_mode == 'dark_select':
                if self.is_showing_dark_preview:
                    # Call the potentially cached drawing function
                    preview_img = self._draw_selected_dark_preview(win_w, win_h)
                    if cv2.getWindowProperty(self.dark_preview_window_name, cv2.WND_PROP_AUTOSIZE) != -1:
                        cv2.setWindowTitle(self.dark_preview_window_name, "Dark Preview (Selected)")
                        cv2.imshow(self.dark_preview_window_name, preview_img)
                else:
                    # List drawing doesn't need heavy caching
                    self._draw_dark_window(win_w, win_h)
            else: # 'view' or 'stack_config' mode - Show Log Window
                 log_img = self._draw_log_window(win_w, win_h)
                 if cv2.getWindowProperty(self.dark_preview_window_name, cv2.WND_PROP_AUTOSIZE) != -1:
                      # Set title based on mode? Or keep generic?
                      cv2.setWindowTitle(self.dark_preview_window_name, f"Log Messages ({self.current_mode})")
                      cv2.imshow(self.dark_preview_window_name, log_img)

            # --- デベイヤーウィンドウ表示 (右上: Current Debayer ALWAYS) ---
            debayer_title = "Debayer"
            debayer_content = None
            if debayer_frame_bgr is not None:
                debayer_content = debayer_frame_bgr
                debayer_title = f"Debayered Frame {self.current_frame_index + 1}/{self.total_frames}"
            elif can_debayer:
                 debayer_title = f"Debayer Frame {self.current_frame_index + 1} (Error)"
            else:
                 debayer_title = "Debayering Not Applicable"

            if cv2.getWindowProperty(self.debayer_window_name, cv2.WND_PROP_AUTOSIZE) != -1:
                cv2.setWindowTitle(self.debayer_window_name, debayer_title)
                if debayer_content is not None:
                    cv2.imshow(self.debayer_window_name, debayer_content)
                else:
                    h, w = self.header['Height'], self.header['Width']
                    black_img = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.imshow(self.debayer_window_name, black_img)

            # --- スタック結果ウィンドウ表示 (キャッシュ利用) ---
            # Only regenerate stack previews if needed
            if self.needs_stack_redraw:
                print("Redrawing stack previews...") # Info print for redraw trigger
                self.cached_raw_stack_preview = None
                self.cached_dark_sub_stack_preview = None
                self.cached_debayer_stack_preview = None

                if raw_stack_available:
                    self.cached_raw_stack_preview = self._normalize_stack_for_display(self.raw_stacked_image_data, self.stack_display_mode)

                if dark_sub_stack_available:
                    self.cached_dark_sub_stack_preview = self._normalize_stack_for_display(self.dark_subtracted_stacked_image_data, self.stack_display_mode)
                    # Debayer the dark subtracted stack if possible
                    if can_debayer:
                        try:
                            dtype_max = 65535 if self.is_16bit else 255
                            clipped_stack = np.clip(self.dark_subtracted_stacked_image_data, 0, dtype_max)
                            stack_for_debayer = clipped_stack.astype(self.dtype)
                            debayered_stack_rgb = cv2.cvtColor(stack_for_debayer, bayer_code)
                            self.cached_debayer_stack_preview = self._normalize_stack_for_display(debayered_stack_rgb.astype(np.float32), self.stack_display_mode)
                        except Exception as e:
                            print(f"\nError processing stack for debayer display cache: {e}", file=sys.stderr)
                            self.cached_debayer_stack_preview = None # Ensure it's None on error

                self.needs_stack_redraw = False # Reset flag after redrawing

            # --- 左下: Raw Stack ---   <-- この行から
            raw_stack_title = "Stack (Raw) (N/A)"
            raw_stack_content_to_show = self.cached_raw_stack_preview
            if raw_stack_available:
                mode_indicator = f"[{self.stack_display_mode.capitalize()}]" # Add mode indicator
                if raw_stack_content_to_show is not None:
                    raw_stack_title = f"Stack (Raw) {mode_indicator}"
                else:
                    raw_stack_title = f"Stack (Raw) {mode_indicator} (Error Norm)"

            if cv2.getWindowProperty(self.raw_stack_window_name, cv2.WND_PROP_AUTOSIZE) != -1:
                cv2.setWindowTitle(self.raw_stack_window_name, raw_stack_title)
                if raw_stack_content_to_show is not None:
                    cv2.imshow(self.raw_stack_window_name, raw_stack_content_to_show)
                else:
                    h, w = self.header['Height'], self.header['Width']
                    black_img = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.imshow(self.raw_stack_window_name, black_img)

            # --- 右下: Debayered Stack ---
            debayer_stack_title = "Debayered Stack (N/A)"
            debayer_stack_content_to_show = self.cached_debayer_stack_preview
            # print(f"DEBUG Bottom Right: dark_sub_stack_available={dark_sub_stack_available}, can_debayer={can_debayer}") # Debug - Removed
            if dark_sub_stack_available:
                mode_indicator = f"[{self.stack_display_mode.capitalize()}]" # Add mode indicator
                if can_debayer:
                    if debayer_stack_content_to_show is not None:
                        # print(f"DEBUG Bottom Right: Showing cached debayer stack") # Debug - Removed
                        debayer_stack_title = f"Debayered Stack {mode_indicator}"
                    else:
                        # print(f"DEBUG Bottom Right: No cached debayer stack (or error occurred)") # Debug - Removed
                        debayer_stack_title = f"Debayered Stack {mode_indicator} (Error)"
                else:
                    # print("DEBUG Bottom Right: Stack available but cannot debayer") # Debug - Removed
                    debayer_stack_title = "Debayering Stack Not Applicable"
            # else: # Stack not available
                 # print("DEBUG Bottom Right: Stack not available") # Debug - Removed
                 # debayer_stack_title = "Debayered Stack (N/A)" # Already set

            if cv2.getWindowProperty(self.debayer_stack_window_name, cv2.WND_PROP_AUTOSIZE) != -1:
                cv2.setWindowTitle(self.debayer_stack_window_name, debayer_stack_title)
                if debayer_stack_content_to_show is not None:
                    cv2.imshow(self.debayer_stack_window_name, debayer_stack_content_to_show)
                else:
                    h, w = self.header['Height'], self.header['Width']
                    black_img = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.imshow(self.debayer_stack_window_name, black_img)

            # --- 中央下: Dark Subtracted Stack ---
            dark_sub_stack_title = "Stack (Dark Sub) (N/A)"
            dark_sub_stack_content_to_show = self.cached_dark_sub_stack_preview
            # print(f"DEBUG Bottom Center: dark_sub_stack_available = {dark_sub_stack_available}") # Debug - Removed
            if dark_sub_stack_available:
                mode_indicator = f"[{self.stack_display_mode.capitalize()}]" # Add mode indicator
                # print(f"DEBUG Bottom Center: Showing cached dark sub stack") # Debug - Removed
                if dark_sub_stack_content_to_show is not None:
                    dark_sub_stack_title = f"Stack (Dark Sub) {mode_indicator}"
                else:
                    # print("DEBUG Bottom Center: No cached dark sub stack (or error occurred)") # Debug - Removed
                    dark_sub_stack_title = f"Stack (Dark Sub) {mode_indicator} (Error Norm)"
            # else: # Stack not available
                 # print("DEBUG Bottom Center: No dark_sub_stack data available") # Debug - Removed
                 # dark_sub_stack_title remains "Stack (Dark Sub) (N/A)"

            if cv2.getWindowProperty(self.dark_sub_stack_window_name, cv2.WND_PROP_AUTOSIZE) != -1:
                cv2.setWindowTitle(self.dark_sub_stack_window_name, dark_sub_stack_title)
                if dark_sub_stack_content_to_show is not None:
                    cv2.imshow(self.dark_sub_stack_window_name, dark_sub_stack_content_to_show)
                else:
                    h, w = self.header['Height'], self.header['Width']
                    black_img = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.imshow(self.dark_sub_stack_window_name, black_img)

            # --- キー入力処理 ---      <-- この行は変更しない
            key = cv2.waitKeyEx(30)
            if key != -1: # 何かキーが押された場合のみ表示
                print(f"DEBUG: Key pressed, code = {key}")

            if key == KEY_ESC or key == KEY_Q: # Added KEY_Q
                self._add_log("Exit key pressed.")
                break
            elif key == KEY_TAB:
                prev_mode = self.current_mode
                # Changed Order: view -> dark_select -> stack_config -> align_config -> view
                if self.current_mode == 'view':
                    self.current_mode = 'dark_select'
                    self._update_dark_file_list()
                    # Clear alignment selection state when leaving view?
                    # self.align_roi_selecting = False
                elif self.current_mode == 'dark_select':
                    self.current_mode = 'stack_config'
                    self.is_showing_dark_preview = False # Ensure preview flag is off
                elif self.current_mode == 'stack_config':
                    self.current_mode = 'align_config' # New mode
                    # Reset mouse selection state when entering align mode
                    self.align_roi_selecting = False
                    self.align_roi_start_point = None
                elif self.current_mode == 'align_config':
                    self.current_mode = 'view'
                    # Reset mouse selection state when leaving align mode
                    self.align_roi_selecting = False

                if prev_mode != self.current_mode:
                     self._add_log(f"Mode switched to: {self.current_mode.replace('_', ' ').title()}")

            # --- Global Keys / Mode-Specific Key Handling ---
            elif key == KEY_B: # Toggle stack display normalization mode (Global)
                 # print("DEBUG: B key detected!") # Add console debug print - Removed
                 try:
                     current_index = self.stack_display_modes.index(self.stack_display_mode)
                     next_index = (current_index + 1) % len(self.stack_display_modes)
                     self.stack_display_mode = self.stack_display_modes[next_index]
                 except ValueError: # Should not happen if initialized correctly
                     self.stack_display_mode = self.stack_display_modes[0] # Fallback to first mode

                 self._add_log(f"Stack display normalization set to: {self.stack_display_mode}")
                 self.needs_stack_redraw = True # Trigger redraw with new mode
            elif key == KEY_W: # Save stacked images (FITS + PNG)
                 # print("DEBUG: S key detected!") # Console debug - Removed
                 self._add_log("Save key pressed, attempting to save stack results (FITS & PNG)...")
                 # Pass the *cached* preview images to the save function
                 self._save_stacked_images(self.cached_raw_stack_preview,
                                           self.cached_dark_sub_stack_preview,
                                           self.cached_debayer_stack_preview)
            elif key == KEY_S: # Modified: Call running stack SER save
                self._add_log("Running SER Save key pressed...")
                raw_running_saved = False
                darks_running_saved = False
                # Save Raw Running Stack
                raw_running_saved = self._save_running_stack_to_ser("raw")
                # Save Dark Subtracted Running Stack (if dark is applied)
                if self.dark_frame_data is not None:
                     darks_running_saved = self._save_running_stack_to_ser("darks")
                else:
                     self._add_log("No dark frame applied, skipping darks running stack save.")

                if not raw_running_saved and not darks_running_saved:
                    self._add_log("Running SER Save Failed: Neither raw nor darks could be saved.")

            # elif key == ord('v'): # Reverted: Save running stack as AVI
            #     self._add_log("Running AVI Save key pressed...")
            #     self._save_running_stack_to_avi(self.avi_fps)

            elif self.current_mode == 'view': # Mode specific keys start here
                # ... (view mode keys)
                # Ensure this block is correctly indented
                frame_changed = False
                bit_shift_changed = False
                dark_shift_changed = False
                frame_step = 0 # Initialize step to 0

                # Check specific key codes for modifiers + base key
                if key == KEY_J: # j (plain)
                    frame_step = -1
                elif key == KEY_SHIFT_J:  # Shift + j (Observed code)
                    frame_step = -10
                elif key == KEY_CTRL_J:  # Ctrl + j (Observed code)
                    frame_step = -100
                # Alt + j case cannot be handled based on provided data
                elif key == KEY_K: # k (plain)
                    frame_step = 1
                elif key == KEY_SHIFT_K:  # Shift + k (Observed code)
                    frame_step = 10
                elif key == KEY_CTRL_K:  # Ctrl + k (Observed code)
                    frame_step = 100
                # Alt + k case cannot be handled based on provided data
                elif key == KEY_H: # H -> -1000 frames
                    frame_step = -1000
                elif key == KEY_L: # L -> +1000 frames
                    frame_step = 1000

                # Apply frame change if step is non-zero
                if frame_step != 0:
                    self.current_frame_index = (self.current_frame_index + frame_step + self.total_frames) % self.total_frames
                    frame_changed = True

                    # --- Auto-recalculate stack --- (Moved inside frame_step check)
                    if self.raw_stacked_image_data is not None or self.dark_subtracted_stacked_image_data is not None:
                        self._add_log(f"Frame changed by {frame_step}. Recalculating stack for frame {self.current_frame_index + 1}...")
                        self._stack_frames()

                # --- Remaining View Mode Keys (These should be elif, not if) ---
                elif key == KEY_I: # Changed from KEY_U
                    if self.is_16bit:
                        if self.bit_shift < 8: self.bit_shift += 1; bit_shift_changed = True
                    if bit_shift_changed: self._add_log(f"Display shift set to: {self.bit_shift}")
                elif key == KEY_M: # Changed from KEY_N
                    if self.is_16bit:
                        if self.bit_shift > 0: self.bit_shift -= 1; bit_shift_changed = True
                    if bit_shift_changed: self._add_log(f"Display shift set to: {self.bit_shift}")
                elif key == KEY_LEFT_BRACKET: # '[' - Increase Dark Shift (JIS layout)
                    if self.dark_frame_data is not None:
                        # Define a reasonable upper limit, e.g., 8
                        max_dark_shift = 8
                        if self.dark_bit_shift < max_dark_shift:
                            self.dark_bit_shift += 1
                            dark_shift_changed = True # Set flag when changed
                            self.needs_stack_redraw = True # Recalculate stack needed
                            print(f"Dark calculation shift set to: {self.dark_bit_shift}")
                        else:
                            print(f"Dark calculation shift is already at maximum ({max_dark_shift})")
                        if dark_shift_changed: self._add_log(f"Dark calculation shift set to: {self.dark_bit_shift}")
                    else:
                        print("Apply a dark frame first to adjust its calculation shift.")
                elif key == KEY_RIGHT_BRACKET: # ']' - Decrease Dark Shift (JIS layout)
                    if self.dark_frame_data is not None:
                        if self.dark_bit_shift > 0:
                            self.dark_bit_shift -= 1
                            dark_shift_changed = True # Set flag when changed
                            self.needs_stack_redraw = True # Recalculate stack needed
                            print(f"Dark calculation shift set to: {self.dark_bit_shift}")
                        else:
                            print("Dark calculation shift is already at minimum (0)")
                        if dark_shift_changed: self._add_log(f"Dark calculation shift set to: {self.dark_bit_shift}")
                    else:
                        print("Apply a dark frame first to adjust its calculation shift.")
                # Recalculate stack if dark shift changed and stack exists
                # NOTE: Automatic recalculation on shift change in View mode is now handled by needs_stack_redraw flag
                #       The actual recalculation happens in _stack_frames when called next (e.g., in stack_config Enter)
                #       If you want immediate recalculation on shift change *in view mode*, you would need to call _stack_frames() here.
                #       However, this was identified as a potential performance bottleneck, so we rely on the flag.
                # Removed redundant pass here

            elif self.current_mode == 'stack_config':
                # ... (stack config keys)
                # Ensure this block is correctly indented
                config_changed = False
                stack_step = 0 # Initialize step to 0

                # Check specific key codes
                if key == KEY_J: # j (left) -> Increase count
                    stack_step = 1
                elif key == KEY_SHIFT_J:  # Shift + j -> Increase count by 10
                    stack_step = 10
                elif key == KEY_CTRL_J:  # Ctrl + j -> Increase count by 100
                    stack_step = 100
                # Alt + j case cannot be handled
                elif key == KEY_K: # k (right) -> Decrease count
                    stack_step = -1
                elif key == KEY_SHIFT_K:  # Shift + k -> Decrease count by 10
                    stack_step = -10
                elif key == KEY_CTRL_K:  # Ctrl + k -> Decrease count by 100
                    stack_step = -100
                # Alt + k case cannot be handled
                elif key == KEY_H: # H -> Increase count by 1000
                    stack_step = 1000
                elif key == KEY_L: # L -> Decrease count by 1000
                    stack_step = -1000

                # Apply stack count change
                if stack_step != 0:
                    current_count = self.stack_frame_count
                    # Use numpy.clip for simpler boundary check
                    new_count = np.clip(current_count + stack_step, 1, self.total_frames)
                    if new_count != current_count:
                        self.stack_frame_count = int(new_count) # Ensure it stays int
                        config_changed = True
                    else:
                        # Inform user if bounds were hit (optional)
                        if stack_step < 0 and current_count == 1:
                            print("Stack frame count cannot be less than 1.")
                        elif stack_step > 0 and current_count == self.total_frames:
                            print(f"Stack frame count cannot exceed total frames ({self.total_frames}).")

                # --- Remaining Stack Config Keys (These should be elif) ---
                elif key == KEY_I: # Up -> Previous stack operation
                    self.stack_operation_index = (self.stack_operation_index - 1) % len(self.stack_operations)
                    config_changed = True
                elif key == KEY_M: # Down -> Next stack operation
                    self.stack_operation_index = (self.stack_operation_index + 1) % len(self.stack_operations)
                    config_changed = True
                elif key == KEY_ENTER:
                    # Execute stack
                    self._add_log("Starting stack process...") # Log before call
                    success = self._stack_frames()
                    if success:
                        # Stack complete message is now in _stack_frames
                        self._add_log("Stacking finished successfully.")
                    else:
                        self._add_log("Stacking failed. Check console for errors.")

                if config_changed:
                    current_op = self.stack_operations[self.stack_operation_index]
                    # Remove print, use log
                    self._add_log(f"Stack: Frames={self.stack_frame_count}, Op='{current_op}'")

            elif self.current_mode == 'dark_select':
                # ... (dark select keys)
                # Ensure this block is correctly indented
                index_changed = False
                preview_toggled = False # Flag for logging preview toggle
                if key == KEY_I: # Changed from KEY_U
                    if self.dark_file_list and self.selected_dark_index != -1:
                        self.selected_dark_index -= 1
                        if self.selected_dark_index < 0:
                            self.selected_dark_index = len(self.dark_file_list) - 1
                        index_changed = True
                        self.cached_dark_preview_index = -1 # Invalidate cache on selection change
                elif key == KEY_M: # Changed from KEY_N
                    if self.dark_file_list and self.selected_dark_index != -1:
                        self.selected_dark_index += 1
                        if self.selected_dark_index >= len(self.dark_file_list):
                            self.selected_dark_index = 0
                        index_changed = True
                        self.cached_dark_preview_index = -1 # Invalidate cache on selection change
                elif key == KEY_K: # Toggle preview ON - Changed from J
                     if self.dark_file_list and 0 <= self.selected_dark_index < len(self.dark_file_list):
                         if not self.is_showing_dark_preview:
                            self.is_showing_dark_preview = True
                            preview_toggled = True # Flag it was toggled on
                            self.cached_dark_preview_index = -1 # Invalidate cache when turning preview on
                     else:
                          print("No file selected to preview.")
                elif key == KEY_J: # Toggle preview OFF - Changed from H
                     if self.is_showing_dark_preview:
                         self.is_showing_dark_preview = False
                         preview_toggled = True # Flag it was toggled off
                elif key == KEY_ENTER:
                    if self.dark_file_list and 0 <= self.selected_dark_index < len(self.dark_file_list):
                        selected_filename = self.dark_file_list[self.selected_dark_index]
                        filepath = os.path.join(".", selected_filename) # Assuming current dir
                        print(f"\nApplying dark frame: {selected_filename}...")
                        try:
                            with fits.open(filepath) as hdul:
                                image_hdu, hdu_index = find_first_image_hdu(hdul)
                                if image_hdu is None:
                                    print(f"Error: No image data found in FITS file: {selected_filename}", file=sys.stderr)
                                else:
                                    dark_data_raw = image_hdu.data
                                    print(f"  Dark HDU:{hdu_index}, Shape:{dark_data_raw.shape}, Dtype:{dark_data_raw.dtype}")
                                    # Dimension Check
                                    ser_h, ser_w = self.header['Height'], self.header['Width']
                                    if dark_data_raw.shape == (ser_h, ser_w):
                                        self.dark_frame_data = dark_data_raw.astype(np.float32) # Store as float for subtraction
                                        self.dark_file_path = filepath # Store full path or just filename?
                                        self.needs_stack_redraw = True # Need to redraw stack if dark changes
                                        print("  Dark frame applied successfully.")
                                        # Maybe switch back to view mode automatically?
                                        # self.current_mode = 'view'
                                    else:
                                        print(f"Error: Dimension mismatch! SER({ser_w}x{ser_h}) vs Dark({dark_data_raw.shape[1]}x{dark_data_raw.shape[0]}). Dark not applied.", file=sys.stderr)
                                        # Keep previous dark if any?
                        except FileNotFoundError:
                            print(f"Error: Dark file not found: {filepath}", file=sys.stderr)
                        except Exception as e:
                            print(f"Error reading or processing dark file {selected_filename}: {e}", file=sys.stderr)
                    else:
                        print("No dark file selected to apply.")
                elif key == KEY_D:
                    if self.dark_frame_data is not None:
                        print("\nClearing applied dark frame...")
                        self.dark_file_path = None
                        self.dark_frame_data = None
                        # インデックスは選択リスト上の現在の位置を維持する
                        # self.selected_dark_index = -1 # リセットしない方が良いかも
                        self.needs_stack_redraw = True # Need to redraw stack if dark changes
                        print("  Applied dark frame cleared.")
                        self.is_showing_dark_preview = False
                    else:
                        print("No dark frame is currently applied.")

                if index_changed:
                    if self.is_showing_dark_preview: # Update preview if it was already showing
                        # The drawing logic will handle the update in the next loop iteration
                        pass # No immediate action needed, redraw will happen
                    # Log the selection change?
                    if 0 <= self.selected_dark_index < len(self.dark_file_list):
                         sel_file = os.path.basename(self.dark_file_list[self.selected_dark_index])
                         self._add_log(f"Dark Select: Highlighted '{sel_file}'")

                if preview_toggled:
                    action = "ON" if self.is_showing_dark_preview else "OFF"
                    sel_file = "N/A"
                    if 0 <= self.selected_dark_index < len(self.dark_file_list):
                         sel_file = os.path.basename(self.dark_file_list[self.selected_dark_index])
                    self._add_log(f"Dark Select: Preview for '{sel_file}' toggled {action}")

            elif self.current_mode == 'align_config':
                # --- Frame Navigation Keys (Copied from View Mode) ---
                frame_changed = False
                frame_step = 0
                if key == KEY_J: frame_step = -1
                elif key == KEY_SHIFT_J: frame_step = -10
                elif key == KEY_CTRL_J: frame_step = -100
                elif key == KEY_K: frame_step = 1
                elif key == KEY_SHIFT_K: frame_step = 10
                elif key == KEY_CTRL_K: frame_step = 100
                elif key == KEY_H: frame_step = -1000
                elif key == KEY_L: frame_step = 1000

                if frame_step != 0:
                    self.current_frame_index = (self.current_frame_index + frame_step + self.total_frames) % self.total_frames
                    frame_changed = True
                    # DO NOT trigger auto-stack recalculation here
                    self._add_log(f"Align Config: Navigated to frame {self.current_frame_index + 1}")

                # --- Existing Align Config Keys (must be elif) ---
                elif key == KEY_E: # Toggle alignment enabled
                    self.alignment_enabled = not self.alignment_enabled
                    status_str = "ENABLED" if self.alignment_enabled else "DISABLED"
                    self._add_log(f"Alignment {status_str}")
                elif key == KEY_ENTER:
                    # Set current frame as reference
                    try:
                        self.align_reference_data = self.get_frame_data(self.current_frame_index).copy()
                        self._add_log(f"Alignment reference frame set to: {self.current_frame_index + 1}")
                        print(f"DEBUG Alignment reference set, shape: {self.align_reference_data.shape}, dtype: {self.align_reference_data.dtype}")
                        if self.align_roi is None:
                             self._add_log("Reference set. Please select ROI using mouse drag on main window.")
                    except Exception as e:
                        self._add_log(f"Error setting alignment reference: {e}")
                        print(f"Error setting alignment reference: {e}", file=sys.stderr)
                elif key == KEY_D:
                    # Disable alignment and clear reference/ROI
                    # Corrected condition to check for None explicitly
                    if self.alignment_enabled or self.align_reference_data is not None or self.align_roi is not None:
                        self.alignment_enabled = False
                        self.align_reference_data = None
                        self.align_roi = None
                        # --- REMOVED clearing window coords ---
                        self.align_roi_selecting = False # Just in case
                        self.align_roi_start_point = None
                        self.current_mouse_roi_endpoint = None
                        self._add_log("Alignment disabled, reference and ROI cleared.")
                    else:
                        self._add_log("Alignment is already disabled and no reference/ROI set.")

            # End of the main key handling if/elif chain

            # ... (Potential window closing checks could go here)

        # 終了処理 (This block should be OUTSIDE the while True loop)
        cv2.destroyAllWindows()
        print("\nビューアーを終了しました。")
        self._add_log("Viewer closing.") # Final log

    # --- Mouse Callback for ROI Selection ---
    def _mouse_callback_align_roi(self, event, x, y, flags, param):
        """Mouse callback function for selecting alignment ROI in Align Config mode."""
        if self.current_mode != 'align_config':
            return # Only active in align config mode

        # --- Get current window dimensions inside the callback ---
        try:
            rect = cv2.getWindowImageRect(self.window_name)
            # rect is (x, y, width, height) of the image area in the window
            win_w = rect[2]
            win_h = rect[3]
            if win_w <= 0 or win_h <= 0:
                print("Warning: Could not get valid window dimensions in mouse callback.", file=sys.stderr)
                # Try falling back to initial window size (less accurate but better than nothing)
                if param and 'win_w' in param and 'win_h' in param:
                     win_w = param['win_w']
                     win_h = param['win_h']
                     if win_w <= 0 or win_h <= 0: return # Fallback also invalid
                else:
                     return # Cannot proceed without window size
        except cv2.error:
             print("Warning: cv2.getWindowImageRect failed in mouse callback.", file=sys.stderr)
             # Fallback similar to above
             if param and 'win_w' in param and 'win_h' in param:
                  win_w = param['win_w']
                  win_h = param['win_h']
                  if win_w <= 0 or win_h <= 0: return
             else:
                  return

        # --- The rest of the callback logic remains largely the same ---
        if event == cv2.EVENT_LBUTTONDOWN:
            self.align_roi_selecting = True
            self.align_roi_start_point = (x, y) # Store window coords
            self.align_roi = None # Clear previous ROI when starting new selection
            self.current_mouse_roi_endpoint = (x, y) # Initialize endpoint for immediate feedback
            print(f"DEBUG ROI Select START at {self.align_roi_start_point} (Window Coords)")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.align_roi_selecting:
                self.current_mouse_roi_endpoint = (x, y) # Store current window coords

        elif event == cv2.EVENT_LBUTTONUP:
            if self.align_roi_selecting:
                self.align_roi_selecting = False
                end_point = (x, y) # Window coords
                # Check if start point exists (it should, but safety check)
                if self.align_roi_start_point is None:
                     print("Warning: LBUTTONUP received without start_point.", file=sys.stderr)
                     self.current_mouse_roi_endpoint = None
                     return

                start_x, start_y = self.align_roi_start_point # Window coords
                end_x, end_y = end_point # Window coords

                # Ensure coordinates are ordered correctly (window coords)
                roi_x_win = min(start_x, end_x)
                roi_y_win = min(start_y, end_y)
                roi_w_win = abs(start_x - end_x)
                roi_h_win = abs(start_y - end_y)

                # --- Convert window ROI coordinates to original image coordinates ---
                img_w_orig = self.header['Width']
                img_h_orig = self.header['Height']
                if img_w_orig <= 0 or img_h_orig <= 0:
                    print("Warning: Invalid original image dimensions from header.", file=sys.stderr)
                    self.align_roi = None
                    # Reset temporary points as well
                    self.align_roi_start_point = None
                    self.current_mouse_roi_endpoint = None
                    return

                # Use the window dimensions obtained at the start of the callback
                inv_scale_x = img_w_orig / win_w if win_w > 0 else 1
                inv_scale_y = img_h_orig / win_h if win_h > 0 else 1

                orig_roi_x = int(roi_x_win * inv_scale_x)
                orig_roi_y = int(roi_y_win * inv_scale_y)
                orig_roi_w = int(roi_w_win * inv_scale_x)
                orig_roi_h = int(roi_h_win * inv_scale_y)
                # --- End Conversion ---

                # Basic validation (on original scale)
                min_roi_size = 16
                if orig_roi_w >= min_roi_size and orig_roi_h >= min_roi_size:
                    # Clamp ROI to original image bounds
                    orig_roi_x = max(0, orig_roi_x)
                    orig_roi_y = max(0, orig_roi_y)
                    # Adjust width/height based on clamped origin
                    orig_roi_w = min(orig_roi_w, img_w_orig - orig_roi_x)
                    orig_roi_h = min(orig_roi_h, img_h_orig - orig_roi_y)

                    # Check again after clamping
                    if orig_roi_w >= min_roi_size and orig_roi_h >= min_roi_size:
                        # Store ROI coordinates relative to the ORIGINAL image
                        self.align_roi = (orig_roi_x, orig_roi_y, orig_roi_w, orig_roi_h)
                        # --- REMOVED storing final WINDOW coordinates ---
                        self._add_log(f"Alignment ROI set (orig coords): x={orig_roi_x}, y={orig_roi_y}, w={orig_roi_w}, h={orig_roi_h}")
                        print(f"DEBUG ROI Select END, ROI Set (Original): {self.align_roi}") # Removed Window coords from log
                    else:
                        self._add_log("ROI selection too small after clamping. Please try again.")
                        print(f"DEBUG ROI Select END, ROI too small after clamp (orig w={orig_roi_w}, orig h={orig_roi_h}).")
                        self.align_roi = None # Clear invalid ROI
                        # --- REMOVED clearing window coords ---
                else:
                    self._add_log(f"ROI selection too small (min {min_roi_size}x{min_roi_size}). Please try again.")
                    print(f"DEBUG ROI Select END, ROI too small (orig w={orig_roi_w}, orig h={orig_roi_h}).")
                    self.align_roi = None # Clear invalid ROI
                    # --- REMOVED clearing window coords ---

            # Reset temporary points regardless of success/failure after button up
            self.align_roi_start_point = None
            self.current_mouse_roi_endpoint = None

    # --- End of Mouse Callback ---

    # --- Alignment Calculation ---
    def _calculate_alignment_shift(self, current_frame_data):
        """Calculates the shift (dx, dy) between the reference ROI and current frame ROI using template matching."""
        if not self.alignment_enabled or self.align_reference_data is None or self.align_roi is None:
            return (0.0, 0.0) # Alignment not active or not configured

        try:
            # Extract ROI coordinates from original image space
            x, y, w, h = self.align_roi
            img_h_orig, img_w_orig = self.align_reference_data.shape[:2]

            # Ensure ROI dimensions are valid
            if w <= 0 or h <= 0:
                print("Warning: Invalid ROI dimensions for alignment.", file=sys.stderr)
                return (0.0, 0.0)

            # --- Prepare Adjusted Dark Frame (if applicable) ---
            adjusted_dark = None
            has_dark = self.dark_frame_data is not None
            if has_dark:
                dark_divisor = 2.0 ** self.dark_bit_shift
                if dark_divisor <= 0: dark_divisor = 1.0
                adjusted_dark = self.dark_frame_data / dark_divisor # float32 division

            # --- Define max value for clipping based on original dtype ---
            dtype_max = 65535 if self.is_16bit else 255

            # --- 1. Prepare Template ROI (from reference frame) ---
            # Extract the raw ROI data from reference
            y_end_ref = min(y + h, img_h_orig)
            x_end_ref = min(x + w, img_w_orig)
            ref_roi_raw = self.align_reference_data[y:y_end_ref, x:x_end_ref]

            if ref_roi_raw.shape != (h, w):
                print(f"Warning: Reference ROI extraction shape mismatch. Expected {(h, w)}, got {ref_roi_raw.shape}", file=sys.stderr)
                return (0.0, 0.0)

            # --- NEW: Apply Dark Subtraction to Template ---
            ref_roi_float = ref_roi_raw.astype(np.float32)
            if has_dark and adjusted_dark is not None:
                # Ensure dark ROI matches template ROI shape before subtracting
                dark_roi = adjusted_dark[y:y_end_ref, x:x_end_ref]
                if dark_roi.shape == ref_roi_float.shape:
                    ref_roi_float -= dark_roi
                else:
                    print(f"Warning: Dark ROI shape mismatch ({dark_roi.shape}) for template ({ref_roi_float.shape}). Skipping dark sub for template.", file=sys.stderr)

            # Clip back to original dtype range and convert dtype
            ref_roi_clipped = np.clip(ref_roi_float, 0, dtype_max)
            ref_roi_orig_dtype = ref_roi_clipped.astype(self.dtype)
            # --- END Dark Subtraction ---

            # Preprocess template: Bit shift (if 16bit) then grayscale (using the dark-subtracted data)
            template_8bit = self._apply_bit_shift_and_clip(ref_roi_orig_dtype, self.is_16bit, self.bit_shift, 0, 100) # Apply only display shift
            if template_8bit.ndim == 3:
                template_gray = cv2.cvtColor(template_8bit, cv2.COLOR_BGR2GRAY)
            elif template_8bit.ndim == 2:
                template_gray = template_8bit
            else:
                 print("Warning: Unexpected template dimension after bit shift.", file=sys.stderr)
                 return (0.0, 0.0)

            # --- 2. Prepare Search Area (from current frame) ---
            search_margin = 256
            search_x_start = max(0, x - search_margin)
            search_y_start = max(0, y - search_margin)
            search_x_end = min(img_w_orig, x + w + search_margin)
            search_y_end = min(img_h_orig, y + h + search_margin)

            # Extract the raw search area data from current frame
            search_area_raw = current_frame_data[search_y_start:search_y_end, search_x_start:search_x_end]

            # --- NEW: Apply Dark Subtraction to Search Area ---
            search_area_float = search_area_raw.astype(np.float32)
            if has_dark and adjusted_dark is not None:
                 # Ensure dark search area matches search area shape before subtracting
                 dark_search = adjusted_dark[search_y_start:search_y_end, search_x_start:search_x_end]
                 if dark_search.shape == search_area_float.shape:
                     search_area_float -= dark_search
                 else:
                     print(f"Warning: Dark ROI shape mismatch ({dark_search.shape}) for search area ({search_area_float.shape}). Skipping dark sub for search area.", file=sys.stderr)

            # Clip back to original dtype range and convert dtype
            search_area_clipped = np.clip(search_area_float, 0, dtype_max)
            search_area_orig_dtype = search_area_clipped.astype(self.dtype)
            # --- END Dark Subtraction ---

            # Preprocess search area: Bit shift (if 16bit) then grayscale (using the dark-subtracted data)
            search_area_8bit = self._apply_bit_shift_and_clip(search_area_orig_dtype, self.is_16bit, self.bit_shift, 0, 100) # Apply only display shift
            if search_area_8bit.ndim == 3:
                search_area_gray = cv2.cvtColor(search_area_8bit, cv2.COLOR_BGR2GRAY)
            elif search_area_8bit.ndim == 2:
                search_area_gray = search_area_8bit
            else:
                 print("Warning: Unexpected search area dimension after bit shift.", file=sys.stderr)
                 return (0.0, 0.0)

            # --- 3. Perform Template Matching ---
            # Ensure template is not larger than search area (can happen with edge ROIs)
            if template_gray.shape[0] > search_area_gray.shape[0] or template_gray.shape[1] > search_area_gray.shape[1]:
                print(f"Warning: Template ({template_gray.shape}) larger than search area ({search_area_gray.shape}). Skipping alignment.", file=sys.stderr)
                return (0.0, 0.0)

            # Use TM_CCOEFF_NORMED for robustness
            method = cv2.TM_CCOEFF_NORMED
            result = cv2.matchTemplate(search_area_gray, template_gray, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # For TM_CCOEFF_NORMED, the best match is at max_loc
            top_left = max_loc # This is the top-left coord of the match within the search area

            # --- 4. Calculate Shift ---
            # Convert top_left from search area coords to original image coords
            found_x_orig = search_x_start + top_left[0]
            found_y_orig = search_y_start + top_left[1]

            # Calculate the shift needed to move the current frame's feature
            # to the reference frame's feature position
            dx = float(x - found_x_orig) # Shift = Reference Pos - Found Pos
            dy = float(y - found_y_orig)

            # print(f"DEBUG MatchTemplate: Ref({x},{y}), Found({found_x_orig},{found_y_orig}), Shift({dx:.2f},{dy:.2f}), Score:{max_val:.4f}") # Optional debug

            return (dx, dy)

        except cv2.error as cv_err:
             print(f"Error during template matching: {cv_err}", file=sys.stderr)
             return (0.0, 0.0)
        except Exception as e:
            print(f"Error calculating alignment shift (dark sub included): {e}", file=sys.stderr) # Updated error message context
            import traceback
            traceback.print_exc()
            return (0.0, 0.0)

    # --- End Alignment Calculation ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SERファイルフレームをOpenCVでインタラクティブに表示します。スタック機能は未実装。')
    parser.add_argument('ser_file_path', help='SERファイルのパス')
    parser.add_argument('--start-frame', type=int, default=1, help='最初に表示するフレーム番号 (1始まり, デフォルト: 1)')
    parser.add_argument('--clip-low', type=float, default=0.0, help='表示範囲の下限パーセンタイル (ビットシフト後の8bitデータに対して適用, 例: 1.0)')
    parser.add_argument('--clip-high', type=float, default=100.0, help='表示範囲の上限パーセンタイル (ビットシフト後の8bitデータに対して適用, 例: 99.0)')
    parser.add_argument('--force-endian', choices=['little', 'big'], default=None, help='16bit画像データのエンディアンを強制します (little/big)')
    parser.add_argument('--ser-player-compat', action='store_true', help='SER Player互換モード: wキー保存時にヘッダーのエンディアンフラグを0(Big)にする(データはLittleのまま)')
    args = parser.parse_args()

    try:
        # クラス名を SERStacker に変更することを推奨
        viewer = InteractiveViewer(
            ser_file_path=args.ser_file_path,
            start_frame=args.start_frame,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            force_endian=args.force_endian,
            ser_player_compat=args.ser_player_compat # Pass the new argument
        )
        viewer.run()
    except ImportError as e:
         missing_lib = str(e).split("'")[-2]
         # エラーメッセージで必要なライブラリを明記
         install_cmd = "pip install numpy opencv-python"
         if missing_lib == 'cv2':
             print(f"エラー: 必要なライブラリ '{missing_lib}' (OpenCV) が見つかりません。", file=sys.stderr)
             print(f"実行前に {install_cmd} を試してください。", file=sys.stderr)
         else:
             print(f"エラー: 必要なライブラリ '{missing_lib}' が見つかりません。", file=sys.stderr)
             print(f"実行前に {install_cmd} を試してください。", file=sys.stderr)
         sys.exit(1)
    except FileNotFoundError:
         # _initialize 内で処理されるが、念のため
         print(f"エラー: ファイル '{args.ser_file_path}' が見つかりません。", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         print(f"\nスクリプト実行中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
         import traceback
         traceback.print_exc() # スタックトレースも表示
         sys.exit(1) 