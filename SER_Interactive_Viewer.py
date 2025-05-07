import struct
import sys
import argparse
import time # 時間計測用 (オプション)
import numpy as np
# Pillowは不要になったためコメントアウト (エラー処理のためimport自体は残す)
# from PIL import Image
import matplotlib.pyplot as plt

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

class InteractiveViewer:
    def __init__(self, ser_file_path, start_frame=1, clip_low=0.0, clip_high=100.0, force_endian=None):
        self.ser_file_path = ser_file_path
        self.clip_low_percent = clip_low
        self.clip_high_percent = clip_high
        self.force_endian = force_endian
        # self.f = None # ファイルハンドルは初期化時にのみ使用
        self.header = None
        self.image_frame_size = 0 # 1フレームのバイト数
        self.pixels_per_frame = 0 # 1フレームのピクセル(要素)数
        self.current_frame_index = 0
        self.total_frames = 0
        self.fig = None
        self.ax = None
        self.im = None
        self.num_planes = 1
        self.dtype = np.uint8
        self.is_16bit = False # 16bitデータかどうかのフラグ
        self.bit_shift = 8 # 16bit->8bit変換時のビットシフト量 (初期値: 上位8bit)
        # self.needs_byteswap = False # 初期化時に処理
        # self.endian_source = "ヘッダー" # 初期化時に処理
        self.color_id = 0
        self.all_frames_data = None # 全フレームデータを保持するNumPy配列
        self.display_min = 0.0
        self.display_max = 255.0

        self._initialize(start_frame)

    def _initialize(self, start_frame):
        print(f"ファイル {self.ser_file_path} を読み込んでいます...")
        start_time = time.time()
        try:
            with open(self.ser_file_path, 'rb') as f:
                self.header = read_ser_header(f)
                header_size = 178

                self.total_frames = self.header['FrameCount']
                if not (1 <= start_frame <= self.total_frames):
                    raise ValueError(f"指定された開始フレーム番号 {start_frame} は無効です。有効範囲: 1 から {self.total_frames}")
                self.current_frame_index = start_frame - 1

                self.image_frame_size = calculate_bytes_per_frame(self.header)
                self.color_id = self.header['ColorID']

                if self.color_id == 100 or self.color_id == 101:
                    self.num_planes = 3

                if self.header['PixelDepthPerPlane'] > 8:
                    self.dtype = np.uint16
                    self.is_16bit = True # 16bitフラグを立てる

                self.pixels_per_frame = (self.header['Width'] *
                                         self.header['Height'] *
                                         self.num_planes)

                # --- 全画像データをメモリに読み込む --- 
                print(f"全 {self.total_frames} フレーム ({self.total_frames * self.image_frame_size / (1024*1024):.2f} MB) をメモリに読み込み中...")
                f.seek(header_size)
                all_image_data_raw = f.read(self.total_frames * self.image_frame_size)
                if len(all_image_data_raw) < self.total_frames * self.image_frame_size:
                    raise IOError("ファイルの読み込みに失敗しました。ファイルサイズが期待値より小さいです。")
                # ファイルはここで閉じる

            load_end_time = time.time()
            print(f"読み込み完了 ({load_end_time - start_time:.2f}秒)")

            # --- エンディアン決定とバイトスワップ (必要な場合) --- 
            needs_byteswap = False
            endian_source = "ヘッダー"
            if self.dtype == np.uint16:
                is_system_little_endian = sys.byteorder == 'little'
                is_little_endian_image = self.header['LittleEndian'] != 0
                print("--- デバッグ情報 ---")
                print(f"  ヘッダー LittleEndian Flag: {self.header['LittleEndian']}")
                print(f"  画像データはリトルエンディアンか (ヘッダー値): {is_little_endian_image}")
                print(f"  システムエンディアン: {sys.byteorder}")

                if self.force_endian:
                    endian_source = f"強制({self.force_endian})"
                    print(f"  [情報] --force-endian オプションによりエンディアンを {self.force_endian} に強制します。")
                    force_is_little = (self.force_endian == 'little')
                    if force_is_little != is_system_little_endian:
                        needs_byteswap = True
                else:
                    endian_source = "ヘッダー"
                    if is_little_endian_image != is_system_little_endian:
                        needs_byteswap = True
                print(f"  エンディアン決定元: {endian_source}")
                print(f"  バイトスワップが必要か: {needs_byteswap}")
                print("--------------------	")

            # --- NumPy配列に変換 & バイトスワップ --- 
            self.all_frames_data = np.frombuffer(all_image_data_raw, dtype=self.dtype)
            if needs_byteswap:
                print("バイトスワップを実行中...")
                swap_start_time = time.time()
                self.all_frames_data = self.all_frames_data.byteswap()
                swap_end_time = time.time()
                print(f"バイトスワップ完了 ({swap_end_time - swap_start_time:.2f}秒)")

            expected_total_elements = self.total_frames * self.pixels_per_frame
            if self.all_frames_data.size != expected_total_elements:
                 raise ValueError(f"読み込んだ総データサイズ ({self.all_frames_data.size} 要素) が期待値 ({expected_total_elements} 要素) と一致しません。")

            # --- 初期フレーム準備と表示 --- 
            initial_image_data_display = self.prepare_frame_for_display(self.current_frame_index)

            # Matplotlib Figure設定
            self.fig, self.ax = plt.subplots()
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)

            # 画像表示
            self.im = self.ax.imshow(initial_image_data_display, cmap='gray' if self.num_planes == 1 else None)
            self.update_title()

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
        start_idx = frame_index * self.pixels_per_frame
        end_idx = start_idx + self.pixels_per_frame
        frame_data_1d = self.all_frames_data[start_idx:end_idx]

        if frame_data_1d.size != self.pixels_per_frame:
            raise ValueError(f"フレーム {frame_index + 1} のデータ抽出に失敗しました。サイズ不一致。")

        if self.num_planes == 1:
            image_array = frame_data_1d.reshape((self.header['Height'], self.header['Width']))
        else:
            try:
                 image_array = frame_data_1d.reshape((self.header['Height'], self.header['Width'], self.num_planes))
            except ValueError as reshape_error:
                 raise ValueError(f"フレーム {frame_index + 1} の画像データの形状変更に失敗しました: {reshape_error}. データサイズとヘッダー情報が一致しない可能性があります。")
        return image_array

    def prepare_frame_for_display(self, frame_index):
        """指定されたフレームデータを取得し、表示用に正規化する"""
        image_array = self.get_frame_data(frame_index)

        # --- 16bitの場合、ビットシフトしてuint8に変換 ---
        if self.is_16bit:
            # 選択したビット範囲に対応する16bitでの最大値・最小値を計算
            display_max_16bit = (255 << self.bit_shift)
            display_min_16bit = (0 << self.bit_shift) # 常に0だが明示的に

            # 元の16bitデータを、選択したビット範囲に対応する値域 [min, max] でクリップ
            # 例: shift=8 -> [0, 65280], shift=0 -> [0, 255]
            clipped_16bit = np.clip(image_array, display_min_16bit, display_max_16bit)

            # クリップした値を右シフトして 8bit に変換
            # これにより、選択範囲内の値が 0-255 にマッピングされ、範囲外は飽和(クリップ)する
            image_array_8bit = np.right_shift(clipped_16bit, self.bit_shift).astype(np.uint8)
        else:
            image_array_8bit = image_array # 元々uint8

        # 表示範囲計算 (8bit化されたデータに対して行う)
        # 注意: --clip-low/--clip-high オプションによるパーセンタイルクリップは
        #       ビットシフト後の8bitデータに対して適用される
        min_val_8bit = np.min(image_array_8bit)
        max_val_8bit = np.max(image_array_8bit)

        if self.clip_low_percent > 0.0 or self.clip_high_percent < 100.0:
            # np.percentile は float64 で計算されるため、先にfloatに変換しておく方が無難
            self.display_min = np.percentile(image_array_8bit.astype(np.float32), self.clip_low_percent)
            self.display_max = np.percentile(image_array_8bit.astype(np.float32), self.clip_high_percent)
            # print(f"\nフレーム {frame_index+1} 表示範囲 (8bit): {self.display_min:.1f}-{self.display_max:.1f}") # デバッグ用
        else:
            self.display_min = float(min_val_8bit)
            self.display_max = float(max_val_8bit)

        if self.display_max <= self.display_min:
            self.display_max = self.display_min + 1

        # 表示用に正規化 (uint8)
        # ここでは既に image_array_8bit (uint8) を対象とする
        clipped_array = np.clip(image_array_8bit.astype(np.float32), self.display_min, self.display_max)
        scale_factor = self.display_max - self.display_min
        if scale_factor > 0:
            normalized_array_float = (clipped_array - self.display_min) / scale_factor
        else:
            normalized_array_float = np.zeros_like(clipped_array, dtype=np.float32)

        # BGR -> RGB 変換 (正規化後)
        if self.num_planes == 3 and self.color_id == 101:
             normalized_array_float = normalized_array_float[..., ::-1]

        normalized_array_uint8 = (normalized_array_float * 255).astype(np.uint8)
        return normalized_array_uint8

    def update_display(self):
        """現在のフレーム番号に基づいて表示を更新する"""
        try:
            new_image_data = self.prepare_frame_for_display(self.current_frame_index)
            self.im.set_data(new_image_data)
            # クリップ範囲が変わる可能性があるのでclimも更新
            # self.im.set_clim(vmin=0, vmax=255) # uint8に正規化しているのでclimは不要
            self.update_title()
            self.fig.canvas.draw_idle()
        except (ValueError, IOError) as e:
            print(f"\nエラー: フレーム {self.current_frame_index + 1} の表示更新に失敗しました: {e}", file=sys.stderr)
        except Exception as e:
             print(f"\n予期せぬエラー: {e}", file=sys.stderr)

    def update_title(self):
        """ウィンドウタイトルを現在のフレーム番号とビットシフト量で更新"""
        title = f"SER Viewer: Frame {self.current_frame_index + 1}/{self.total_frames}"
        if self.is_16bit:
            # 表示している8bitが元の16bitデータのどの範囲かを表示
            # shift = 8 -> bits 15-8 (上位)
            # shift = 0 -> bits 7-0 (下位)
            upper_bit = 7 + self.bit_shift
            lower_bit = 0 + self.bit_shift
            title += f" | Viewing Bits: {upper_bit}-{lower_bit} (Shift: {self.bit_shift})"
        self.ax.set_title(title)
        self.fig.canvas.draw_idle() # タイトル更新を即時反映

    def on_key(self, event):
        """キーボード入力イベントの処理"""
        prev_index = self.current_frame_index
        needs_update = False # 表示更新が必要かどうかのフラグ

        # print(f"Key pressed: {event.key}") # デバッグ用

        # --- フレーム移動 --- 
        key_parts = event.key.lower().split('+')
        step = 1
        direction = 0 # 0: no move, 1: right, -1: left
        base_key = key_parts[-1]
        modifiers = key_parts[:-1]

        if 'shift' in modifiers:
            step = 10
        elif 'control' in modifiers or 'ctrl' in modifiers:
            step = 100
        elif 'alt' in modifiers:
            step = 1000

        if base_key == 'right':
            direction = 1
        elif base_key == 'left':
            direction = -1
        
        if direction != 0:
            new_index = (self.current_frame_index + direction * step) % self.total_frames
            if new_index != self.current_frame_index:
                self.current_frame_index = new_index
                needs_update = True

        # --- ビットシフト変更 (Up/Down) ---
        elif event.key == 'up': # 上矢印キー: 上位ビット方向へ (Shift増)
            if self.is_16bit:
                if self.bit_shift < 8:
                    self.bit_shift += 1
                    print(f"Bit shift increased to: {self.bit_shift} (Viewing higher bits)")
                    needs_update = True
            else:
                print("Bit shift change is only available for 16-bit images.")
                return
        elif event.key == 'down': # 下矢印キー: 下位ビット方向へ (Shift減)
            if self.is_16bit:
                if self.bit_shift > 0:
                    self.bit_shift -= 1
                    print(f"Bit shift decreased to: {self.bit_shift} (Viewing lower bits)")
                    needs_update = True
            else:
                print("Bit shift change is only available for 16-bit images.")
                return

        # --- 終了 --- 
        elif event.key == 'q' or event.key == 'escape':
            plt.close(self.fig)
            return
        
        # --- その他 --- 
        else:
            # print(f"Key '{event.key}' ignored.") # デバッグ用
            return # 未定義のキーは無視

        # --- 表示更新が必要な場合 --- 
        if needs_update:
            self.update_display()

    def run(self):
        """インタラクティブ表示を開始する"""
        if not self.fig:
             print("エラー: 初期化に失敗したため表示を開始できません。", file=sys.stderr)
             return
        print("画像ウィンドウで ← → キーを押してフレームを移動できます。Shift/Ctrl/Alt + ←/→ で高速移動。ウィンドウを閉じると終了します。") # 操作説明更新
        plt.show()
        # 表示ループ終了。ファイルは初期化時に閉じている
        print("\nビューアーを終了しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SERファイルフレームをインタラクティブに表示します（全フレームをメモリに読み込みます）。')
    parser.add_argument('ser_file_path', help='SERファイルのパス')
    parser.add_argument('--start-frame', type=int, default=1, help='最初に表示するフレーム番号 (デフォルト: 1)')
    parser.add_argument('--clip-low', type=float, default=0.0, help='表示範囲の下限パーセンタイル (例: 1.0)')
    parser.add_argument('--clip-high', type=float, default=100.0, help='表示範囲の上限パーセンタイル (例: 99.0)')
    parser.add_argument('--force-endian', choices=['little', 'big'], default=None, help='16bit画像データのエンディアンを強制します (little/big)')
    args = parser.parse_args()

    try:
        viewer = InteractiveViewer(
            ser_file_path=args.ser_file_path,
            start_frame=args.start_frame,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            force_endian=args.force_endian
        )
        viewer.run()
    except ImportError as e:
         missing_lib = str(e).split("'")[-2]
         install_cmd = "pip install numpy Pillow matplotlib"
         print(f"エラー: 必要なライブラリ '{missing_lib}' が見つかりません。{install_cmd} を実行してください。", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         print(f"スクリプト実行中にエラーが発生しました: {e}", file=sys.stderr)
         sys.exit(1) 