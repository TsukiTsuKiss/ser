import struct
import sys
import argparse
import numpy as np
from PIL import Image

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# SER_Viewer.py から必要な関数を移植
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

def main():
    parser = argparse.ArgumentParser(description='SERファイルから指定されたフレームの画像を表示し、情報とヒストグラム（オプション）を出力します。')
    parser.add_argument('ser_file_path', help='SERファイルのパス')
    parser.add_argument('frame_number', type=int, help='表示するフレーム番号 (1から始まるインデックス)')
    parser.add_argument('-hist', '--histogram', action='store_true', help='ピクセル値のヒストグラムを表示します (matplotlibが必要)')
    parser.add_argument('--clip-low', type=float, default=0.0, help='表示範囲の下限パーセンタイル (例: 1.0)')
    parser.add_argument('--clip-high', type=float, default=100.0, help='表示範囲の上限パーセンタイル (例: 99.0)')
    parser.add_argument('--force-endian', choices=['little', 'big'], default=None, help='画像データのエンディアンを強制します (little/big)')
    args = parser.parse_args()

    ser_file_path = args.ser_file_path
    frame_index_to_show = args.frame_number - 1
    show_histogram = args.histogram
    clip_low_percent = args.clip_low
    clip_high_percent = args.clip_high
    force_endian = args.force_endian

    if not (0.0 <= clip_low_percent < 100.0):
         raise ValueError("clip-low は 0.0 以上 100.0 未満である必要があります。")
    if not (0.0 < clip_high_percent <= 100.0):
         raise ValueError("clip-high は 0.0 より大きく 100.0 以下である必要があります。")
    if clip_low_percent >= clip_high_percent:
        raise ValueError("clip-low は clip-high より小さい必要があります。")

    if show_histogram and not MATPLOTLIB_AVAILABLE:
        print("エラー: ヒストグラム表示には matplotlib が必要です。pip install matplotlib を実行してください。", file=sys.stderr)
        sys.exit(1)

    try:
        with open(ser_file_path, 'rb') as f:
            header = read_ser_header(f)
            header_size = 178

            frame_count = header['FrameCount']
            if not (0 <= frame_index_to_show < frame_count):
                raise ValueError(f"指定されたフレーム番号 {args.frame_number} は無効です。有効範囲: 1 から {frame_count}")

            width = header['Width']
            height = header['Height']
            pixel_depth = header['PixelDepthPerPlane']
            color_id = header['ColorID']
            little_endian_flag = header['LittleEndian']
            is_little_endian_image = little_endian_flag != 0

            print("--- デバッグ情報 ---")
            print(f"  ヘッダー LittleEndian Flag: {little_endian_flag}")
            print(f"  画像データはリトルエンディアンか (ヘッダー値): {is_little_endian_image}")
            print(f"  システムエンディアン: {sys.byteorder}")
            print(f"  ColorID: {color_id}")

            image_frame_size = calculate_bytes_per_frame(header)

            image_data_offset = header_size + frame_index_to_show * image_frame_size

            f.seek(image_data_offset)
            image_data_raw = f.read(image_frame_size)
            if len(image_data_raw) < image_frame_size:
                raise IOError(f"フレーム {args.frame_number} の画像データの読み込みに失敗しました。ファイル末尾が予期せず終了した可能性があります。")

            num_planes = 1
            if color_id == 100 or color_id == 101:
                num_planes = 3

            dtype = np.uint8
            if pixel_depth > 8:
                dtype = np.uint16

            needs_byteswap = False
            endian_source = "ヘッダー"
            if dtype == np.uint16:
                is_system_little_endian = sys.byteorder == 'little'
                if force_endian:
                    # --- エンディアン強制ロジック --- 
                    endian_source = f"強制({force_endian})"
                    print(f"  [情報] --force-endian オプションによりエンディアンを {force_endian} に強制します。")
                    force_is_little = (force_endian == 'little')
                    if force_is_little != is_system_little_endian:
                        needs_byteswap = True
                else:
                    # --- 元のロジック (ヘッダーフラグに基づく) ---
                    if is_little_endian_image != is_system_little_endian:
                        needs_byteswap = True

            print(f"  エンディアン決定元: {endian_source}")
            print(f"  バイトスワップが必要か: {needs_byteswap}")
            print("--------------------	")

            image_array = np.frombuffer(image_data_raw, dtype=dtype)
            if needs_byteswap:
                image_array = image_array.byteswap()

            expected_elements = height * width * num_planes
            if image_array.size != expected_elements:
                 raise ValueError(f"読み込んだデータサイズ ({image_array.size} 要素) が期待値 ({expected_elements} 要素) と一致しません。")

            if num_planes == 1:
                image_array = image_array.reshape((height, width))
            else:
                try:
                     image_array = image_array.reshape((height, width, num_planes))
                except ValueError as reshape_error:
                     raise ValueError(f"画像データの形状変更に失敗しました: {reshape_error}. データサイズとヘッダー情報が一致しない可能性があります。")

            # --- ピクセル情報と表示範囲の計算 ---
            min_val = np.min(image_array)
            max_val = np.max(image_array)
            print(f"--- フレーム {args.frame_number} のピクセル情報 ---")
            print(f"  元データ最小値: {min_val}")
            print(f"  元データ最大値: {max_val}")

            # 表示範囲をパーセンタイルで決定
            if clip_low_percent > 0.0 or clip_high_percent < 100.0:
                display_min = np.percentile(image_array, clip_low_percent)
                display_max = np.percentile(image_array, clip_high_percent)
                print(f"  表示範囲 (クリップ後): {display_min:.2f} ({clip_low_percent}%) - {display_max:.2f} ({clip_high_percent}%)")
            else:
                display_min = float(min_val)
                display_max = float(max_val)
                print(f"  表示範囲: {display_min:.0f} - {display_max:.0f} (クリップなし)")

            # 0除算防止
            if display_max <= display_min:
                print(f"警告: 計算された表示範囲の上限({display_max:.2f})が下限({display_min:.2f})以下です。表示範囲を調整します。", file=sys.stderr)
                display_max = display_min + 1

            # --- 表示用配列の正規化 (uint8) ---
            # 元の配列を float に変換して計算
            clipped_array = np.clip(image_array.astype(np.float32), display_min, display_max)
            # 0-1 にスケーリング
            scale_factor = display_max - display_min
            if scale_factor > 0:
                normalized_array_float = (clipped_array - display_min) / scale_factor
            else:
                normalized_array_float = np.zeros_like(clipped_array, dtype=np.float32)
            # 0-255 にスケーリングして uint8 に変換
            normalized_array_uint8 = (normalized_array_float * 255).astype(np.uint8)

            # --- Pillow Image 作成 (正規化後データを使用) ---
            mode = 'L' # 8bit モノクロ
            display_array_final = normalized_array_uint8
            if num_planes == 3:
                mode = 'RGB' # 8bit カラー
                if color_id == 101: # BGRの場合、チャンネル入れ替え
                     # 元の image_array からではなく、正規化後の uint8 配列を使う
                     # しかし、image_array が uint16 の場合、display_array_final はすでに shape(h,w)になっている可能性がある
                     # 輝度計算後のため。この場合はRGBモードは使えない。
                     # 再度 reshape が必要か、あるいは輝度計算前の正規化が必要。
                     # ここでは、num_planes=3 の場合は常に RGB で正規化すると仮定して修正。
                     # uint8 に変換する前にチャンネル入れ替えを行うべき。

                     # 修正：uint8変換前にチャンネル操作
                     clipped_array_color = np.clip(image_array.astype(np.float32), display_min, display_max)
                     if scale_factor > 0:
                        normalized_array_float_color = (clipped_array_color - display_min) / scale_factor
                     else:
                        normalized_array_float_color = np.zeros_like(clipped_array_color, dtype=np.float32)
                     # BGR -> RGB 変換
                     if color_id == 101:
                         normalized_array_float_color = normalized_array_float_color[..., ::-1]
                     display_array_final = (normalized_array_float_color * 255).astype(np.uint8)

            elif color_id >= 8 and color_id <= 19: # BAYER
                 print("情報: BAYERパターンは現在デモザイクされず、モノクロ画像として表示されます。", file=sys.stderr)

            img_for_display = Image.fromarray(display_array_final, mode=mode)

            # --- 画像表示 ---
            window_title = f"{ser_file_path} - Frame {args.frame_number} (Display Range: {display_min:.1f}-{display_max:.1f})"
            img_for_display.show(title=window_title)
            print(f"フレーム {args.frame_number} を表示します... (表示ウィンドウを閉じてください)")

            # --- ヒストグラム表示 (オプション) ---
            if show_histogram:
                print("ヒストグラムを生成・表示します...")
                # ヒストグラムは元のデータ (image_array) で作成
                hist_data = image_array
                if num_planes == 3:
                    hist_data = np.mean(image_array, axis=2).astype(image_array.dtype)
                    print("情報: カラー画像のため、輝度（RGB平均）のヒストグラムを表示します。")

                plt.figure()
                bins = 256 if dtype == np.uint8 else 1024
                n, bins_edges, patches = plt.hist(hist_data.ravel(), bins=bins, range=(min_val, max_val + 1))
                plt.title(f'Pixel Value Histogram (Frame {args.frame_number})')
                plt.xlabel('Pixel Value')
                plt.ylabel('Frequency (Count)')
                plt.grid(True)
                plt.ticklabel_format(style='plain', axis='y')
                # plt.yscale('log') # Logスケールは一旦コメントアウト

                # 表示範囲を示す縦線を追加
                plt.axvline(display_min, color='r', linestyle='dashed', linewidth=1, label=f'Clip Low ({clip_low_percent}%)')
                plt.axvline(display_max, color='g', linestyle='dashed', linewidth=1, label=f'Clip High ({clip_high_percent}%)')
                plt.legend() # 凡例を表示

                plt.show()
                print("ヒストグラムウィンドウを閉じてください。")

    except FileNotFoundError:
        print(f"エラー: ファイル '{ser_file_path}' が見つかりません。", file=sys.stderr)
        sys.exit(1)
    except (ValueError, IOError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
         missing_lib = str(e).split("'")[-2]
         install_cmd = "pip install numpy Pillow"
         if missing_lib == 'matplotlib':
             install_cmd += " matplotlib"
         print(f"エラー: 必要なライブラリ '{missing_lib}' が見つかりません。{install_cmd} を実行してください。", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 