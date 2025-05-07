import struct
import sys
import datetime
import argparse # argparse をインポート

def read_ser_header(f):
    """SERファイルのヘッダーを読み込み、辞書として返す"""
    header_data = f.read(178)
    if len(header_data) < 178:
        raise ValueError("ヘッダーの読み込みに失敗しました。ファイルサイズが不足しています。")

    # ヘッダーは常にリトルエンディアン
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
        'q '     # DateTime (Ticksとして解釈してみる)
        'q '     # DateTimeUTC (Ticks)
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
        "DateTime_Ticks": unpacked_header[11], # Ticks値として解釈
        "DateTimeUTC_Ticks": unpacked_header[12],
    }

    # DateTime Ticksをdatetimeオブジェクトに変換 (UTCと仮定して変換するが、表示時にLocalと明記)
    try:
        ticks_offset = (datetime.datetime(1970, 1, 1) - datetime.datetime(1, 1, 1)).total_seconds() * 10**7
        timestamp_100ns_since_epoch = header["DateTime_Ticks"] - int(ticks_offset)
        timestamp_us_since_epoch = timestamp_100ns_since_epoch // 10
        # タイムゾーン情報は付与せずに変換
        header["DateTime"] = datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=timestamp_us_since_epoch)
    except OverflowError:
        header["DateTime"] = "変換エラー (Overflow)"
    except Exception as e:
        # Ticksが0以下の場合もここに該当する可能性があるが、別途チェックする
        if header["DateTime_Ticks"] <= 0:
            header["DateTime"] = "無効 (0以下)"
        else:
            header["DateTime"] = f"変換エラー: {e} (Ticks: {header.get('DateTime_Ticks')})"

    # DateTimeUTC Ticksをdatetimeオブジェクトに変換
    try:
        ticks_offset = (datetime.datetime(1970, 1, 1) - datetime.datetime(1, 1, 1)).total_seconds() * 10**7
        timestamp_100ns_since_epoch = header["DateTimeUTC_Ticks"] - int(ticks_offset)
        timestamp_us_since_epoch = timestamp_100ns_since_epoch // 10
        header["DateTimeUTC"] = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(microseconds=timestamp_us_since_epoch)
    except OverflowError:
        header["DateTimeUTC"] = "変換エラー (Overflow)"
    except Exception as e:
        header["DateTimeUTC"] = f"変換エラー: {e} (Ticks: {header.get('DateTimeUTC_Ticks')})"

    return header

def read_ser_trailer(f, frame_count):
    """SERファイルのトレーラー (タイムスタンプ) を読み込み、リストとして返す"""
    timestamp_size = 8  # DateTimeUTCは8バイト
    trailer_size = timestamp_size * frame_count
    endian_prefix = '<' # トレーラーもリトルエンディアンと仮定

    if frame_count == 0:
        return [] # フレーム数が0ならタイムスタンプも0個

    try:
        # ファイルの末尾から読み込む
        f.seek(-trailer_size, 2) # 2は末尾からのseekを示す
        trailer_data = f.read(trailer_size)
    except OSError:
        # ファイルサイズが不足している場合など
        raise ValueError(f"トレーラーの読み込み位置({trailer_size}バイト)への移動、または読み込みに失敗しました。ファイルが破損しているか、フレーム数が正しくありません。")

    if len(trailer_data) < trailer_size:
        raise ValueError(f"トレーラーの読み込みに失敗しました。期待されるサイズ ({trailer_size} バイト) に対して、読み込めたのは {len(trailer_data)} バイトです。")

    timestamps_ticks = []
    timestamp_format = endian_prefix + 'q' # 8byte signed integer (long long)
    for i in range(frame_count):
        offset = i * timestamp_size
        try:
            ticks = struct.unpack_from(timestamp_format, trailer_data, offset)[0]
            timestamps_ticks.append(ticks)
        except struct.error as e:
            print(f"警告: タイムスタンプ {i+1} の unpack に失敗しました: {e}", file=sys.stderr)
            timestamps_ticks.append(None) # エラーの場合はNoneを追加

    return timestamps_ticks

def format_timestamp_ticks(ticks):
    """Ticks値を人間が読める形式のUTC文字列に変換"""
    if ticks is None:
        return "読み取りエラー"
    try:
        # 0001年1月1日から1970年1月1日までのTicksオフセット
        ticks_offset = (datetime.datetime(1970, 1, 1) - datetime.datetime(1, 1, 1)).total_seconds() * 10**7
        timestamp_100ns_since_epoch = ticks - int(ticks_offset)
        timestamp_us_since_epoch = timestamp_100ns_since_epoch // 10
        dt_utc = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(microseconds=timestamp_us_since_epoch)
        # マイクロ秒まで表示する場合
        return dt_utc.strftime('%Y-%m-%d %H:%M:%S.%f') + " UTC"
    except OverflowError:
        return f"変換エラー (Overflow): Ticks={ticks}"
    except Exception as e:
        return f"変換エラー: {e}, Ticks={ticks}"

def calculate_bytes_per_frame(header):
    """ヘッダー情報から1フレームあたりのバイト数を計算する"""
    width = header['Width']
    height = header['Height']
    pixel_depth = header['PixelDepthPerPlane']
    color_id = header['ColorID']

    # プレーン数を決定
    num_planes = 1 # MONO or BAYER (default)
    if color_id == 100 or color_id == 101: # RGB or BGR
        num_planes = 3

    # 1プレーンあたりのバイト数を計算
    bytes_per_plane_pixel = 1
    if pixel_depth > 8:
        bytes_per_plane_pixel = 2
    elif pixel_depth <= 0:
         raise ValueError(f"PixelDepthPerPlane が不正な値です: {pixel_depth}")

    if pixel_depth > 16:
         print(f"警告: PixelDepthPerPlane ({pixel_depth}) が16を超えています。計算は16bitとして扱います。", file=sys.stderr)
         bytes_per_plane_pixel = 2

    # 1ピクセルあたりの総バイト数
    bytes_per_pixel_total = bytes_per_plane_pixel * num_planes

    # 1フレームの総バイト数
    frame_size = width * height * bytes_per_pixel_total
    return frame_size

def main():
    # コマンドライン引数のパーサーを設定
    parser = argparse.ArgumentParser(description='SERファイルのヘッダーとトレーラー情報を表示します。')
    parser.add_argument('ser_file_path', help='SERファイルのパス')
    parser.add_argument('-a', '--all-timestamps', action='store_true', help='全てのタイムスタンプを表示します')
    parser.add_argument('-o', '--show-offsets', action='store_true', help='各フレームの画像データ開始オフセットを表示します')

    # 引数をパース
    args = parser.parse_args()
    ser_file_path = args.ser_file_path
    show_all_timestamps = args.all_timestamps
    show_offsets = args.show_offsets

    try:
        with open(ser_file_path, 'rb') as f:
            print(f"--- {ser_file_path} のヘッダー情報 ---")
            header = read_ser_header(f)
            for key, value in header.items():
                if key == "DateTime_Ticks":
                    print(f"  DateTime (Ticks): {value}")
                elif key == "DateTime":
                    if isinstance(value, datetime.datetime):
                        # ローカルタイムとして表示
                        print(f"  DateTime (Converted): {value.strftime('%Y-%m-%d %H:%M:%S.%f')} (Local Time)")
                    else:
                        print(f"  DateTime (Converted): {value}") # エラーメッセージ("無効 (0以下)"など)
                elif key == "DateTimeUTC_Ticks":
                    print(f"  DateTimeUTC (Ticks): {value}")
                elif key == "DateTimeUTC":
                    if isinstance(value, datetime.datetime):
                        print(f"  DateTimeUTC (Converted): {value.strftime('%Y-%m-%d %H:%M:%S.%f')} UTC")
                    else:
                        print(f"  DateTimeUTC (Converted): {value}")
                else:
                    print(f"  {key}: {value}")

            # DateTime_Ticks が 0 以下ならトレーラーは無効
            trailer_is_valid = header["DateTime_Ticks"] > 0

            # 画像フレームサイズ計算
            try:
                image_frame_size = calculate_bytes_per_frame(header)
            except ValueError as e:
                print(f"エラー: 画像フレームサイズの計算に失敗しました: {e}", file=sys.stderr)
                # オフセット表示ができなくなるので無効化
                show_offsets = False
                image_frame_size = 0 # 計算不能

            header_size = 178 # ヘッダは178バイト

            print("\n--- トレーラー情報 (タイムスタンプ) ---")
            if not trailer_is_valid:
                print("  DateTimeヘッダー値が0以下であるため、トレーラーは無効です。")
            elif header["FrameCount"] > 0:
                timestamps_ticks = read_ser_trailer(f, header["FrameCount"])
                print(f"  フレーム数: {header['FrameCount']}")

                if show_all_timestamps:
                    print(f"  全タイムスタンプ ({len(timestamps_ticks)}個) (UTC)" + (" と画像オフセット" if show_offsets else ""))
                    for i, ticks in enumerate(timestamps_ticks):
                        frame_offset_str = ""
                        if show_offsets and image_frame_size > 0:
                            current_frame_offset = header_size + i * image_frame_size
                            frame_offset_str = f" | Offset: {hex(current_frame_offset)}"
                        elif show_offsets:
                            frame_offset_str = " | Offset: 計算不能"
                        print(f"    Frame {i+1}: {format_timestamp_ticks(ticks)}{frame_offset_str}")
                else:
                    # 省略表示
                    num_timestamps_to_show = 5
                    print(f"  タイムスタンプ (UTC)" + (" と画像オフセット" if show_offsets else ""))
                    if len(timestamps_ticks) <= 2 * num_timestamps_to_show:
                        # フレーム数が少ない場合は全表示
                        for i, ticks in enumerate(timestamps_ticks):
                            frame_offset_str = ""
                            if show_offsets and image_frame_size > 0:
                                current_frame_offset = header_size + i * image_frame_size
                                frame_offset_str = f" | Offset: {hex(current_frame_offset)}"
                            elif show_offsets:
                                frame_offset_str = " | Offset: 計算不能"
                            print(f"    Frame {i+1}: {format_timestamp_ticks(ticks)}{frame_offset_str}")
                    else:
                        # 最初と最後を表示
                        print(f"  最初の{num_timestamps_to_show}個:")
                        for i, ticks in enumerate(timestamps_ticks[:num_timestamps_to_show]):
                            frame_offset_str = ""
                            if show_offsets and image_frame_size > 0:
                                current_frame_offset = header_size + i * image_frame_size
                                frame_offset_str = f" | Offset: {hex(current_frame_offset)}"
                            elif show_offsets:
                                frame_offset_str = " | Offset: 計算不能"
                            print(f"    Frame {i+1}: {format_timestamp_ticks(ticks)}{frame_offset_str}")
                        print("    ...")
                        print(f"  最後の{num_timestamps_to_show}個:")
                        for i, ticks in enumerate(timestamps_ticks[-num_timestamps_to_show:]):
                            frame_index = header['FrameCount'] - num_timestamps_to_show + i
                            frame_offset_str = ""
                            if show_offsets and image_frame_size > 0:
                                current_frame_offset = header_size + frame_index * image_frame_size
                                frame_offset_str = f" | Offset: {hex(current_frame_offset)}"
                            elif show_offsets:
                                frame_offset_str = " | Offset: 計算不能"
                            print(f"    Frame {frame_index + 1}: {format_timestamp_ticks(ticks)}{frame_offset_str}")
            else:
                print("  フレーム数が0のため、タイムスタンプはありません。")

    except FileNotFoundError:
        print(f"エラー: ファイル '{ser_file_path}' が見つかりません。", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 