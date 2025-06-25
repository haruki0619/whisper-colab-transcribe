# ================================================
#  Whisper Ultra‑Lite – Colab 用
# ------------------------------------------------
# 手順（Google Colab）
#   1. 画面上部「ランタイム」→「ランタイムのタイプを変更」→ GPU を選択
#   2. Drive に音声ファイルを置き、下の SRC_PATH を書き換え
#   3. 上から順番にセルを実行（Shift+Enter）
#   4. 同フォルダに <音声名>.txt / .json が生成される
# ================================================

# 0) Google Drive をマウント（Drive 不使用ならコメントアウト可）
from google.colab import drive

drive.mount('/content/drive')

# 1) ライブラリをインストール（所要 ≈30 秒）
!pip -q install git+https://github.com/openai/whisper.git soundfile ffmpeg-python

# 2) 必要モジュールを読み込み
import subprocess, json, math, os, tempfile, pathlib, warnings, shutil
import torch, whisper, soundfile as sf
warnings.filterwarnings('ignore')  # 警告表示をオフ

# 3) ★ ユーザが変更するパラメータ ★ -------------------------
SRC_PATH    = ''  # 文字起こししたい音声/動画
MODEL_NAME  = 'small'     # tiny / base / small / medium / large
SEG_LEN_SEC = 300         # 分割長（秒）0 なら全体を一気に処理
HIGH_PASS   = 100         # ハイパス (Hz) : 低域ノイズをカット
LOW_PASS    = 8000        # ローパス (Hz) : 高域ノイズをカット
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------------

# 4) ffmpeg が入っていない Colab イメージ対策（ほぼ不要だが安全のため）
if not shutil.which('ffmpeg'):
    subprocess.check_call(['apt-get', '-y', 'update'])
    subprocess.check_call(['apt-get', '-y', 'install', 'ffmpeg'])

# 5) 任意フォーマット → 16 kHz モノラル WAV へ変換（＋簡易ノイズ除去）
print('▶️  音声を WAV に変換中...')
_tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
subprocess.check_call([
    'ffmpeg', '-y', '-loglevel', 'error',
    '-i', SRC_PATH,
    '-af', f'highpass=f={HIGH_PASS},lowpass=f={LOW_PASS}',
    '-ar', '16000', '-ac', '1', _tmp_wav
])

# 6) 音声長（秒数）を取得
_duration = len(sf.SoundFile(_tmp_wav)) / 16000  # 16 kHz 固定なのでサンプル数÷16k
clip_len  = SEG_LEN_SEC or _duration             # 0 の場合は分割しない
print(f'⏱️  全体長: {_duration:.1f} 秒 , 分割長: {clip_len} 秒')

# 7) Whisper モデルをロード
print(f'🚀 Whisper "{MODEL_NAME}" をロード ({DEVICE})...')
_model = whisper.load_model(MODEL_NAME, device=DEVICE)

# 8) 分割して逐次文字起こし（メイン処理）
_segments = []
for idx in range(0, math.ceil(_duration / clip_len)):
    start_sec = idx * clip_len
    # ffmpeg で区間切り出し
    _clip_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    subprocess.check_call([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', _tmp_wav,
        '-ss', str(start_sec),
        '-t', str(min(clip_len, _duration - start_sec)),
        _clip_wav
    ])
    # Whisper 推論（GPU の場合 fp16=true）
    result = _model.transcribe(_clip_wav, fp16=(DEVICE == 'cuda'))
    text   = result['text'].strip()
    end_sec = start_sec + (result['segments'][-1]['end'] if result['segments'] else 0)
    _segments.append({'start': start_sec, 'end': end_sec, 'text': text})
    os.remove(_clip_wav)  # 一時ファイル削除
    print(f'  ✔️  {idx+1}/{math.ceil(_duration/clip_len)} クリップ完了')

# 9) 生成した一時 WAV を削除
os.remove(_tmp_wav)

# 10) TXT と JSON に保存
stem      = pathlib.Path(SRC_PATH).with_suffix('').name  # 拡張子を外したベース名
_txt_path = f'{stem}.txt'
_json_path = f'{stem}.json'

with open(_txt_path, 'w', encoding='utf-8') as f_txt:
    for seg in _segments:
        f_txt.write(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}\n")

with open(_json_path, 'w', encoding='utf-8') as f_json:
    json.dump(_segments, f_json, ensure_ascii=False, indent=2)

print('\n✅ 文字起こしが完了しました →', _txt_path, _json_path)
