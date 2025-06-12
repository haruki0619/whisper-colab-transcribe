# 🎙️ Whisper Colab Transcribe

Google Colab で **1 セル実行 → 日本語音声を JSON / TXT に文字起こし**  
ノイズ除去（High-pass / Low-pass）＋ 5 分刻み分割に対応します。

## Demo (Open in Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<ユーザ名>/whisper-colab-transcribe/blob/main/colab_notebook.ipynb)

## Features

| 機能 | 説明 |
|---|---|
| ノイズ除去 | `ffmpeg` で High-Pass / Low-Pass (100–8000 Hz) |
| モデル自動切替 | VRAM が足りない場合は `medium` → `small` |
| 断片処理 | 300 秒ごとに分割して並列推論も容易 |
| 出力 | `xxx.json` + `xxx.txt`（タイムタグ付き） |

## Usage

1. Google Drive に音声をアップロード  
2. Colab で `SRC_PATH` を書き換えて **Runtime ▸ Run all**  

**ローカル実行**（GPU/CPU 共通）

```bash
git clone https://github.com/<user>/whisper-colab-transcribe.git
cd whisper-colab-transcribe
pip install -r requirements.txt
python transcribe.py --src ./sample.wav --model medium --seg 300
