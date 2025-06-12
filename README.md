# 🎙️ Whisper Colab & CLI Transcriber

**Whisper Colab Transcribe** は、OpenAI Whisper と FFmpeg を使って **完全無料で** 日本語音声を高速かつ手軽に文字起こしできるツールです。Google Colab でもローカル CLI でも同じスクリプトで動作し、ノイズ除去・長時間ファイル分割・タイムスタンプ付き JSON/TXT の同時出力をワンコマンドで実現します。

---

## ✨ 主な機能

| 機能      | 説明                                                        |
| ------- | --------------------------------------------------------- |
| ノイズ除去   | `ffmpeg` の `highpass` / `lowpass` フィルタで 100–8000 Hz 帯域を抽出 |
| 分割処理    | 5 分 (= 300 秒) ごとに音声を分割／並列化することで長時間ファイルも安定処理               |
| 自動モデル切替 | 指定モデルが VRAM 不足なら `small` へフォールバック                         |
| 出力形式    | `<base>.json` (タイムタグ付セグメント) & `<base>.txt` (全文)           |
| 詳細ログ    | `-v/--verbose` で FFmpeg / Whisper の stderr を表示            |

---

## 🚀 クイックスタート

### 1) Colab で試す

1. 上の **Open In Colab** バッジをクリック
2. セル先頭で `SRC_PATH` を自分の音声ファイルに変更
3. **Runtime ▸ Run all** を実行
4. `/content` に `.json` と `.txt` が保存されます

### 2) ローカル CLI で使う

> **前提**: システムに `ffmpeg` がインストールされていること。未導入の場合は `brew install ffmpeg` (macOS) または `apt-get install ffmpeg` (Debian 系 Linux) などで導入してください。

```bash
# 依存ライブラリをインストール (CUDA に合わせて torch を追加しても OK)
$ pip install -r requirements.txt

# 標準設定 (model=medium, seg=300)
$ python transcribe.py ~/audio.m4a

# オプション指定
$ python transcribe.py in.wav -m small -s 600 -o out/meeting -v
```

| 主要オプション         | デフォルト    | 説明                                             |
| --------------- | -------- | ---------------------------------------------- |
| `-m, --model`   | `medium` | Whisper モデル名 (`tiny`/`small`/`medium`/`large`) |
| `-d, --device`  | auto     | 使用デバイス (`cuda` / `cpu`)                        |
| `-s, --seg`     | `300`    | 分割長 (秒)                                        |
| `-o, --out`     | ―        | 出力ファイル名ベース (`.json`/`.txt` は自動付与)              |
| `-v, --verbose` | off      | FFmpeg/FFprobe の stderr を表示                    |

---

## 📂 プロジェクト構成

```text
whisper-colab-transcribe/
├── transcribe.py            # メインスクリプト (GPU/CPU 共通)
├── colab_notebook.ipynb     # Colab デモ
├── requirements.txt         # 最小依存
├── README.md                # ← 本ファイル
├── LICENSE                  # MIT
```

---

## 💡 Tips

* **GPU なのに CPU で実行される場合**

  * `nvidia-smi` で空き VRAM を確認し、足りなければ `-m small` を指定します。
  * それでも GPU が使われない場合は `--verbose` で詳細ログを確認し、CUDA の初期化エラーなどをチェックしてください。

* **`ffprobe error`**\*\* が出る場合\*\*

  * `ffmpeg`/`ffprobe` が未インストール、または PATH が通っていない可能性があります。
  * 例: macOS なら `brew install ffmpeg`、Ubuntu なら `sudo apt install ffmpeg` で導入できます。

* **長時間ファイル (> 1h) を高速処理したい場合**

  * `-s` を短く設定し、複数プロセスや複数 GPU で `transcribe.py` を並列実行すると処理が速くなります。

---

## 📝 ライセンス

本リポジトリは **MIT License** の下で公開されています。詳細は [LICENSE](LICENSE) を参照してください。

---
