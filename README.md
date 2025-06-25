# whisper-colab-transcribe 🐝

**10 秒ごとに高精度文字起こしができる Colab 向け超軽量スクリプト**

* 依存ライブラリ 3 つ（`whisper` / `soundfile` / `ffmpeg-python`）だけ
* 100 行以内の Python ファイル 1 本で完結
* 入力は任意の音声・動画、出力は TXT ＋ JSON
* Google Drive 連携でファイルのアップロード不要

---

## クイックスタート（Google Colab）

```bash
# 1. このリポジトリをクローン
!git clone https://github.com/your-name/whisper-ultralite.git
%cd whisper-ultralite

# 2. Colab ランタイムを GPU に設定
# 3. スクリプトを実行
!python whisper_ultralite_colab.py \
       --src /content/drive/MyDrive/sample.m4a \
       --model small
```

> **所要時間:** Whisper モデル DL + 推論あわせて 1 分前後（small, 1 分音声）

---

## パラメータ一覧

| 引数        | デフォルト   | 説明                                          |
| --------- | ------- | ------------------------------------------- |
| `--src`   | 必須      | 文字起こししたい音声 / 動画ファイルのパス                      |
| `--model` | `small` | `tiny / base / small / medium / large` から選択 |
| `--clip`  | `10`    | 何秒ごとに分割して処理するか（10 で固定推奨）                    |
| `--hpass` | `100`   | ハイパス周波数 (Hz)                                |
| `--lpass` | `8000`  | ローパス周波数 (Hz)                                |

---

## 出力例

```
[00:00-00:10] こんにちは。本日は Whisperのデモを行います。
[00:10-00:20] このスクリプトは 10 秒ごとに文字起こしを行います。
```

JSON ファイルも同名で生成され、下記のような構造になります。

```json
[
  {"start": 0.0, "end": 10.0, "text": "こんにちは。本日は …"},
  {"start": 10.0, "end": 20.0, "text": "このスクリプトは …"}
]
```

---

## よくある質問

<details>
<summary>GPU が無い環境でも動きますか？</summary>
CPU でも動作しますが、推論速度が 5〜10 倍ほど遅くなります。短い音声での検証を推奨します。
</details>


---

## ライセンス

MIT License © 2025 haruki0619
