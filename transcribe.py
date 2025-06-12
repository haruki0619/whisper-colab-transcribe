#!/usr/bin/env python3
"""
transcribe.py
-------------

CLI ツール: 音声ファイル → JSON / TXT 文字起こし（Whisper + FFmpeg）

Examples
--------
$ python transcribe.py input.m4a                     # 標準設定で実行
$ python transcribe.py input.wav -m small -s 600     # モデル/分割秒数を変更
$ python transcribe.py in.mp3 -o out/mtg -v          # 出力名指定 + 詳細ログ

Dependencies
------------
$ pip install git+https://github.com/openai/whisper.git soundfile ffmpeg-python
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import tempfile
from typing import List, Dict

import torch
import whisper

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
HPASS_HZ: int = 100            # ハイパスフィルタ閾値 [Hz]
LPASS_HZ: int = 8_000          # ローパスフィルタ閾値 [Hz]
DEFAULT_SEG_LEN: int = 300     # 分割長 [秒]
DEFAULT_MODEL: str = "medium"  # Whisper モデル

# ---------------------------------------------------------------------------
# ヘルパ関数
# ---------------------------------------------------------------------------

def _run(cmd: List[str], verbose: bool = False) -> None:
    """サブプロセス実行。verbose が False のときは出力を抑制。"""
    subprocess.run(
        cmd,
        check=True,
        **({} if verbose else {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}),
    )


def _get_duration(path: pathlib.Path, verbose: bool = False) -> float:
    """音声長（秒）を取得。"""
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        stderr=None if verbose else subprocess.DEVNULL,
    )
    return float(out.strip())


def _filter_audio(src: pathlib.Path, dst: pathlib.Path, verbose: bool = False) -> None:
    """16 kHz mono WAV に変換 & 簡易ノイズ除去。"""
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-af",
            f"highpass=f={HPASS_HZ},lowpass=f={LPASS_HZ}",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dst),
        ],
        verbose=verbose,
    )


def _smart_load_model(name: str, device: str):  # -> whisper.model.Whisper
    """VRAM 不足なら自動で small にフォールバックしてロード。
    明示的に model.to(device) でデバイスを固定する。"""
    try:
        model = whisper.load_model(name, device=device)
    except RuntimeError:
        print(f"[WARN] '{name}' model requires more VRAM → falling back to 'small'")
        model = whisper.load_model("small", device=device)
    # Whisper 内部で device が指定されても、何らかの理由で CPU にロードされる可能性がある。
    # 明示的に to(device) しておくと誤推論を防げる。
    try:
        model.to(device)
    except Exception as e:
        print(f"[WARN] Failed to move model to {device}: {e}")
    return model


def _transcribe(
    wav: pathlib.Path,
    model,  # whisper model instance
    segment_len: int,
    device: str,
    tmpdir: pathlib.Path,
    verbose: bool = False,
) -> List[Dict]:
    """`wav` を `segment_len` 秒ごとに分割し、順次文字起こし。"""
    duration = _get_duration(wav, verbose)
    segments: List[Dict] = []

    for start in range(0, math.ceil(duration), segment_len):
        clip = tmpdir / f"clip_{start}.wav"
        _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(wav),
                "-ss",
                str(start),
                "-t",
                str(segment_len),
                str(clip),
            ],
            verbose=verbose,
        )

        result = model.transcribe(clip, language="ja", fp16=(device == "cuda"))
        for seg in result["segments"]:
            seg["start"] += start
            seg["end"] += start
            segments.append(seg)

        processed = min(start + segment_len, duration)
        print(f"\r⏳  {processed:.0f}/{duration:.0f} sec processed", end="", flush=True)

    print()  # 改行
    return segments


def _write_outputs(segments: List[Dict], base_path: pathlib.Path) -> None:
    """<base>.json と <base>.txt を保存。"""
    json_path = base_path.with_suffix(".json")
    txt_path = base_path.with_suffix(".txt")

    json_path.write_text(
        json.dumps({"segments": segments}, ensure_ascii=False, indent=2), "utf-8"
    )
    txt_lines = [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments]
    txt_path.write_text("\n".join(txt_lines), "utf-8")

    print("✅  Saved:")
    print("   JSON →", json_path)
    print("   TXT  →", txt_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Noise‑reduced Whisper transcription")
    p.add_argument("src", help="input audio file (.wav/.m4a/.mp3…)")
    p.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "-s",
        "--seg",
        type=int,
        default=DEFAULT_SEG_LEN,
        help=f"segment length in seconds (default: {DEFAULT_SEG_LEN})",
    )
    p.add_argument("-o", "--out", help="output basename (without extension)")
    p.add_argument("-v", "--verbose", action="store_true", help="show ffmpeg / ffprobe output")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    src_path = pathlib.Path(args.src).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    out_base = (
        pathlib.Path(args.out).expanduser().resolve() if args.out else src_path.with_suffix("")
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _smart_load_model(args.model, device)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        clean_wav = tmpdir / "clean.wav"
        _filter_audio(src_path, clean_wav, verbose=args.verbose)
        segments = _transcribe(
            clean_wav, model, args.seg, device, tmpdir, verbose=args.verbose
        )

    _write_outputs(segments, out_base)


if __name__ == "__main__":
    main()
