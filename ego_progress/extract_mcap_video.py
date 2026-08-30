"""Extract a Foxglove H.264 camera topic from an MCAP into a browser MP4."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


def extract_h264_video(mcap_path: Path, output_path: Path, topic: str) -> tuple[int, float]:
    """Copy H.264 access units from ``topic`` and remux them into MP4."""
    first_time = None
    last_time = None
    frame_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mcap-video-") as temp_dir:
        elementary_stream = Path(temp_dir) / "camera.h264"
        with mcap_path.open("rb") as source, elementary_stream.open("wb") as target:
            reader = make_reader(source, decoder_factories=[DecoderFactory()])
            for _, _, message, decoded in reader.iter_decoded_messages(topics=[topic]):
                if decoded.format.lower() != "h264":
                    raise ValueError(f"Expected H.264 on {topic}, got {decoded.format!r}")
                target.write(decoded.data)
                first_time = message.log_time if first_time is None else first_time
                last_time = message.log_time
                frame_count += 1
        if frame_count < 2 or first_time is None or last_time is None:
            raise ValueError(f"Topic {topic!r} did not contain enough video frames")
        duration_s = (last_time - first_time) / 1e9
        fps = (frame_count - 1) / duration_s
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.8f}",
            "-i", str(elementary_stream), "-c:v", "copy", "-an",
            "-movflags", "+faststart", str(output_path),
        ], check=True)
    return frame_count, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcap", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--topic", default="/robot0/sensor/camera0/compressed")
    args = parser.parse_args()
    frame_count, fps = extract_h264_video(args.mcap, args.output, args.topic)
    print(f"Wrote {frame_count} frames at {fps:.6f} FPS to {args.output}")


if __name__ == "__main__":
    main()
