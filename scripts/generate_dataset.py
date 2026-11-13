#!/usr/bin/env python3
import argparse
import math
import os
import random
import struct
import zlib


def write_pgm(path: str, width: int, height: int, pixels: bytes) -> None:
    with open(path, "wb") as f:
        header = f"P5\n{width} {height}\n255\n".encode("ascii")
        f.write(header)
        f.write(pixels)


def read_pgm(path: str):
    with open(path, "rb") as f:
        magic = f.readline().strip()
        assert magic == b"P5"
        line = f.readline().strip()
        while line.startswith(b"#"):
            line = f.readline().strip()
        width, height = map(int, line.split())
        _max_val = int(f.readline().strip())
        pixels = f.read(width * height)
    return width, height, pixels


def write_grayscale_png(path: str, width: int, height: int, pixels: bytes) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw_rows = bytearray()
    for r in range(height):
        raw_rows.append(0)  # Filter type 0 (None)
        raw_rows.extend(pixels[r * width : (r + 1) * width])

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw_rows), level=6))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png_bytes)


def generate_batch(output_dir: str, count: int, width: int, height: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(42)
    for idx in range(count):
        cx = rng.randint(50, width - 50)
        cy = rng.randint(50, height - 50)
        radius = rng.randint(25, 65)
        box_x0 = rng.randint(20, width - 90)
        box_y0 = rng.randint(20, height - 90)

        buf = bytearray(width * height)
        for y in range(height):
            for x in range(width):
                base = 30
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    base = 210
                elif box_x0 <= x <= box_x0 + 60 and box_y0 <= y <= box_y0 + 60:
                    base = 165
                elif int(128 + 50 * math.sin(x * 0.08) * math.cos(y * 0.08)) > 165:
                    base = 120

                # Add sensor noise (Gaussian + impulse)
                noise = int(rng.gauss(0, 18))
                if rng.random() < 0.01:
                    val = 255 if rng.random() < 0.5 else 0
                else:
                    val = max(0, min(255, base + noise))
                buf[y * width + x] = val

        write_pgm(os.path.join(output_dir, f"img_{idx:04d}.pgm"), width, height, bytes(buf))


def export_sample_pngs(input_dir: str, output_dir: str, artifact_dir: str, samples: int = 5) -> None:
    os.makedirs(artifact_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".pgm"))[:samples]
    for fname in files:
        stem = os.path.splitext(fname)[0]
        w, h, before_px = read_pgm(os.path.join(input_dir, fname))
        _, _, after_px = read_pgm(os.path.join(output_dir, fname))
        write_grayscale_png(os.path.join(artifact_dir, f"{stem}_before.png"), w, h, before_px)
        write_grayscale_png(os.path.join(artifact_dir, f"{stem}_after_sobel.png"), w, h, after_px)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "export_png"], required=True)
    parser.add_argument("--input_dir", default="input_data")
    parser.add_argument("--output_dir", default="output_data")
    parser.add_argument("--artifact_dir", default="artifacts")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    if args.mode == "generate":
        generate_batch(args.input_dir, args.count, args.width, args.height)
        print(f"Generated {args.count} noisy {args.width}x{args.height} PGM images in '{args.input_dir}/'.")
    elif args.mode == "export_png":
        export_sample_pngs(args.input_dir, args.output_dir, args.artifact_dir)
        print(f"Exported before/after PNG comparisons to '{args.artifact_dir}/'.")
