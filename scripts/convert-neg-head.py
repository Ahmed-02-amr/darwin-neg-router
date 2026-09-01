from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

from safetensors.torch import load_file


MAGIC = b"DNEGv001"
VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the released Darwin NEG safetensors sidecar to the native runner format."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    state = load_file(str(args.source), device="cpu")
    down_weight = state["head.proj_down.weight"].float().contiguous()
    down_bias = state["head.proj_down.bias"].float().contiguous()
    out_weight = state["head.proj_out.weight"].float().contiguous().view(-1)
    out_bias = state["head.proj_out.bias"].float().contiguous().view(-1)

    reduced, hidden = down_weight.shape
    if down_bias.shape != (reduced,) or out_weight.shape != (reduced,) or out_bias.shape != (1,):
        raise ValueError("Unexpected Darwin NEG tensor dimensions")

    threshold = float(state["gate.threshold"].item())
    temperature_scale = float(state["gate.temp_scale"].item())
    top_k = 20

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("wb") as output:
        output.write(MAGIC)
        output.write(
            struct.pack(
                "<IIIIff",
                VERSION,
                hidden,
                reduced,
                top_k,
                threshold,
                temperature_scale,
            )
        )
        for tensor in (down_weight, down_bias, out_weight, out_bias):
            output.write(tensor.numpy().tobytes(order="C"))

    digest = hashlib.sha256(args.destination.read_bytes()).hexdigest()
    print(f"wrote={args.destination}")
    print(f"sha256={digest}")
    print(f"hidden={hidden} reduced={reduced} top_k={top_k}")
    print(f"threshold={threshold:.10f} temperature_scale={temperature_scale:.10f}")


if __name__ == "__main__":
    main()
