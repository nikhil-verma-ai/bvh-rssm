"""BVH-RSSM serving entry point.

Usage:
  # Serve from a checkpoint:
  python scripts/serve.py --checkpoint runs/phase2/step10000.pt

  # Smoke-test mode (untrained model, no checkpoint file needed):
  python scripts/serve.py --fast-mode

  # Custom host/port:
  python scripts/serve.py --checkpoint runs/phase2/step10000.pt --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch BVH-RSSM inference server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a Checkpointer .pt file. Required unless --fast-mode is set.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        default=False,
        help="Use an untrained predictor (no checkpoint needed). For smoke testing only.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (e.g. 'cpu', 'cuda:0').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes. Must be 1 if using CUDA (no fork-after-CUDA).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.fast_mode and args.checkpoint is None:
        print(
            "ERROR: --checkpoint is required unless --fast-mode is set.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.workers > 1 and args.device.startswith("cuda"):
        print(
            "WARNING: --workers > 1 with CUDA is unsafe (fork after CUDA init). "
            "Falling back to 1 worker.",
            file=sys.stderr,
        )
        args.workers = 1

    # Import here so the error message above fires before any heavy imports
    from bvh_rssm.serving.predictor import Predictor
    from bvh_rssm.serving.server import create_app

    if args.fast_mode:
        print("INFO: fast-mode — using untrained predictor (outputs are meaningless)")
        predictor = Predictor.from_scratch(fast_mode=True)
    else:
        print(f"INFO: loading checkpoint from {args.checkpoint}")
        predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)

    app = create_app(predictor)

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
