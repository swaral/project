"""Build a pushable Kaggle kernel for the GPU pilot.

Embeds the current src/, scripts/ and pyproject.toml into kaggle/pilot_job.py
and writes kaggle/build/{run_pilot.py,kernel-metadata.json}. Push it with:

    python kaggle/build_kernel.py --username <kaggle-user>
    kaggle kernels push -p kaggle/build
    kaggle kernels status <kaggle-user>/llm-session-reco-pilot
    kaggle kernels output <kaggle-user>/llm-session-reco-pilot -p data/processed/kaggle_pilot
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_SLUG = "llm-session-reco-pilot"


def _source_archive() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in ("pyproject.toml", "src", "scripts"):
            tar.add(
                ROOT / name,
                arcname=name,
                filter=lambda info: None
                if "__pycache__" in info.name or info.name.endswith(".egg-info")
                else info,
            )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "kaggle" / "build")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    job = (ROOT / "kaggle" / "pilot_job.py").read_text(encoding="utf-8")
    script = (
        job
        + f"\n\nPAYLOAD = {_source_archive()!r}\n\n"
        + 'if __name__ == "__main__":\n    main()\n'
    )
    (args.output_dir / "run_pilot.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": f"{args.username}/{KERNEL_SLUG}",
        "title": "LLM Session Reco Pilot",
        "code_file": "run_pilot.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (args.output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output_dir} ({len(script) // 1024} KiB)")


if __name__ == "__main__":
    main()
