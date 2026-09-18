"""
Download release checkpoints from HuggingFace dataset repo.

Expected remote layout:
  models/
    README.md
    T2FPV-eth/
      config_updated.yml
      models/checkpoint_best.pt
    T2FPV-hotel/
      ...
    T2FPV-univ/
      ...
    T2FPV-zara1/
      ...
    T2FPV-zara2/
      ...
    EgoTraj-TBD/
      config_updated.yml
      models-v1.1/checkpoint_best.pt
      models-legacy/checkpoint_best.pt

Local output layout (default):
  checkpoints/<ModelName>/
    config_updated.yml
    models/checkpoint_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import urllib.request

from huggingface_hub import HfApi, hf_hub_url


DEFAULT_REPO_ID = "ZoeyLIU1999/EgoTraj-Bench"
DEFAULT_OUTPUT_ROOT = Path("./checkpoints")

DEFAULT_RELEASES = [
    "T2FPV-eth",
    "T2FPV-hotel",
    "T2FPV-univ",
    "T2FPV-zara1",
    "T2FPV-zara2",
    "EgoTraj-TBD",
]

# Remote checkpoint subdirectory per release. EgoTraj-TBD was retrained on the
# current data release; the superseded weights remain under "models-legacy".
CKPT_SUBDIR = {"EgoTraj-TBD": "models-v1.1"}
DEFAULT_CKPT_SUBDIR = "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download EgoTraj-Bench release checkpoints from HuggingFace"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Local checkpoint root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token for gated/private repos (optional for public repo)",
    )
    parser.add_argument(
        "--releases",
        nargs="+",
        default=DEFAULT_RELEASES,
        choices=DEFAULT_RELEASES,
        help="Subset of releases to download (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without writing files",
    )
    return parser.parse_args()


def release_files(release_name: str) -> Iterable[tuple[str, Path]]:
    """
    Return pairs of (remote_path_in_repo, local_relative_path_under_release).
    """
    yield (
        f"models/{release_name}/config_updated.yml",
        Path("config_updated.yml"),
    )
    subdir = CKPT_SUBDIR.get(release_name, DEFAULT_CKPT_SUBDIR)
    yield (
        f"models/{release_name}/{subdir}/checkpoint_best.pt",
        Path("models/checkpoint_best.pt"),
    )


def download_file_to_path(url: str, dst_path: Path, token: str | None) -> None:
    """Download a file directly to destination path."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url=url, headers=headers)
    with urllib.request.urlopen(request) as response, dst_path.open("wb") as out_f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root

    releases = list(args.releases)

    print("Source repo:", args.repo_id)
    print("Output root:", output_root.resolve())
    print("Planned releases:", ", ".join(releases))

    api = HfApi(token=args.token)
    repo_files = set(api.list_repo_files(repo_id=args.repo_id, repo_type="dataset"))

    if args.dry_run:
        print("\nDry-run planned files:")
        for release_name in releases:
            for remote_path, rel_local in release_files(release_name):
                status = "OK" if remote_path in repo_files else "MISSING"
                print(
                    f"  [{status}] {remote_path} -> "
                    f"{(output_root / release_name / rel_local).as_posix()}"
                )
        return

    output_root.mkdir(parents=True, exist_ok=True)

    for release_name in releases:
        print(f"\n=== Downloading {release_name} ===")
        release_dir = output_root / release_name
        for remote_path, rel_local in release_files(release_name):
            if remote_path not in repo_files:
                print(f"  [SKIP] Missing on HF: {remote_path}")
                continue

            local_dst = release_dir / rel_local
            local_dst.parent.mkdir(parents=True, exist_ok=True)

            file_url = hf_hub_url(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=remote_path,
            )
            download_file_to_path(file_url, local_dst, args.token)
            print(f"  saved: {local_dst}")

    print("\nDone.")


if __name__ == "__main__":
    main()
