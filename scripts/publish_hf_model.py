"""Publish the GRPO demo checkpoint to a Hugging Face model repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_MODEL_DIR = Path("outputs/colab_grpo_sme_liquidity/final-demo-model")
DEFAULT_CARD_PATH = Path("huggingface/model_card.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload the SME Negotiator GRPO checkpoint to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Target model repo, for example USER/openenv-sme-negotiator-grpo.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Checkpoint folder to upload.")
    parser.add_argument("--card-path", default=str(DEFAULT_CARD_PATH), help="Markdown model card to upload as README.md.")
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir)
    card_path = Path(args.card_path)

    if not model_dir.exists():
        raise SystemExit(f"Model folder does not exist: {model_dir}")
    if not card_path.exists():
        raise SystemExit(f"Model card does not exist: {card_path}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        ignore_patterns=["*.tmp", "*.log", "checkpoint-*"],
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
    )
    print(f"Published model repo: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
