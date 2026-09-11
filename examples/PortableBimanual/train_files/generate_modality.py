#!/usr/bin/env python3
"""Write the VLAct modality metadata for a canonical vla-hub dataset."""

import argparse
import json
from pathlib import Path


def modality(left_camera: str, right_camera: str, language_key: str) -> dict:
    return {
        "state": {
            "bimanual": {
                "start": 0,
                "end": 20,
                "original_key": "observation.state",
                "dtype": "float32",
                "absolute": True,
            }
        },
        "action": {
            "bimanual": {
                "start": 0,
                "end": 20,
                "original_key": "action",
                "dtype": "float32",
                "absolute": True,
            }
        },
        "video": {
            "left_wrist": {"original_key": left_camera},
            "right_wrist": {"original_key": right_camera},
        },
        "annotation": {
            "human.action.task_description": {"original_key": language_key}
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--left-camera", default="observation.images.left_head")
    parser.add_argument("--right-camera", default="observation.images.right_head")
    parser.add_argument("--language-key", default="task_index")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.dataset_root / "meta" / "modality.json"
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(modality(args.left_camera, args.right_camera, args.language_key), indent=2) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
