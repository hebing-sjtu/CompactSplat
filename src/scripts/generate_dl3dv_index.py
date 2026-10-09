import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index prepared dataset chunks.")
    parser.add_argument("--dataset_path", type=Path, required=True)
    args = parser.parse_args()
    DATASET_PATH = args.dataset_path
    # "train" or "test"
    for stage in ["test"]:
        stage = DATASET_PATH / stage

        index = {}
        for chunk_path in tqdm(
            sorted(list(stage.iterdir())), desc=f"Indexing {stage.name}"
        ):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path, weights_only=False)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage))
        with (stage / "index.json").open("w") as f:
            json.dump(index, f)
