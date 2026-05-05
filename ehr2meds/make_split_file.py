import json
import pickle
import argparse
from pathlib import Path


def main(test_pts: str | Path, train_pts: str | Path, mapping_file: str | Path, output: str | Path) -> None:
    test_pts = Path(test_pts)
    train_pts = Path(train_pts)
    mapping_file = Path(mapping_file)
    output = Path(output)

    test_ids = json.loads(test_pts.read_text())
    train_ids = json.loads(train_pts.read_text())
    with mapping_file.open("rb") as f:
        mapping_dict = pickle.load(f)

    # Use the inverted mapping direction.
    forward = {v: k for k, v in mapping_dict.items()}

    test_ids = [forward[i] for i in test_ids]
    train_ids = [forward[i] for i in train_ids]
    output.write_text(json.dumps({"test": test_ids, "train": train_ids}, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a split file (train/test) after applying an ID mapping."
    )
    parser.add_argument(
        "--test-pts",
        required=True,
        help="Path to JSON list of test subject IDs (pre-mapping).",
    )
    parser.add_argument(
        "--train-pts",
        required=True,
        help="Path to JSON list of train subject IDs (pre-mapping).",
    )
    parser.add_argument(
        "--mapping-file",
        required=True,
        help="Path to pickle file containing mapping dict (old_id -> new_id).",
    )
    parser.add_argument(
        "--output",
        default="split_file.json",
        help="Output JSON path (default: split_file.json).",
    )
    args = parser.parse_args()
    main(
        test_pts=args.test_pts,
        train_pts=args.train_pts,
        mapping_file=args.mapping_file,
        output=args.output,
    )
