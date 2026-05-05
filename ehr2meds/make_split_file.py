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
    # Print a small example to confirm direction/format.
    example_items = list(forward.items())[:5]
    print(f"Loaded mapping (after inversion). Example {len(example_items)} items:")
    for k, v in example_items:
        print(f"  {k!r} -> {v!r}")

    def _map_and_skip(ids: list) -> tuple[list, list]:
        kept: list = []
        skipped: list = []
        for i in ids:
            if i in forward:
                kept.append(forward[i])
            else:
                skipped.append(i)
        return kept, skipped

    test_ids, skipped_test = _map_and_skip(test_ids)
    train_ids, skipped_train = _map_and_skip(train_ids)

    if skipped_test or skipped_train:
        print(
            "Skipped unmapped patients:"
            f" test={len(skipped_test)}, train={len(skipped_train)}"
        )
        if skipped_test:
            print("  examples (test):", [repr(x) for x in skipped_test[:5]])
        if skipped_train:
            print("  examples (train):", [repr(x) for x in skipped_train[:5]])

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
