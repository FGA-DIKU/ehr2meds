import json
import pickle
import argparse
from pathlib import Path
import polars as pl

def main(
    test_pts: str | Path,
    train_pts: str | Path,
    mapping_file: str | Path,
    population_file: str | Path,
    output: str | Path,
) -> None:
    test_pts = Path(test_pts)
    train_pts = Path(train_pts)
    population_file = Path(population_file)
    mapping_file = Path(mapping_file)
    output = Path(output)


    with open(test_pts, "r") as f:
        test_ids = json.load(f)
    with open(train_pts, "r") as f:
        train_ids = json.load(f)
    with open(mapping_file, "rb") as f:
        mapping_dict = pickle.load(f)
    population = pl.read_csv(population_file)
    child_to_parent_mapping = population.select(pl.col("CPR_BARN"), pl.col("CPR_MOR")).to_dicts()

    def _map_and_skip(ids: list) -> tuple[list, list]:
        kept: list = []
        skipped: list = []
        for i in ids:
            p_id = child_to_parent_mapping.get(i, None)
            if p_id is not None and p_id in mapping_dict:
                kept.append(mapping_dict[i])
            else:
                skipped.append(i)
        return kept, skipped

    n_test_in = len(test_ids)
    n_train_in = len(train_ids)
    test_ids, skipped_test = _map_and_skip(test_ids)
    train_ids, skipped_train = _map_and_skip(train_ids)

    print(
        "Mapping results:"
        f" test_in={n_test_in}, test_mapped={len(test_ids)}, test_skipped={len(skipped_test)};"
        f" train_in={n_train_in}, train_mapped={len(train_ids)}, train_skipped={len(skipped_train)}"
    )
    if skipped_test:
        print("  examples (test skipped):", [repr(x) for x in skipped_test[:5]])
    if skipped_train:
        print("  examples (train skipped):", [repr(x) for x in skipped_train[:5]])

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
        "--population-file",
        default="population.json",
        help="Path to csv file containing population",
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
        population_file=args.population_file,
        output=args.output,
    )
