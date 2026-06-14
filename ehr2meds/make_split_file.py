import json
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
    mapping = pl.read_csv(mapping_file)
    mapping_dict = {
        r["m_cpr"]: r["mapping"]
        for r in mapping.select(["m_cpr", "mapping"]).to_dicts()
    }
    population = pl.read_csv(population_file)
    child_to_parent_mapping = {
        r["b_cpr"]: r["m_cpr"]
        for r in population.select(["b_cpr", "m_cpr"]).to_dicts()
    }

    def _map_and_skip(ids: list) -> tuple[list, list, list]:
        kept_mapped: list = []
        kept_b_cprs: list = []
        skipped: list = []
        for b_cpr in ids:
            p_id = child_to_parent_mapping.get(b_cpr)
            if p_id is not None and p_id in mapping_dict:
                kept_mapped.append(mapping_dict[p_id])
                kept_b_cprs.append(b_cpr)
            else:
                skipped.append(b_cpr)
        return kept_mapped, kept_b_cprs, skipped

    def _check_b_cpr_overlap(splits: dict[str, list]) -> None:
        seen: dict[str, str] = {}
        overlaps: list[tuple[str, str, str]] = []
        for split_name, b_cprs in splits.items():
            for b_cpr in b_cprs:
                if b_cpr in seen:
                    overlaps.append((b_cpr, seen[b_cpr], split_name))
                else:
                    seen[b_cpr] = split_name
        if overlaps:
            examples = ", ".join(
                f"{b_cpr!r} in {a} and {b}" for b_cpr, a, b in overlaps[:5]
            )
            raise ValueError(
                f"Overlapping b_cprs across splits ({len(overlaps)} total). "
                f"Examples: {examples}"
            )

    n_test_in = len(test_ids)
    n_train_in = len(train_ids)

    input_overlap = set(test_ids) & set(train_ids)
    if input_overlap:
        examples = [repr(x) for x in list(input_overlap)[:5]]
        raise ValueError(
            f"test-pts and train-pts share {len(input_overlap)} b_cpr(s). "
            f"Examples: {examples}"
        )

    test_ids, test_b_cprs, skipped_test = _map_and_skip(test_ids)
    mapped_train, train_b_cprs, skipped_train = _map_and_skip(train_ids)

    # test: all mapped IDs from --test-pts; train/val: 80/20 split of --train-pts only
    split_at = int(len(mapped_train) * 0.8)
    train_ids = mapped_train[:split_at]
    val_ids = mapped_train[split_at:]
    train_b_cprs = train_b_cprs[:split_at]
    val_b_cprs = train_b_cprs[split_at:]

    _check_b_cpr_overlap(
        {"test": test_b_cprs, "train": train_b_cprs, "val": val_b_cprs}
    )

    print(
        "Mapping results:"
        f" test_in={n_test_in}, test_mapped={len(test_ids)}, test_skipped={len(skipped_test)};"
        f" train_in={n_train_in}, train_mapped={len(mapped_train)}, train_skipped={len(skipped_train)};"
        f" train_split={len(train_ids)}, val_split={len(val_ids)}"
    )
    if skipped_test:
        print("  examples (test skipped):", [repr(x) for x in skipped_test[:5]])
    if skipped_train:
        print("  examples (train skipped):", [repr(x) for x in skipped_train[:5]])

    output.write_text(json.dumps({"test": test_ids, "train": train_ids, "val": val_ids, "held_out": [], "tuning": []}, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create a split file after applying an ID mapping. "
            "test comes from --test-pts; train and val are an 80/20 split of --train-pts."
        )
    )
    parser.add_argument(
        "--test-pts",
        required=True,
        help="Path to JSON list of test b_cprs (mapped in full; not split further).",
    )
    parser.add_argument(
        "--train-pts",
        required=True,
        help="Path to JSON list of train b_cprs (mapped, then split 80/20 into train and val).",
    )
    parser.add_argument(
        "--mapping-file",
        required=True,
        help="Path to CSV with columns m_cpr and mapping (old_id -> new_id).",
    )
    parser.add_argument(
        "--population-file",
        default="population.csv",
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
