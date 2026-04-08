import pandas as pd
import argparse
import random

def summarise_table(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    df_sample = df.sample(n=n_samples, random_state=42)
    rows = []

    for col in df_sample.columns:
        series = df_sample[col]
        nan_count = int(series.isna().sum())
        nan_pct = float(series.isna().mean() * 100.0)
        
        if pd.api.types.is_bool_dtype(series): # Count positive (True) values
            col_summary = float(series.sum())
            
        elif pd.api.types.is_numeric_dtype(series): # Average for numeric columns
            col_summary = float(series.mean())
            
        else:
            col_summary = series.value_counts(dropna=False).to_dict() # Counts for categorical/object columns

        rows.append(
            {
                "column": col,
                "nan_count": nan_count,
                "nan_pct": nan_pct,
                "summary": col_summary,
            }
        )

    out = pd.DataFrame(rows)
    # Print only the summary view.
    print(out[["column", "summary"]])
    return out


if __name__ == "__main__":
    random.seed(0)

    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets based on a YAML configuration."
    )
    parser.add_argument(
        "--table_path", type=str, help="Path to the table to summarise."
    )
    parser.add_argument(
        "--summary_path", type=str, help="Path to the summary file."
    )
    parser.add_argument(
        "--n_samples", type=int, help="Number of samples to summarise.",
        default=10_000,
    )
    parser.add_argument(
        "--ignore_columns", type=str, help="Columns to ignore.",
        default=["m_cpr", "baby_cpr", "baby_birth_id", "pregnancy_start", "pregnancy_end"],
    )
    args = parser.parse_args()

    table = pd.read_csv(args.table_path)
    table = table.drop(columns=args.ignore_columns)
    summary_df = summarise_table(table, args.n_samples)
    summary_df.to_csv(args.summary_path, index=False)
    print("Saved summary to", args.summary_path)