import pandas as pd
import argparse
import random

def summarise_table(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    df_sample = df.sample(n=n_samples, random_state=42)
    summary = {}

    for col in df_sample.columns:
        series = df_sample[col]
        
        if pd.api.types.is_bool_dtype(series): # Count positive (True) values
            summary[col] = series.sum()
            
        elif pd.api.types.is_numeric_dtype(series): # Average for numeric columns
            summary[col] = series.mean()
            
        else:
            summary[col] = series.value_counts(dropna=False).to_dict() # Counts for categorical/object columns

    print(summary)
    return summary


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
    args = parser.parse_args()

    table = pd.read_csv(args.table_path)
    summary = summarise_table(table, args.n_samples)
    summary.to_csv(args.summary_path, index=False)
    print("Saved summary to", args.summary_path)