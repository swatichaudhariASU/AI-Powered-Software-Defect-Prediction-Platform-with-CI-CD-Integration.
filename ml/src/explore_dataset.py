"""Basic data-quality report for the Month 1 dataset."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "preliminary_dataset.csv"


def main():
    df = pd.read_csv(INPUT)

    print("\nShape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    if "candidate_defect" in df.columns:
        print("\nCandidate label distribution:")
        print(df["candidate_defect"].value_counts(dropna=False))

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        print("\nNumeric summary:")
        print(numeric.describe().T)


if __name__ == "__main__":
    main()
