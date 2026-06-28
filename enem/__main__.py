"""Command-line interface for the ENEM EDA workflow."""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
import tempfile
import traceback
import zipfile
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from .version import __version__

DATA_DIR = "data/"
DEFAULT_DATA_PATH = DATA_DIR + "DADOS/RESULTADOS_2025.csv"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="enem",
        description="Download, extract, and analyze ENEM microdata.",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Local path to RESULTADOS_2025.csv.",
    )
    parser.add_argument(
        "--analysis-output",
        default="analysis_output",
        help="Directory where analysis results (CSV and visualizations) should be written.",
    )
    parser.add_argument(
        "--skip-school-ranking",
        action="store_true",
        help="Skip the school ranking analysis for Recife.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def load_recife_schools(csv_path: str | Path) -> pd.DataFrame:
    """Load RESULTADOS_2025.csv, filter for Recife, PE, and apply a stratified
    sample.

    Args:
        csv_path: Path to the ENEM microdata CSV.
    """
    data_path = Path(csv_path)

    # 1. Detect delimiter
    delimiter = ";"

    # 2. Define projection (Now including CO_ESCOLA for stratification)
    required_columns = [
        "CO_ESCOLA",
        "SG_UF_ESC",
        "NO_MUNICIPIO_ESC",
        "NU_NOTA_CN",
        "NU_NOTA_CH",
        "NU_NOTA_LC",
        "NU_NOTA_MT",
        "NU_NOTA_REDACAO",
    ]

    # 3. Create LazyFrame and Validate
    lf = pl.scan_csv(
        str(data_path),
        separator=delimiter,
        encoding="utf8-lossy",
        ignore_errors=True,
        dtypes={"CO_ESCOLA": pl.String, "CO_MUNICIPIO_ESC": pl.String},
    )
    # 4. Lazy Filter & Select (Predicate Pushdown)
    lazy_query = lf.filter(
        (pl.col("SG_UF_ESC") == "PE")
        & (pl.col("NO_MUNICIPIO_ESC").str.strip_chars().str.to_uppercase() == "RECIFE")
    ).select(required_columns)

    # 5. Collect into memory (Data is now small enough to handle safely)
    df_recife = lazy_query.collect()
    return df_recife.to_pandas()


def load_schools_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load escolas.csv to get school names."""
    schools_path = Path(csv_path)

    # Assuming delimiter is ';' and encoding is 'latin-1' or 'utf-8'
    schools_df = pd.read_csv(
        schools_path, sep=",", encoding="latin-1", dtype={"Codigo INEP": str}
    )

    # Ensure required columns exist and rename them
    required_cols_map = {"Codigo INEP": "CO_ESCOLA", "Escola": "NO_ESCOLA"}

    schools_df = schools_df.rename(columns=required_cols_map)
    return schools_df[["CO_ESCOLA", "NO_ESCOLA"]].copy()


def calculate_overall_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-student overall score as mean of 5 subjects.

    Overall score = mean(CN, CH, LC, MT, Redação)
    Removes rows with missing values in any subject.
    """
    score_columns = [
        "NU_NOTA_CN",
        "NU_NOTA_CH",
        "NU_NOTA_LC",
        "NU_NOTA_MT",
        "NU_NOTA_REDACAO",
    ]

    # Create a copy and remove rows with missing scores
    df = df.copy()
    df = df.dropna(subset=score_columns)

    # Calculate overall score
    df["OVERALL_SCORE"] = df[score_columns].mean(axis=1)

    return df


def compute_school_rankings(df: pd.DataFrame, min_students: int = 10) -> pd.DataFrame:
    """Compute per-school statistics and rankings with 95% confidence
    intervals.

    Uses t-distribution for CI calculation. Schools with overlapping CIs
    are statistically equivalent (draws).

    Returns DataFrame with schools ranked by mean score, including CI
    bounds.
    """
    school_stats = []

    for school_code, group in df.groupby(["CO_ESCOLA", "NO_ESCOLA"]):
        scores = group["OVERALL_SCORE"].values
        n_students = len(scores)

        if n_students < min_students:
            continue

        mean_score = scores.mean()
        std_dev = scores.std(ddof=1)  # Sample std dev
        std_error = std_dev / np.sqrt(n_students)

        # Calculate 95% CI using t-distribution
        df_t = n_students - 1
        t_critical = stats.t.ppf(0.975, df_t)  # 0.975 for two-tailed 95%
        ci_lower = mean_score - t_critical * std_error
        ci_upper = mean_score + t_critical * std_error

        school_stats.append(
            {
                "CO_ESCOLA": school_code[0],
                "NO_ESCOLA": school_code[1],
                "N_STUDENTS": n_students,
                "MEAN_SCORE": mean_score,
                "STD_DEV": std_dev,
                "STD_ERROR": std_error,
                "CI_LOWER": ci_lower,
                "CI_UPPER": ci_upper,
                "CI_WIDTH": ci_upper - ci_lower,
            }
        )

    rankings_df = pd.DataFrame(school_stats)

    if len(rankings_df) == 0:
        return rankings_df

    # Sort by mean score (descending) and add rank
    rankings_df = rankings_df.sort_values("MEAN_SCORE", ascending=False).reset_index(
        drop=True
    )
    rankings_df["RANK"] = range(1, len(rankings_df) + 1)

    # Identify statistically equivalent schools (overlapping CIs)
    rankings_df["HAS_OVERLAP"] = False
    for i in range(len(rankings_df)):
        for j in range(i + 1, len(rankings_df)):
            # Check if CIs overlap
            if (
                rankings_df.loc[i, "CI_LOWER"] <= rankings_df.loc[j, "CI_UPPER"]
                and rankings_df.loc[j, "CI_LOWER"] <= rankings_df.loc[i, "CI_UPPER"]
            ):
                rankings_df.loc[i, "HAS_OVERLAP"] = True
                rankings_df.loc[j, "HAS_OVERLAP"] = True

    return rankings_df


def generate_visualizations(rankings_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Generate visualizations for school rankings.

    Creates:
    1. Forest plot (mean ± 95% CI with ranks and connecting lines)
    2. Bar chart with error bars (mean and CI width)
    3. Statistical summary report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if len(rankings_df) == 0:
        print("No schools to visualize.", file=sys.stderr)
        return

    # Sort by rank for display (best schools at top)
    plot_df = rankings_df.sort_values("RANK", ascending=False).reset_index(drop=True)
    school_labels = [
        f"{row['RANK']}. {row['NO_ESCOLA'][:40]}" for _, row in plot_df.iterrows()
    ]

    # ==========================================
    # 1. Forest Plot
    # ==========================================
    plt.figure(figsize=(12, max(8, len(rankings_df) * 0.3)))
    ax = plt.gca()
    y_pos = np.arange(len(plot_df))

    # Calculate x-axis bounds dynamically to anchor the connecting lines perfectly
    x_min = plot_df["CI_LOWER"].min()
    x_max = plot_df["CI_UPPER"].max()
    padding = (x_max - x_min) * 0.05
    left_edge = x_min - padding
    right_edge = x_max + padding

    # Set explicit limits so the hlines touch the y-axis exactly
    ax.set_xlim(left_edge, right_edge)

    for index, row in plot_df.iterrows():
        mean_score = row["MEAN_SCORE"]
        y_value = y_pos[index]
        lower_error = mean_score - row["CI_LOWER"]
        upper_error = row["CI_UPPER"] - mean_score
        color = "#ff7f0e" if row["HAS_OVERLAP"] else "#1f77b4"

        # NEW: Draw horizontal line connecting the y-axis to the CI lower bound
        plt.hlines(
            y=y_value,
            xmin=left_edge,
            xmax=row["CI_LOWER"],
            color="gray",
            linestyle=":",
            alpha=0.5,
            linewidth=1.2,
            zorder=1,  # Keeps the line behind the data points
        )

        # Plot the data points and error bars
        plt.errorbar(
            [mean_score],
            [y_value],
            xerr=[[lower_error], [upper_error]],
            fmt="o",
            capsize=5,
            capthick=2,
            ecolor=color,
            color=color,
            mfc=color,
            mec=color,
            markersize=8,
            alpha=0.8,
            zorder=2,  # Keeps points in front of the connecting lines
        )

    # Labels and formatting
    plt.yticks(y_pos, school_labels, fontsize=9)
    plt.xlabel("Overall Score (Mean ± 95% CI)", fontsize=11)
    plt.ylabel("Schools (Ranked by Mean Score)", fontsize=11)
    plt.title(
        "Recife School Rankings with Statistical Confidence Intervals\n(Orange = Statistically Equivalent)",
        fontsize=12,
        fontweight="bold",
    )
    plt.grid(axis="x", alpha=0.3)
    # Remove borders for a cleaner look since we have horizontal guide lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path / "forest_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==========================================
    # 2. Bar Chart with error bars
    # ==========================================
    plt.figure(figsize=(12, max(8, len(rankings_df) * 0.25)))
    ax = plt.gca()
    y_pos = np.arange(len(plot_df))

    for index, row in plot_df.iterrows():
        mean_score = row["MEAN_SCORE"]
        lower_error = mean_score - row["CI_LOWER"]
        upper_error = row["CI_UPPER"] - mean_score
        color = "#ff7f0e" if row["HAS_OVERLAP"] else "#1f77b4"

        plt.barh(
            [y_pos[index]],
            [mean_score],
            xerr=[[lower_error], [upper_error]],
            capsize=5,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )

    plt.yticks(y_pos, school_labels, fontsize=9)
    plt.xlabel("Overall Score (Mean ± 95% CI)", fontsize=11)
    plt.ylabel("Schools (Ranked by Mean Score)", fontsize=11)
    plt.title(
        "Recife School Rankings: Mean Scores with Confidence Intervals\n(Width indicates consistency - narrow = consistent)",
        fontsize=12,
        fontweight="bold",
    )
    plt.grid(axis="x", alpha=0.3)
    # NEW: Add a faint y-axis grid to the bar chart to separate the schools visually
    plt.grid(axis="y", linestyle="--", alpha=0.2, color="gray", zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path / "bar_chart.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualizations saved to {output_path}:")
    print(f"  - forest_plot.png")
    print(f"  - bar_chart.png")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    try:

        # School ranking analysis for Recife
        if not args.skip_school_ranking:
            print("Analyzing Recife schools...", file=sys.stderr)
            recife_df = load_recife_schools(args.data_path)
            print(
                f"Loaded {len(recife_df)} student records from Recife schools",
                file=sys.stderr,
            )

            if len(recife_df) > 0:
                recife_df = calculate_overall_scores(recife_df)

                # Load schools metadata and join
                escolas_path = Path(DATA_DIR) / "escolas.csv"
                schools_metadata_df = load_schools_metadata(escolas_path)

                recife_df = recife_df.merge(
                    schools_metadata_df,
                    left_on="CO_ESCOLA",
                    right_on="CO_ESCOLA",
                    how="left",
                )
                # Fill any missing school names if a school in ENEM data is not in escolas.csv
                recife_df["NO_ESCOLA"] = recife_df["NO_ESCOLA"].fillna("Desconhecida")

                rankings_df = compute_school_rankings(recife_df, min_students=10).head(
                    30
                )
                print(
                    f"Ranked {len(rankings_df)} schools (with >=10 students)",
                    file=sys.stderr,
                )

                # Save rankings to CSV
                output_path = Path(args.analysis_output)
                output_path.mkdir(parents=True, exist_ok=True)

                rankings_csv = output_path / "recife_school_rankings.csv"
                rankings_df.to_csv(rankings_csv, index=False)
                print(f"Rankings saved to {rankings_csv}")

                # Generate visualizations
                generate_visualizations(rankings_df, output_path)
            else:
                print("No student records found for Recife schools", file=sys.stderr)
    except Exception as exc:  # pragma: no cover - defensive CLI handling
        traceback.print_exc(file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
