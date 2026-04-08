"""
Price Analytics Engine
Computes cross-hospital price comparisons, geographic variation,
payer-specific patterns, and outlier detection.
"""
import pandas as pd
import numpy as np
import json
import logging
from scipy.spatial.distance import cdist
from src.config import *

logger = logging.getLogger(__name__)


def compute_price_variation(normalized):
    """Compute price variation metrics for each procedure."""
    logger.info("Computing price variation metrics...")

    if "gross_charge" not in normalized.columns or "procedure_code" not in normalized.columns:
        logger.warning("Missing required columns for price variation analysis")
        return pd.DataFrame()

    # Filter to known procedure codes only (exclude junk like rev codes)
    valid_codes = set(BENCHMARK_PROCEDURES.keys())
    filtered = normalized[normalized["procedure_code"].isin(valid_codes)].copy()

    variation = filtered.groupby("procedure_code").agg(
        procedure_name=("procedure_description", "first"),
        n_hospitals=("hospital_id", "nunique"),
        n_records=("gross_charge", "count"),
        mean_charge=("gross_charge", "mean"),
        median_charge=("gross_charge", "median"),
        min_charge=("gross_charge", "min"),
        max_charge=("gross_charge", "max"),
        std_charge=("gross_charge", "std"),
        p10=("gross_charge", lambda x: x.quantile(0.1)),
        p90=("gross_charge", lambda x: x.quantile(0.9)),
    ).reset_index()

    variation["price_range"] = variation["max_charge"] - variation["min_charge"]
    variation["variation_ratio"] = (variation["max_charge"] / variation["min_charge"].replace(0, 1)).round(1)
    variation["cv"] = (variation["std_charge"] / variation["mean_charge"].replace(0, 1)).round(3)
    variation["iqr_ratio"] = ((variation["p90"] - variation["p10"]) / variation["median_charge"].replace(0, 1)).round(3)

    # Round for readability
    for col in ["mean_charge", "median_charge", "min_charge", "max_charge",
                 "std_charge", "p10", "p90", "price_range"]:
        variation[col] = variation[col].round(2)

    variation = variation.sort_values("variation_ratio", ascending=False)
    logger.info(f"  Price variation computed for {len(variation)} procedures")
    return variation


def compute_geographic_analysis(normalized, hospitals):
    """Compute geographic price patterns."""
    logger.info("Computing geographic price analysis...")

    if "hospital_id" not in normalized.columns:
        return pd.DataFrame()

    # Merge hospital location data
    merged = normalized.merge(
        hospitals[["hospital_id", "state", "lat", "lon"]],
        on="hospital_id", how="left"
    )

    # State-level price comparison
    state_prices = merged.groupby(["state", "procedure_code"]).agg(
        procedure_name=("procedure_description", "first"),
        n_hospitals=("hospital_id", "nunique"),
        mean_charge=("gross_charge", "mean"),
        median_charge=("gross_charge", "median"),
        min_charge=("gross_charge", "min"),
        max_charge=("gross_charge", "max"),
    ).reset_index()

    state_prices["variation_ratio"] = (
        state_prices["max_charge"] / state_prices["min_charge"].replace(0, 1)
    ).round(1)

    for col in ["mean_charge", "median_charge", "min_charge", "max_charge"]:
        state_prices[col] = state_prices[col].round(2)

    logger.info(f"  Geographic analysis: {len(state_prices)} state-procedure combinations")
    return state_prices


def compute_payer_analysis(normalized):
    """Compute payer-specific pricing patterns."""
    logger.info("Computing payer analysis...")

    if "payer_name" not in normalized.columns:
        return pd.DataFrame()

    # Average negotiated rates by payer and procedure
    price_col = "min_negotiated_rate" if "min_negotiated_rate" in normalized.columns else "gross_charge"

    payer_prices = normalized.groupby(["payer_name", "procedure_code"]).agg(
        procedure_name=("procedure_description", "first"),
        n_hospitals=("hospital_id", "nunique"),
        mean_rate=(price_col, "mean"),
        median_rate=(price_col, "median"),
    ).reset_index()

    payer_prices["mean_rate"] = payer_prices["mean_rate"].round(2)
    payer_prices["median_rate"] = payer_prices["median_rate"].round(2)

    # Payer ranking per procedure (which payer pays least/most)
    payer_ranks = payer_prices.groupby("procedure_code").apply(
        lambda g: g.assign(payer_rank=g["mean_rate"].rank(method="min")),
        include_groups=False
    ).reset_index(drop=True)

    logger.info(f"  Payer analysis: {payer_prices['payer_name'].nunique()} payers, "
                f"{payer_prices['procedure_code'].nunique()} procedures")
    return payer_prices


def detect_outliers(normalized):
    """Detect hospitals with outlier pricing."""
    logger.info("Detecting pricing outliers...")

    if "gross_charge" not in normalized.columns:
        return pd.DataFrame()

    # Z-score per procedure
    proc_stats = normalized.groupby("procedure_code")["gross_charge"].agg(["mean", "std"]).reset_index()
    proc_stats.columns = ["procedure_code", "proc_mean", "proc_std"]

    merged = normalized.merge(proc_stats, on="procedure_code")
    merged["price_zscore"] = (
        (merged["gross_charge"] - merged["proc_mean"]) / merged["proc_std"].replace(0, 1)
    )

    # Flag outliers (z-score > 2 or < -2)
    outliers = merged[merged["price_zscore"].abs() > 2].copy()
    outliers["outlier_direction"] = np.where(outliers["price_zscore"] > 0, "HIGH", "LOW")
    outliers["percentile"] = merged.groupby("procedure_code")["gross_charge"].rank(pct=True)

    outliers = outliers.sort_values("price_zscore", ascending=False)
    logger.info(f"  Found {len(outliers):,} outlier records ({len(outliers)/len(normalized):.1%} of total)")

    return outliers[["hospital_id", "procedure_code", "procedure_description",
                      "gross_charge", "proc_mean", "price_zscore",
                      "outlier_direction"]].head(500)


def compute_chicago_mri_comparison(normalized, hospitals):
    """Compute the headline finding: MRI price variation within 30 miles of Chicago."""
    logger.info("Computing Chicago MRI comparison (headline finding)...")

    if "procedure_code" not in normalized.columns:
        return {}

    merged = normalized.merge(
        hospitals[["hospital_id", "lat", "lon", "state"]],
        on="hospital_id", how="left"
    )

    # Filter to MRI knee (73721) near Chicago
    mri = merged[merged["procedure_code"] == "73721"].copy()
    mri = mri.dropna(subset=["gross_charge", "lat", "lon"])

    if len(mri) == 0:
        logger.warning("  No MRI data found")
        return {}

    # Calculate distance from Chicago center (approximate)
    mri["distance_miles"] = np.sqrt(
        ((mri["lat"] - CHICAGO_CENTER[0]) * 69) ** 2 +
        ((mri["lon"] - CHICAGO_CENTER[1]) * 54.6) ** 2
    )

    nearby = mri[mri["distance_miles"] <= 100]

    if len(nearby) < 5:
        nearby = mri.nsmallest(min(30, len(mri)), "distance_miles")

    if len(nearby) == 0:
        logger.warning("  No MRI data found after filtering")
        return {}

    radius_used = int(nearby["distance_miles"].max()) + 1

    result = {
        "procedure": "MRI Knee without contrast (CPT 73721)",
        "radius_miles": radius_used,
        "n_hospitals": int(nearby["hospital_id"].nunique()),
        "n_prices": len(nearby),
        "min_price": float(nearby["gross_charge"].min()),
        "max_price": float(nearby["gross_charge"].max()),
        "median_price": float(nearby["gross_charge"].median()),
        "variation_ratio": round(float(nearby["gross_charge"].max() / max(nearby["gross_charge"].min(), 1)), 1),
    }

    logger.info(f"  MRI within {result['radius_miles']} miles of Chicago:")
    logger.info(f"  Min: ${result['min_price']:,.0f} | Max: ${result['max_price']:,.0f} | "
                f"Ratio: {result['variation_ratio']}x")

    return result


def run_analytics():
    """Run full analytics pipeline."""
    logger.info("=" * 60)
    logger.info("PRICE ANALYTICS ENGINE")
    logger.info("=" * 60)

    normalized = pd.read_parquet(DATA_DIR / "normalized_pricing.parquet")
    hospitals = pd.read_parquet(DATA_DIR / "hospitals.parquet")

    # Run analyses
    variation = compute_price_variation(normalized)
    geo = compute_geographic_analysis(normalized, hospitals)
    payer = compute_payer_analysis(normalized)
    outliers = detect_outliers(normalized)
    mri_finding = compute_chicago_mri_comparison(normalized, hospitals)

    # Save
    variation.to_parquet(DATA_DIR / "price_variation.parquet", index=False)
    geo.to_parquet(DATA_DIR / "geographic_prices.parquet", index=False)
    payer.to_parquet(DATA_DIR / "payer_prices.parquet", index=False)
    outliers.to_parquet(DATA_DIR / "outliers.parquet", index=False)
    json.dump(mri_finding, open(MODEL_DIR / "mri_finding.json", "w"), indent=2)

    logger.info(f"\n  Analytics complete. All results saved.")
    return variation, geo, payer, outliers, mri_finding


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_analytics()
