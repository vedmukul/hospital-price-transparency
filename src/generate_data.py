"""
Synthetic Hospital Pricing Data Generator
Simulates 500 hospitals publishing pricing files in wildly inconsistent formats:
  - Different column naming conventions per hospital
  - Mixed payer name formats (BCBS vs Blue_Cross_Blue_Shield vs BC/BS)
  - Variable file structures (some have all fields, some are sparse)
  - Realistic price distributions with geographic and payer variation
"""
import numpy as np
import pandas as pd
import logging
from src.config import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)

HOSPITAL_PREFIXES = [
    "Memorial", "St. Mary's", "Community", "Regional", "University",
    "Mercy", "Providence", "Good Samaritan", "Northwestern", "Rush",
    "Sacred Heart", "General", "Methodist", "Baptist", "Presbyterian",
    "Children's", "Mount Sinai", "Holy Cross", "Valley", "Lakeside",
    "Riverside", "Hillcrest", "Sunrise", "Bayview", "Summit",
]
HOSPITAL_SUFFIXES = [
    "Medical Center", "Hospital", "Health System", "Regional Hospital",
    "Community Hospital", "General Hospital", "Health Center",
]

PLAN_TYPES = ["PPO", "HMO", "EPO", "POS"]


def generate_hospitals(n):
    """Generate hospital metadata."""
    logger.info(f"Generating {n} hospitals...")
    hospitals = []
    for i in range(n):
        state = np.random.choice(STATES, p=[0.15, 0.07, 0.05, 0.06, 0.06, 0.04,
                                              0.03, 0.04, 0.1, 0.12, 0.1, 0.06,
                                              0.05, 0.04, 0.03])
        prefix = np.random.choice(HOSPITAL_PREFIXES)
        suffix = np.random.choice(HOSPITAL_SUFFIXES)
        name = f"{prefix} {suffix}"
        # Add city/state uniqueness
        lat = CHICAGO_CENTER[0] + np.random.normal(0, 4)
        lon = CHICAGO_CENTER[1] + np.random.normal(0, 5)
        npi = f"{np.random.randint(1000000000, 9999999999)}"
        # Price multiplier varies by region (coastal = more expensive)
        if state in ["CA", "NY"]:
            price_multiplier = np.random.uniform(1.3, 2.0)
        elif state in ["IL", "PA", "FL"]:
            price_multiplier = np.random.uniform(1.0, 1.5)
        else:
            price_multiplier = np.random.uniform(0.7, 1.2)
        hospitals.append({
            "hospital_id": f"H_{i:04d}",
            "hospital_name": f"{name} - {state}",
            "npi": npi,
            "state": state,
            "lat": lat,
            "lon": lon,
            "price_multiplier": round(price_multiplier, 2),
            "bed_count": np.random.choice([50, 100, 200, 300, 500, 800]),
        })
    return pd.DataFrame(hospitals)


def pick_random_column_name(canonical_name):
    """Pick a random variant for a canonical column name."""
    variants = COLUMN_VARIANTS.get(canonical_name, [canonical_name])
    return np.random.choice(variants)


def pick_random_payer_name(canonical_payer):
    """Pick a random variant for a canonical payer name."""
    aliases = PAYER_ALIASES.get(canonical_payer, [canonical_payer])
    return np.random.choice(aliases)


def generate_hospital_pricing_file(hospital, payers, procedures):
    """Generate a single hospital's pricing file with randomized column names."""

    # Each hospital picks its own column naming convention
    col_map = {}
    for canonical in ["hospital_name", "procedure_code", "procedure_description",
                      "gross_charge", "discounted_cash_price", "min_negotiated_rate",
                      "max_negotiated_rate", "payer_name", "plan_name"]:
        col_map[canonical] = pick_random_column_name(canonical)

    rows = []
    proc_codes = list(procedures.keys())
    # Each hospital has a random subset of procedures
    n_procs = np.random.randint(max(5, len(proc_codes) // 3), len(proc_codes))
    selected_procs = np.random.choice(proc_codes, n_procs, replace=False)

    # Each hospital works with a random subset of payers
    n_payers = np.random.randint(3, len(payers) + 1)
    selected_payers = np.random.choice(payers, n_payers, replace=False)

    for proc_code in selected_procs:
        proc_name, low, high = procedures[proc_code]
        base_price = np.random.uniform(low, high) * hospital["price_multiplier"]

        # Gross charge
        gross = round(base_price * np.random.uniform(1.0, 1.5), 2)
        # Cash discount (typically 30-60% off gross)
        cash = round(gross * np.random.uniform(0.4, 0.7), 2)

        for payer in selected_payers:
            # Negotiated rates vary by payer
            payer_discount = np.random.uniform(0.3, 0.8)
            neg_rate = round(gross * payer_discount, 2)
            min_neg = round(neg_rate * np.random.uniform(0.85, 0.95), 2)
            max_neg = round(neg_rate * np.random.uniform(1.05, 1.20), 2)
            plan_type = np.random.choice(PLAN_TYPES)

            row = {
                col_map["hospital_name"]: hospital["hospital_name"],
                col_map["procedure_code"]: proc_code,
                col_map["procedure_description"]: proc_name,
                col_map["gross_charge"]: gross,
                col_map["discounted_cash_price"]: cash,
                col_map["min_negotiated_rate"]: min_neg,
                col_map["max_negotiated_rate"]: max_neg,
                col_map["payer_name"]: pick_random_payer_name(payer),
                col_map["plan_name"]: f"{pick_random_payer_name(payer)} {plan_type}",
            }

            # Randomly drop some columns (simulating sparse files)
            if np.random.random() < 0.15:
                drop_col = np.random.choice([col_map["discounted_cash_price"],
                                              col_map["min_negotiated_rate"]])
                row.pop(drop_col, None)

            rows.append(row)

    df = pd.DataFrame(rows)

    # Randomly add junk columns (some hospitals include extra fields)
    if np.random.random() < 0.3:
        df["_internal_code"] = np.random.randint(1000, 9999, len(df))
    if np.random.random() < 0.2:
        df["last_updated"] = "2024-01-01"
    if np.random.random() < 0.15:
        df["rev_code"] = np.random.choice(["0250", "0260", "0270", "0300"], len(df))

    return df, col_map


def generate_all():
    """Generate pricing files for all hospitals."""
    logger.info("=" * 60)
    logger.info("GENERATING SYNTHETIC HOSPITAL PRICING DATA")
    logger.info("=" * 60)

    hospitals = generate_hospitals(N_HOSPITALS)
    payers = list(PAYER_ALIASES.keys())
    procedures = BENCHMARK_PROCEDURES

    all_files = []
    all_column_maps = []

    for idx, hospital in hospitals.iterrows():
        df, col_map = generate_hospital_pricing_file(hospital, payers, procedures)
        df["_hospital_id"] = hospital["hospital_id"]
        all_files.append(df)
        all_column_maps.append({
            "hospital_id": hospital["hospital_id"],
            "hospital_name": hospital["hospital_name"],
            "column_map": col_map,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
        })

    # Combine all into one "raw" dataset (simulating what you'd get after crawling)
    raw_combined = pd.concat(all_files, ignore_index=True)

    logger.info(f"  Generated {len(hospitals)} hospital files")
    logger.info(f"  Total raw rows: {len(raw_combined):,}")
    logger.info(f"  Unique column names across all hospitals: {len(set(c for f in all_files for c in f.columns))}")
    logger.info(f"  Average rows per hospital: {len(raw_combined) // len(hospitals)}")

    # Save
    hospitals.to_parquet(DATA_DIR / "hospitals.parquet", index=False)
    raw_combined.to_parquet(DATA_DIR / "raw_pricing_files.parquet", index=False)
    pd.DataFrame(all_column_maps).to_parquet(DATA_DIR / "column_maps_ground_truth.parquet", index=False)

    logger.info(f"  Data saved to {DATA_DIR}")
    return hospitals, raw_combined, all_column_maps


if __name__ == "__main__":
    generate_all()
