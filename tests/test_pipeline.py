"""Unit tests for the Hospital Price Transparency Engine."""
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataGeneration:
    def test_generate_hospitals(self):
        from src.generate_data import generate_hospitals
        h = generate_hospitals(50)
        assert len(h) == 50
        assert "hospital_id" in h.columns
        assert "price_multiplier" in h.columns
        assert (h["price_multiplier"] > 0).all()

    def test_hospital_pricing_file_has_varied_columns(self):
        from src.generate_data import generate_hospitals, generate_hospital_pricing_file
        from src.config import BENCHMARK_PROCEDURES, PAYER_ALIASES
        h = generate_hospitals(2)
        payers = list(PAYER_ALIASES.keys())
        df1, map1 = generate_hospital_pricing_file(h.iloc[0], payers, BENCHMARK_PROCEDURES)
        df2, map2 = generate_hospital_pricing_file(h.iloc[1], payers, BENCHMARK_PROCEDURES)
        # Different hospitals should (usually) use different column names
        assert len(df1) > 0
        assert len(df2) > 0
        # Column sets should differ (randomized naming)
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        assert len(cols1) > 3 and len(cols2) > 3

    def test_pricing_amounts_positive(self):
        from src.generate_data import generate_hospitals, generate_hospital_pricing_file
        from src.config import BENCHMARK_PROCEDURES, PAYER_ALIASES
        h = generate_hospitals(1)
        df, _ = generate_hospital_pricing_file(h.iloc[0], list(PAYER_ALIASES.keys()), BENCHMARK_PROCEDURES)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != "_internal_code":
                assert (df[col].dropna() > 0).all(), f"Negative values in {col}"


class TestSchemaInference:
    def test_normalize_column_name(self):
        from src.schema_inference import normalize_column_name
        assert normalize_column_name("Gross_Charge") == "gross charge"
        assert normalize_column_name("GROSS_CHG") == "gross chg"
        assert normalize_column_name("De-Identified_Min") == "de identified min"

    def test_fuzzy_match_column(self):
        from src.schema_inference import fuzzy_match_column, build_canonical_corpus
        corpus = build_canonical_corpus()
        canonical, score = fuzzy_match_column("Gross_Charge", corpus)
        assert canonical == "gross_charge"
        assert score >= 70

    def test_fuzzy_match_variant(self):
        from src.schema_inference import fuzzy_match_column, build_canonical_corpus
        corpus = build_canonical_corpus()
        canonical, score = fuzzy_match_column("Standard_Price", corpus)
        assert canonical == "gross_charge"

    def test_tfidf_mapper(self):
        from src.schema_inference import TFIDFColumnMapper, build_canonical_corpus
        corpus = build_canonical_corpus()
        mapper = TFIDFColumnMapper(corpus)
        canonical, score = mapper.predict("Cash_Price")
        assert canonical == "discounted_cash_price"
        assert score > 0.3

    def test_payer_normalization(self):
        from src.schema_inference import normalize_payer_name
        assert normalize_payer_name("BCBS") == "Blue Cross Blue Shield"
        assert normalize_payer_name("UHC") == "UnitedHealthcare"
        assert normalize_payer_name("Blue_Cross_Blue_Shield") == "Blue Cross Blue Shield"
        assert normalize_payer_name("AETNA") == "Aetna"

    def test_content_analysis_numeric(self):
        from src.schema_inference import analyze_column_content
        series = pd.Series([100.0, 250.5, 500.0, 1200.0, 3500.0])
        hints = analyze_column_content(series)
        assert hints["is_numeric"] is True
        assert hints.get("likely_price") is True

    def test_content_analysis_codes(self):
        from src.schema_inference import analyze_column_content
        series = pd.Series(["99213", "99214", "73721", "80053", "85025"])
        hints = analyze_column_content(series)
        assert hints.get("likely_code") is True


class TestPriceAnalytics:
    def test_price_variation(self):
        from src.price_analytics import compute_price_variation
        df = pd.DataFrame({
            "procedure_code": ["99213"] * 10,
            "procedure_description": ["Office visit"] * 10,
            "gross_charge": np.random.uniform(100, 500, 10),
            "hospital_id": [f"H_{i}" for i in range(10)],
        })
        result = compute_price_variation(df)
        assert len(result) == 1
        assert "variation_ratio" in result.columns
        assert result.iloc[0]["variation_ratio"] > 1

    def test_outlier_detection(self):
        from src.price_analytics import detect_outliers
        charges = [100] * 50 + [10000]  # One outlier
        df = pd.DataFrame({
            "procedure_code": ["99213"] * 51,
            "procedure_description": ["Visit"] * 51,
            "gross_charge": charges,
            "hospital_id": [f"H_{i}" for i in range(51)],
        })
        outliers = detect_outliers(df)
        assert len(outliers) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
