"""
Pipeline Orchestrator: Hospital Price Transparency Engine
  1. Generate synthetic hospital pricing data (500 hospitals, inconsistent formats)
  2. Run schema inference & normalization (fuzzy matching + TF-IDF)
  3. Run price analytics (variation, geographic, payer, outliers)
"""
import logging
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log")],
)
logger = logging.getLogger(__name__)


def main():
    start = time.time()
    logger.info("=" * 70)
    logger.info("  HOSPITAL PRICE TRANSPARENCY INTELLIGENCE ENGINE")
    logger.info("  Schema Inference | Price Analytics | Geographic Comparison")
    logger.info("=" * 70)

    logger.info("\n[1/3] GENERATING SYNTHETIC HOSPITAL PRICING DATA...")
    from src.generate_data import generate_all
    generate_all()

    logger.info("\n[2/3] RUNNING SCHEMA INFERENCE & NORMALIZATION...")
    from src.schema_inference import run_normalization_pipeline
    normalized, norm_results = run_normalization_pipeline()

    logger.info("\n[3/3] RUNNING PRICE ANALYTICS...")
    from src.price_analytics import run_analytics
    variation, geo, payer, outliers, mri_finding = run_analytics()

    elapsed = time.time() - start
    logger.info("\n" + "=" * 70)
    logger.info(f"  PIPELINE COMPLETE in {elapsed:.1f} seconds")
    logger.info(f"  Schema mapping accuracy: {norm_results['accuracy']:.1%}")
    logger.info(f"  Total normalized rows: {norm_results['total_normalized_rows']:,}")
    if mri_finding:
        logger.info(f"  MRI price variation (30mi Chicago): {mri_finding['variation_ratio']}x")
    logger.info("=" * 70)
    logger.info("\n  Launch dashboard: streamlit run dashboards/app.py")


if __name__ == "__main__":
    main()
