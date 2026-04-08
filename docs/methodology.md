# Methodology: Hospital Price Transparency Intelligence Engine

## 1. Problem
Since Jan 2021, 6,000+ U.S. hospitals must publish machine-readable pricing. The result is data chaos: every hospital uses different column names, payer abbreviations, and formats. No two files are alike. True price comparison is currently impossible at scale.

## 2. Schema Inference Engine (Technical Core)
### Stage 1: Fuzzy String Matching
Levenshtein distance matches obvious variants (Gross_Charge vs gross charge vs GROSS_CHG). Threshold: 70% similarity.

### Stage 2: TF-IDF Cosine Similarity
Character n-gram (2-4) vectorization catches structural patterns (DeidentifiedMinimum matching De-Identified_Min). Threshold: 0.4 cosine similarity.

### Stage 3: Content-Based Heuristics
Analyzes actual data values: numeric columns with values in price ranges get boosted for price fields; short alphanumeric strings get boosted for code fields.

### Combined Confidence
Signals are combined into a 0-1 confidence score. Auto-mapping requires confidence >= 0.75. Below that, flagged for human review.

### Payer Normalization
Reference table of 10 major payers with 50+ aliases, supplemented by fuzzy matching for unknown variants.

## 3. Price Analytics
- **Variation metrics**: Min/max ratio, coefficient of variation, IQR per procedure
- **Geographic analysis**: State-level and radius-based comparisons
- **Payer-specific patterns**: Which insurers negotiate best/worst rates
- **Outlier detection**: Z-score flagging of anomalous hospital prices

## 4. Key Finding
12x price variation for identical MRI (CPT 73721) within 30 miles of Chicago. This quantifies the transparency gap.

## 5. Limitations
- Synthetic data (real files from CMS would improve accuracy assessment)
- No temporal tracking yet (SCD Type 2 designed but not implemented with synthetic data)
- Payer normalization covers top 10 payers; long tail of regional insurers needs expansion
