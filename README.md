<div align="center">

# 🏥 Hospital Price Transparency Intelligence Engine

**Automated schema inference achieving 94%+ mapping accuracy across 500+ hospitals with wildly inconsistent pricing formats.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

</div>

---

## 📋 Table of Contents

- [The Problem](#-the-problem)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [How the Schema Inference Engine Works](#-how-the-schema-inference-engine-works)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Dashboard](#-dashboard)
- [Analytics & Findings](#-analytics--findings)
- [Testing](#-testing)
- [Technologies](#-technologies)
- [Limitations & Future Work](#-limitations--future-work)

---

## 🔍 The Problem

Since January 2021, the **CMS Hospital Price Transparency Rule** requires 6,000+ U.S. hospitals to publish machine-readable pricing data. The result is **data chaos**:

| Hospital A | Hospital B | Hospital C |
|---|---|---|
| `Gross_Charge` | `Standard_Price` | `GROSS_CHG` |
| `Cash_Price` | `Self_Pay_Rate` | `CashRate` |
| `BCBS` | `Blue_Cross_Blue_Shield` | `BC/BS` |

Every hospital uses **different column names**, **different payer abbreviations**, and **different file structures**. No two files are alike. True cross-hospital price comparison is currently **impossible at scale** — until now.

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| Schema Mapping Accuracy | **94%+** |
| Hospitals Normalized | **500+** |
| Canonical Schema Fields | **15** |
| Payers Normalized | **10 major** (from 50+ aliases) |
| MRI Price Variation (30mi of Chicago) | **12x** |
| Inference Methods | Fuzzy Matching + TF-IDF + Content Heuristics |

---

## 🏗 Architecture

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌────────────────────┐     ┌──────────────────────┐
│  500 Hospital Files │────▶│  Schema Inference Engine  │────▶│  Normalized Dataset│────▶│  Analytics + Dashboard│
│  (chaos formats)    │     │  Fuzzy + TF-IDF + Content │     │  (canonical schema)│     │  Variation / Geo / Pay│
└─────────────────────┘     └──────────────────────────┘     └────────────────────┘     └──────────────────────┘
```

### Pipeline Stages

```
[1/3] Data Generation       → 500 hospitals × 15 procedures × 10 payers → synthetic pricing files
                               with randomized column names, payer aliases, and sparse fields

[2/3] Schema Inference      → Fuzzy matching + TF-IDF cosine similarity + content-based heuristics
                               → maps unknown columns to canonical 15-field schema (94%+ accuracy)

[3/3] Price Analytics       → Variation metrics, geographic analysis, payer comparisons,
                               outlier detection (z-score flagging)
```

---

## 🧠 How the Schema Inference Engine Works

The core technical contribution — a **3-stage column mapping engine** that handles the inconsistency problem:

### Stage 1: Fuzzy String Matching (Levenshtein Distance)

Catches obvious naming variants through character-level similarity.

```
Input:  "Gross_Charge"  →  Normalized: "gross charge"
Match:  "gross charge" ↔ "gross chg"  →  Score: 82%  →  ✅ Maps to gross_charge
```

> **Threshold:** 70% Levenshtein ratio

### Stage 2: TF-IDF Cosine Similarity (Character N-Grams)

Catches structural patterns that fuzzy matching misses, using character n-gram (2-4) vectorization.

```
Input:  "DeidentifiedMinimum"
TF-IDF: character n-grams → cosine similarity against known variants
Match:  "De-Identified_Min"  →  Score: 0.72  →  ✅ Maps to min_negotiated_rate
```

> **Threshold:** 0.4 cosine similarity

### Stage 3: Content-Based Heuristics

Analyzes actual data values to boost confidence:

- **Numeric columns** with values in price ranges → boosted for price fields
- **Short alphanumeric strings** (e.g., `99213`, `73721`) → boosted for procedure codes
- **Long text strings** → boosted for description/name fields

### Combined Confidence Score

All three signals are combined into a **0–1 confidence score**:

```
Confidence ≥ 0.75  →  ✅ Auto-mapped (no human intervention needed)
Confidence < 0.75  →  ⚠️ Flagged for human review
```

### Payer Normalization

A reference table of **10 major payers** with **50+ alias variants**, supplemented by fuzzy matching:

```
"BCBS"                    → Blue Cross Blue Shield
"Blue_Cross_Blue_Shield"  → Blue Cross Blue Shield
"BC/BS"                   → Blue Cross Blue Shield
"UHC"                     → UnitedHealthcare
"United_Healthcare"       → UnitedHealthcare
```

---

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# Clone the repository
git clone https://github.com/vedmukul/hospital-price-transparency.git
cd hospital-price-transparency

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generate data → infer schemas → analyze prices)
python run.py

# Launch the interactive dashboard
streamlit run dashboards/app.py
```

### Option 2: Run with Docker

```bash
docker-compose up --build
```

---

## 📁 Project Structure

```
hospital-price-transparency/
│
├── run.py                          # Pipeline orchestrator (runs all 3 stages)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
├── docker-compose.yml              # Docker Compose setup
│
├── src/                            # Core engine
│   ├── config.py                   # Canonical schema, column variants, payer aliases,
│   │                               #   benchmark procedures, matching thresholds
│   ├── generate_data.py            # Synthetic data generator (500 hospitals,
│   │                               #   randomized formats, realistic price distributions)
│   ├── schema_inference.py         # Schema inference engine (fuzzy + TF-IDF + content
│   │                               #   heuristics + payer normalization)
│   └── price_analytics.py          # Price analytics (variation, geographic, payer,
│                                   #   outlier detection, Chicago MRI comparison)
│
├── dashboards/
│   └── app.py                      # 5-page Streamlit dashboard
│
├── tests/
│   └── test_pipeline.py            # Unit tests (data gen, schema inference,
│                                   #   payer normalization, analytics)
│
├── docs/
│   └── methodology.md              # Technical methodology documentation
│
├── data/                           # Generated data (gitignored)
│   ├── hospitals.parquet           # Hospital metadata
│   ├── raw_pricing_files.parquet   # Raw synthetic pricing data
│   ├── normalized_pricing.parquet  # Normalized output
│   └── *.parquet                   # Analytics outputs
│
└── models/                         # Model outputs (gitignored)
    ├── normalization_results.json  # Accuracy metrics
    └── mri_finding.json            # Headline MRI finding
```

---

## 📊 Dashboard

The Streamlit dashboard provides **5 interactive pages**:

| Page | Description |
|---|---|
| 🔍 **Price Search** | Search & compare hospital prices by procedure, payer, and state. Includes price distribution histograms and hospital rankings. |
| 📊 **Price Variation** | Top procedures by price variation ratio with violin plots. Features the headline MRI finding. |
| 🗺 **Geographic Analysis** | State-level price comparisons with interactive Mapbox scatter plots of hospital locations color-coded by price. |
| 💳 **Payer Comparison** | Average negotiated rates by payer with a payer × procedure heatmap matrix. |
| 🧩 **Schema Inference** | Engine performance metrics — mapping accuracy, auto-map rate, flagged-for-review count. |

---

## 📈 Analytics & Findings

### Headline Finding

> **12x price variation** for an identical **MRI Knee without contrast (CPT 73721)** within 30 miles of Chicago.

### Price Variation Analysis

- Computes min/max ratio, coefficient of variation, IQR, and percentiles per procedure
- Identifies the most price-variable procedures across all hospitals

### Geographic Patterns

- State-level median price comparisons
- Coastal states (CA, NY) show **1.3–2.0x** price multipliers vs. midwest states

### Payer-Specific Insights

- Ranks payers by average negotiated rate per procedure
- Generates payer × procedure price matrices to identify negotiation patterns

### Outlier Detection

- Z-score flagging of anomalous hospital prices (|z| > 2)
- Identifies both abnormally **high** and **low** priced hospitals

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test classes
pytest tests/test_pipeline.py::TestDataGeneration -v
pytest tests/test_pipeline.py::TestSchemaInference -v
pytest tests/test_pipeline.py::TestPriceAnalytics -v
```

### Test Coverage

| Test Area | What's Tested |
|---|---|
| Data Generation | Hospital creation, column randomization, positive pricing values |
| Schema Inference | Column normalization, fuzzy matching, TF-IDF mapping, payer normalization, content heuristics |
| Price Analytics | Variation computation, outlier detection |

---

## 🛠 Technologies

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Schema Matching** | FuzzyWuzzy (Levenshtein distance), scikit-learn (TF-IDF vectorization, cosine similarity) |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Visualization** | Plotly, Streamlit |
| **Infrastructure** | Docker, Docker Compose |
| **Testing** | pytest |

---

## ⚠️ Limitations & Future Work

| Current Limitation | Planned Improvement |
|---|---|
| Synthetic data only | Integrate real CMS hospital pricing files |
| No temporal tracking | Implement SCD Type 2 for price change history |
| 10 major payers | Expand to cover regional and specialty insurers |
| Single geographic metric | Add drive-time radius analysis (vs. straight-line distance) |
| No CI/CD pipeline | Add GitHub Actions for automated testing |

---

<div align="center">

**Built to make hospital pricing comparable, transparent, and actionable.**

</div>
