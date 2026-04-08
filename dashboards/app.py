"""
Hospital Price Transparency Dashboard
5 pages: Price Search, Price Variation, Geographic Heatmap, Payer Comparison, Schema Inference.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Hospital Price Transparency", page_icon="\U0001F3E5", layout="wide")


@st.cache_data
def load_data():
    d = {}
    for name in ["normalized_pricing", "hospitals", "price_variation",
                  "geographic_prices", "payer_prices", "outliers"]:
        p = DATA_DIR / f"{name}.parquet"
        d[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    for name in ["normalization_results", "mri_finding"]:
        p = MODEL_DIR / f"{name}.json"
        d[name] = json.load(open(p)) if p.exists() else {}
    return d


data = load_data()

st.sidebar.title("\U0001F3E5 Price Transparency")
page = st.sidebar.radio("Navigate", [
    "\U0001F50D Price Search",
    "\U0001F4CA Price Variation",
    "\U0001F5FA Geographic Analysis",
    "\U0001F4B3 Payer Comparison",
    "\U0001F9E9 Schema Inference",
])

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PRICE SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
if page == "\U0001F50D Price Search":
    st.title("\U0001F50D Search & Compare Hospital Prices")

    normalized = data["normalized_pricing"]
    hospitals = data["hospitals"]

    if len(normalized) == 0:
        st.error("No data loaded. Run the pipeline first: `python run.py`")
        st.stop()

    # Search controls
    col1, col2, col3 = st.columns(3)
    procedures = normalized["procedure_code"].dropna().unique()
    proc_names = normalized.groupby("procedure_code")["procedure_description"].first().to_dict()
    proc_options = {f"{k} - {v}": k for k, v in proc_names.items() if pd.notna(v)}

    with col1:
        selected_proc_display = st.selectbox("Procedure", list(proc_options.keys()))
        selected_proc = proc_options.get(selected_proc_display, "")
    with col2:
        payers = ["All"] + sorted(normalized["payer_name"].dropna().unique().tolist())
        selected_payer = st.selectbox("Insurance", payers)
    with col3:
        states = ["All"] + sorted(hospitals["state"].unique().tolist())
        selected_state = st.selectbox("State", states)

    # Filter
    filtered = normalized[normalized["procedure_code"] == selected_proc]
    if selected_payer != "All":
        filtered = filtered[filtered["payer_name"] == selected_payer]
    if selected_state != "All":
        state_hospitals = hospitals[hospitals["state"] == selected_state]["hospital_id"]
        filtered = filtered[filtered["hospital_id"].isin(state_hospitals)]

    if len(filtered) == 0:
        st.warning("No results found for this combination.")
        st.stop()

    # Price summary
    st.subheader(f"Results: {len(filtered):,} prices from {filtered['hospital_id'].nunique()} hospitals")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lowest", f"${filtered['gross_charge'].min():,.0f}")
    c2.metric("Median", f"${filtered['gross_charge'].median():,.0f}")
    c3.metric("Highest", f"${filtered['gross_charge'].max():,.0f}")
    ratio = filtered['gross_charge'].max() / max(filtered['gross_charge'].min(), 1)
    c4.metric("Price Ratio", f"{ratio:.1f}x")

    # Distribution
    fig = px.histogram(filtered, x="gross_charge", nbins=40, color_discrete_sequence=["#0F3A5F"],
                       labels={"gross_charge": "Gross Charge ($)"},
                       title="Price Distribution")
    fig.add_vline(x=filtered["gross_charge"].median(), line_dash="dash", line_color="red",
                  annotation_text=f"Median: ${filtered['gross_charge'].median():,.0f}")
    st.plotly_chart(fig, use_container_width=True)

    # Ranked table
    st.subheader("Hospital Rankings (Lowest to Highest)")
    ranked = filtered.merge(hospitals[["hospital_id", "hospital_name", "state"]],
                            on="hospital_id", how="left")
    ranked["percentile"] = ranked["gross_charge"].rank(pct=True).round(2)
    display_cols = ["hospital_name", "state", "gross_charge", "payer_name", "percentile"]
    available = [c for c in display_cols if c in ranked.columns]
    st.dataframe(ranked[available].sort_values("gross_charge").head(50),
                 use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PRICE VARIATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "\U0001F4CA Price Variation":
    st.title("\U0001F4CA Price Variation by Procedure")

    # MRI headline finding
    mri = data["mri_finding"]
    if mri:
        st.error(f"\U0001F4A1 **Key Finding:** {mri['procedure']} — "
                 f"**{mri['variation_ratio']}x price variation** within {mri['radius_miles']} miles "
                 f"of Chicago (${mri['min_price']:,.0f} to ${mri['max_price']:,.0f})")
    st.divider()

    variation = data["price_variation"]
    if len(variation) == 0:
        st.warning("No variation data. Run pipeline first.")
        st.stop()

    # Variation chart
    fig = px.bar(variation.head(15), x="variation_ratio", y="procedure_code",
                 orientation="h", color="variation_ratio",
                 color_continuous_scale="Reds",
                 hover_data=["procedure_name", "min_charge", "max_charge", "n_hospitals"],
                 labels={"variation_ratio": "Max/Min Price Ratio", "procedure_code": "Procedure"},
                 title="Top 15 Procedures by Price Variation")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Violin plots for top procedures
    st.subheader("Price Distribution by Procedure")
    normalized = data["normalized_pricing"]
    top_procs = variation.head(6)["procedure_code"].tolist()
    violin_data = normalized[normalized["procedure_code"].isin(top_procs)]
    if len(violin_data) > 0:
        fig = px.violin(violin_data, x="procedure_code", y="gross_charge",
                        box=True, points=False, color="procedure_code",
                        labels={"gross_charge": "Gross Charge ($)", "procedure_code": "Procedure"})
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Full table
    st.subheader("Complete Variation Data")
    st.dataframe(variation, use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: GEOGRAPHIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "\U0001F5FA Geographic Analysis":
    st.title("\U0001F5FA Geographic Price Patterns")

    geo = data["geographic_prices"]
    hospitals = data["hospitals"]
    normalized = data["normalized_pricing"]

    if len(geo) == 0:
        st.warning("No geographic data.")
        st.stop()

    # State comparison for selected procedure
    proc_options = geo["procedure_code"].unique()
    selected = st.selectbox("Select Procedure",
                            [f"{p} - {geo[geo['procedure_code']==p]['procedure_name'].iloc[0]}"
                             for p in proc_options])
    proc_code = selected.split(" - ")[0]

    state_data = geo[geo["procedure_code"] == proc_code]

    fig = px.bar(state_data.sort_values("median_charge"), x="state", y="median_charge",
                 color="median_charge", color_continuous_scale="RdYlGn_r",
                 error_y=state_data["max_charge"] - state_data["median_charge"],
                 labels={"median_charge": "Median Charge ($)", "state": "State"},
                 title=f"Median Price by State: {selected}")
    st.plotly_chart(fig, use_container_width=True)

    # Hospital map
    st.subheader("Hospital Locations")
    merged = normalized[normalized["procedure_code"] == proc_code].merge(
        hospitals[["hospital_id", "hospital_name", "lat", "lon", "state"]], on="hospital_id"
    )
    if len(merged) > 0:
        avg_prices = merged.groupby("hospital_id").agg(
            lat=("lat", "first"), lon=("lon", "first"),
            hospital_name=("hospital_name", "first"),
            avg_charge=("gross_charge", "mean"),
        ).reset_index()
        fig = px.scatter_mapbox(avg_prices, lat="lat", lon="lon",
                                color="avg_charge", size="avg_charge",
                                hover_name="hospital_name",
                                color_continuous_scale="RdYlGn_r",
                                mapbox_style="carto-positron",
                                zoom=4, height=500,
                                title=f"Hospital Prices: {selected}")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PAYER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "\U0001F4B3 Payer Comparison":
    st.title("\U0001F4B3 Payer Price Comparison")

    payer = data["payer_prices"]
    if len(payer) == 0:
        st.warning("No payer data.")
        st.stop()

    # Payer average rates
    st.subheader("Average Negotiated Rates by Payer")
    payer_avg = payer.groupby("payer_name")["mean_rate"].mean().sort_values()
    fig = px.bar(payer_avg.reset_index(), x="payer_name", y="mean_rate",
                 color="mean_rate", color_continuous_scale="RdYlGn_r",
                 labels={"mean_rate": "Average Rate ($)", "payer_name": "Payer"})
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap: payer × procedure
    st.subheader("Payer × Procedure Price Matrix")
    pivot = payer.pivot_table(index="payer_name", columns="procedure_code",
                               values="mean_rate", aggfunc="mean")
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                    labels={"color": "Avg Rate ($)"},
                    aspect="auto", height=400)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: SCHEMA INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "\U0001F9E9 Schema Inference":
    st.title("\U0001F9E9 Schema Inference Engine Performance")

    results = data["normalization_results"]
    if not results:
        st.warning("No normalization results.")
        st.stop()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mapping Accuracy", f"{results.get('accuracy', 0):.1%}")
    c2.metric("Auto-Mapped Rate", f"{results.get('auto_map_rate', 0):.1%}")
    c3.metric("Flagged for Review", results.get("flagged_for_review", 0))
    c4.metric("Total Normalized Rows", f"{results.get('total_normalized_rows', 0):,}")

    st.divider()

    st.markdown("""
    ### How It Works

    The schema inference engine uses a **3-stage approach** to map unknown column names
    from each hospital's unique format to a canonical schema:

    1. **Fuzzy String Matching** (Levenshtein distance): Catches obvious variants like
       `Gross_Charge` vs `gross charge` vs `GROSS_CHG`
    2. **TF-IDF Cosine Similarity** (character n-grams): Catches structural patterns like
       `DeidentifiedMinimum` matching `De-Identified_Min`
    3. **Content-Based Heuristics**: Analyzes actual data values — numeric columns with values
       in price ranges get boosted confidence for price-related canonical fields

    The combined confidence score must exceed **0.75** for automatic mapping.
    Below that threshold, the column is **flagged for human review**.
    """)

    st.markdown("""
    ### Why This Matters

    The CMS Hospital Price Transparency rule requires 6,000+ hospitals to publish
    machine-readable pricing data. But there's **no standardized format**. Each hospital
    uses different column names, payer abbreviations, and file structures. This engine
    solves the normalization problem at scale, making true cross-hospital price comparison possible.
    """)
