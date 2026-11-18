"""Premium King County real-estate valuation platform."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from feature_engineering import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    load_bundle,
    load_raw_dataset,
    prepare_inference_frame,
    prepare_training_frame,
    validate_prediction_payload,
)

NAVY = "#071427"
NAVY_CARD = "#0E1C33"
GOLD = "#D4AF37"
BLUE = "#4CC9F0"
WHITE = "#F7F4EC"
MUTED = "#9AA7BD"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8, 18, 36, 0.65)",
    font=dict(color=WHITE, family="Georgia, serif"),
    colorway=[GOLD, BLUE, "#7EE0A6", "#F4A261", "#C77DFF"],
    margin=dict(l=40, r=20, t=50, b=40),
)


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(1200px 500px at 10% -10%, rgba(76, 201, 240, 0.12), transparent 45%),
                radial-gradient(900px 400px at 100% 0%, rgba(212, 175, 55, 0.10), transparent 40%),
                linear-gradient(180deg, {NAVY} 0%, #050d1a 100%);
            color: {WHITE};
        }}
        .block-container {{ padding-top: 1.4rem; max-width: 1280px; }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0a1730 0%, #07101f 100%);
            border-right: 1px solid rgba(212, 175, 55, 0.18);
        }}
        h1, h2, h3, h4 {{ font-family: Georgia, serif; letter-spacing: 0.02em; color: {WHITE}; }}
        .hero-kicker {{
            color: {GOLD};
            font-size: 0.78rem;
            letter-spacing: 0.28em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }}
        .hero-title {{
            font-size: 2.15rem;
            line-height: 1.15;
            margin: 0 0 0.4rem 0;
        }}
        .hero-sub {{ color: {MUTED}; font-size: 1.02rem; max-width: 720px; }}
        .metric-card, .panel-card, .pred-card {{
            background: linear-gradient(180deg, rgba(18, 34, 61, 0.92), rgba(10, 20, 38, 0.92));
            border: 1px solid rgba(212, 175, 55, 0.16);
            border-radius: 18px;
            padding: 1.05rem 1.15rem;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.28);
        }}
        .metric-label {{ color: {MUTED}; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }}
        .metric-value {{ color: {WHITE}; font-size: 1.45rem; font-weight: 600; margin-top: 0.25rem; }}
        .metric-value.gold {{ color: {GOLD}; }}
        .pred-price {{
            font-size: 2.6rem;
            color: {GOLD};
            font-weight: 700;
            margin: 0.2rem 0;
        }}
        .interval {{ color: {BLUE}; font-size: 0.98rem; }}
        .insight {{
            border-left: 3px solid {GOLD};
            padding: 0.15rem 0 0.15rem 0.8rem;
            color: {MUTED};
            margin-bottom: 0.55rem;
        }}
        div[data-testid="stMetric"] {{
            background: {NAVY_CARD};
            border: 1px solid rgba(212, 175, 55, 0.14);
            border-radius: 16px;
            padding: 0.75rem 0.9rem;
        }}
        .stButton>button {{
            background: linear-gradient(90deg, #b9922c, {GOLD});
            color: #1a1404;
            border: 0;
            border-radius: 999px;
            font-weight: 700;
            padding: 0.6rem 1.2rem;
        }}
        .stButton>button:hover {{ transform: translateY(-1px); }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_market_data() -> pd.DataFrame:
    raw = load_raw_dataset(DATA_PATH)
    return prepare_training_frame(raw)


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    return load_bundle(MODEL_PATH)


def money(value: float) -> str:
    return f"${value:,.0f}"


def styled_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


def metric_card(label: str, value: str, gold: bool = False) -> str:
    klass = "metric-value gold" if gold else "metric-value"
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="{klass}">{value}</div></div>'
    )


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-kicker">Aetherline Valuation</div>
        <h1 class="hero-title">{title}</h1>
        <p class="hero-sub">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )


def page_dashboard(market: pd.DataFrame) -> None:
    render_hero(
        "King County market command center",
        "Institutional overview of 2014–2015 residential transactions used to train the valuation engine.",
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Properties analyzed", f"{len(market):,}", False),
        (c2, "Average price", money(market["price"].mean()), True),
        (c3, "Median price", money(market["price"].median()), True),
        (c4, "Maximum price", money(market["price"].max()), False),
        (c5, "Avg living area", f"{market['sqft_living'].mean():,.0f} sqft", False),
    ]
    for col, label, value, gold in cards:
        col.markdown(metric_card(label, value, gold), unsafe_allow_html=True)

    left, right = st.columns((1.35, 1))
    with left:
        fig = px.histogram(
            market, x="price", nbins=60, title="Sale price distribution"
        )
        fig.update_traces(marker_color=GOLD, opacity=0.85)
        fig.update_layout(xaxis_title="Price (USD)", yaxis_title="Homes")
        st.plotly_chart(styled_figure(fig), use_container_width=True)
    with right:
        monthly = (
            market.assign(month=market["date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month", as_index=False)["price"]
            .median()
        )
        fig = px.line(monthly, x="month", y="price", title="Median sale price over time")
        fig.update_traces(line_color=BLUE, line_width=3)
        st.plotly_chart(styled_figure(fig), use_container_width=True)

    st.markdown("#### Market insights")
    waterfront_lift = (
        market.loc[market["waterfront"] == 1, "price"].median()
        / market.loc[market["waterfront"] == 0, "price"].median()
    )
    insights = [
        f"Waterfront homes transacted at a {waterfront_lift:.1f}x median premium versus inland properties.",
        f"Living area is the strongest linear correlate of price (r = {market[['price','sqft_living']].corr().iloc[0,1]:.2f}).",
        f"The market is right-skewed: mean {money(market['price'].mean())} vs median {money(market['price'].median())}.",
        "Luxury IQR outliers exist and are retained as genuine high-end sales, not data errors.",
        "The model is calibrated to 2014–2015 King County prices and should not be treated as a 2026 AVM.",
    ]
    for text in insights:
        st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def local_sensitivity(bundle: dict, payload: dict, base_pred: float) -> pd.DataFrame:
    defaults = bundle["valuation_defaults"]
    medians = bundle["feature_medians"]
    rows = []
    numeric_keys = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "grade",
        "waterfront",
        "view",
        "condition",
        "yr_built",
        "lat",
        "long",
        "sqft_basement",
    ]
    for key in numeric_keys:
        counterfactual = dict(payload)
        counterfactual[key] = medians.get(key, payload[key])
        x_alt = prepare_inference_frame(counterfactual, defaults)
        alt_pred = float(bundle["pipeline"].predict(x_alt)[0])
        rows.append(
            {
                "feature": key,
                "your_value": payload[key],
                "market_median": medians.get(key, np.nan),
                "delta_vs_median_input": base_pred - alt_pred,
            }
        )
    frame = pd.DataFrame(rows)
    frame["abs_delta"] = frame["delta_vs_median_input"].abs()
    return frame.sort_values("abs_delta", ascending=False)


def page_predict(bundle: dict, market: pd.DataFrame) -> None:
    render_hero(
        "Private valuation desk",
        "Enter property characteristics to generate a model-based estimate in 2014–2015 King County dollars.",
    )
    defaults = bundle["valuation_defaults"]
    zipcodes = bundle.get("zipcodes") or sorted(market["zipcode"].astype(str).unique())

    with st.form("valuation_form"):
        a, b, c = st.columns(3)
        bedrooms = a.number_input("Bedrooms", 1, 12, 3, 1)
        bathrooms = b.number_input("Bathrooms", 0.5, 10.0, 2.25, 0.25)
        floors = c.number_input("Floors", 1.0, 3.5, 1.0, 0.5)
        a, b, c = st.columns(3)
        sqft_living = a.number_input("Living area (sqft)", 250, 15000, 1960, 10)
        sqft_lot = b.number_input("Lot area (sqft)", 400, 1500000, 7500, 50)
        sqft_basement = c.number_input("Basement area (sqft)", 0, 8000, 0, 10)
        a, b, c, d = st.columns(4)
        waterfront = a.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x else "No")
        view = b.slider("View rating", 0, 4, 0)
        condition = c.slider("Condition", 1, 5, 3)
        grade = d.slider("Grade (King County)", 1, 13, 7)
        a, b, c = st.columns(3)
        yr_built = a.number_input("Year built", 1900, 2015, 1975, 1)
        yr_renovated = b.number_input("Renovation year (0 = none)", 0, 2015, 0, 1)
        zipcode = c.selectbox("Zipcode", zipcodes, index=min(10, len(zipcodes) - 1))
        a, b = st.columns(2)
        lat = a.number_input("Latitude", 47.0, 48.0, float(market["lat"].median()), 0.0001, format="%.4f")
        lon = b.number_input("Longitude", -123.0, -121.0, float(market["long"].median()), 0.0001, format="%.4f")
        submitted = st.form_submit_button("Generate valuation")

    if not submitted:
        st.info("Submit the form to produce a live valuation. The model is never retrained in this app.")
        return

    payload = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "zipcode": str(zipcode),
        "lat": lat,
        "long": lon,
        "sqft_basement": sqft_basement,
        "sale_year": defaults["sale_year"],
        "sale_month": defaults["sale_month"],
    }
    errors = validate_prediction_payload(payload)
    if errors:
        for err in errors:
            st.error(err)
        return

    features = prepare_inference_frame(payload, defaults)
    pred = float(bundle["pipeline"].predict(features)[0])
    q = float(bundle["conformal_q90_log"])
    low = float(np.expm1(np.log1p(pred) - q))
    high = float(np.expm1(np.log1p(pred) + q))
    sensitivity = local_sensitivity(bundle, payload, pred)

    st.markdown(
        f"""
        <div class="pred-card">
            <div class="metric-label">Estimated market value · {bundle['model_name']}</div>
            <div class="pred-price">{money(pred)}</div>
            <div class="interval">Approximate 90% conformal interval: {money(low)} – {money(high)}</div>
            <p class="hero-sub" style="margin-top:0.8rem;">
                Interval width is calibrated on chronological holdout absolute log-errors.
                It is not a formal appraisal band and does not capture 2026 market inflation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3)
    s1.markdown(
        metric_card("Living / lot ratio", f"{sqft_living / sqft_lot:.3f}"),
        unsafe_allow_html=True,
    )
    age = max(defaults["sale_year"] - int(yr_built), 0)
    s2.markdown(metric_card("House age at valuation", f"{age} years"), unsafe_allow_html=True)
    s3.markdown(
        metric_card("Basement share", f"{(sqft_basement / sqft_living) * 100:.1f}%"),
        unsafe_allow_html=True,
    )

    st.markdown("#### Property summary")
    st.dataframe(
        pd.DataFrame(
            {
                "Attribute": [
                    "Beds / baths",
                    "Living / lot",
                    "Quality",
                    "Location",
                    "Vintage",
                ],
                "Detail": [
                    f"{bedrooms:.0f} bed · {bathrooms:.2f} bath · {floors:.1f} floors",
                    f"{sqft_living:,} sqft living on {sqft_lot:,} sqft lot",
                    f"Grade {grade} · condition {condition} · view {view} · waterfront {bool(waterfront)}",
                    f"ZIP {zipcode} · ({lat:.4f}, {lon:.4f})",
                    f"Built {yr_built}"
                    + (f", renovated {yr_renovated}" if yr_renovated else ", never renovated"),
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### Characteristics that move this estimate")
    st.caption(
        "These values show how the estimate changes when each input is replaced by the "
        "training-set median. They are sensitivity effects, not causal proof."
    )
    top = sensitivity.head(8)
    fig = px.bar(
        top,
        x="delta_vs_median_input",
        y="feature",
        orientation="h",
        title="Change in predicted price versus median property",
    )
    fig.update_traces(marker_color=GOLD)
    st.plotly_chart(styled_figure(fig), use_container_width=True)


def page_analytics(market: pd.DataFrame) -> None:
    render_hero(
        "Market analytics studio",
        "Interactive exploration of price formation across size, quality, location, and time.",
    )
    f1, f2, f3 = st.columns(3)
    price_range = f1.slider(
        "Price range (USD)",
        float(market["price"].min()),
        float(market["price"].max()),
        (float(market["price"].quantile(0.05)), float(market["price"].quantile(0.95))),
        step=5000.0,
    )
    beds = f2.multiselect(
        "Bedrooms",
        sorted(market["bedrooms"].unique()),
        default=sorted(market["bedrooms"].unique()),
    )
    water = f3.selectbox("Waterfront filter", ["All", "Waterfront only", "Inland only"])
    filtered = market[
        (market["price"].between(price_range[0], price_range[1]))
        & (market["bedrooms"].isin(beds if beds else market["bedrooms"]))
    ]
    if water == "Waterfront only":
        filtered = filtered[filtered["waterfront"] == 1]
    elif water == "Inland only":
        filtered = filtered[filtered["waterfront"] == 0]

    st.caption(f"{len(filtered):,} homes in the current filter.")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered, x="price", nbins=50, title="Price distribution")
        fig.update_traces(marker_color=GOLD)
        st.plotly_chart(styled_figure(fig), use_container_width=True)
    with c2:
        sample = filtered.sample(min(4000, len(filtered)), random_state=42)
        fig = px.scatter(
            sample,
            x="sqft_living",
            y="price",
            color="grade",
            title="Price versus living area",
            opacity=0.65,
        )
        st.plotly_chart(styled_figure(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(filtered, x="bedrooms", y="price", title="Price versus bedrooms")
        fig.update_traces(marker_color=BLUE)
        st.plotly_chart(styled_figure(fig), use_container_width=True)
    with c2:
        fig = px.box(filtered, x="bathrooms", y="price", title="Price versus bathrooms")
        st.plotly_chart(styled_figure(fig), use_container_width=True)

    geo = filtered.sample(min(3500, len(filtered)), random_state=42)
    fig = px.scatter(
        geo,
        x="long",
        y="lat",
        color="price",
        title="Geographic price distribution",
        color_continuous_scale="YlOrRd",
        opacity=0.75,
    )
    st.plotly_chart(styled_figure(fig), use_container_width=True)

    zip_stats = (
        filtered.groupby("zipcode", as_index=False)
        .agg(median_price=("price", "median"), n=("price", "size"))
        .sort_values("median_price", ascending=False)
        .head(20)
    )
    fig = px.bar(zip_stats, x="zipcode", y="median_price", title="Highest median prices by zipcode")
    fig.update_traces(marker_color=GOLD)
    st.plotly_chart(styled_figure(fig), use_container_width=True)

    trend = (
        filtered.assign(month=filtered["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(median_price=("price", "median"), sales=("price", "size"))
    )
    fig = go.Figure()
    fig.add_bar(x=trend["month"], y=trend["sales"], name="Sales", marker_color=BLUE, opacity=0.45)
    fig.add_trace(
        go.Scatter(
            x=trend["month"],
            y=trend["median_price"],
            name="Median price",
            yaxis="y2",
            line=dict(color=GOLD, width=3),
        )
    )
    fig.update_layout(
        title="Sales volume and median price",
        yaxis=dict(title="Sales"),
        yaxis2=dict(title="Median price", overlaying="y", side="right"),
    )
    st.plotly_chart(styled_figure(fig), use_container_width=True)

    corr_cols = [
        "price",
        "sqft_living",
        "sqft_lot",
        "bedrooms",
        "bathrooms",
        "grade",
        "view",
        "waterfront",
        "house_age",
        "basement_pct",
        "living_lot_ratio",
        "lat",
        "long",
    ]
    corr = filtered[corr_cols].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation heatmap",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    st.plotly_chart(styled_figure(fig), use_container_width=True)


def page_performance(bundle: dict) -> None:
    render_hero(
        "Model governance",
        "Chronological holdout results, residual diagnostics, and permutation importance for the production pipeline.",
    )
    comparison = pd.DataFrame(bundle["comparison"])
    display = comparison.copy()
    for col in ["train_mae", "train_rmse", "test_mae", "test_rmse"]:
        display[col] = display[col].map(lambda x: f"${x:,.0f}")
    for col in ["train_r2", "test_r2"]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    display["test_mape"] = display["test_mape"].map(lambda x: f"{x:.2f}%")
    display["cv_mae"] = display["cv_mae"].map(
        lambda x: "—" if pd.isna(x) else f"${x:,.0f}"
    )
    st.markdown("#### Chronological test comparison")
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption(bundle.get("mape_note", ""))

    metrics = bundle["chrono_test_metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(metric_card("Best model", bundle["model_name"], True), unsafe_allow_html=True)
    m2.markdown(metric_card("Test MAE", money(metrics["mae"])), unsafe_allow_html=True)
    m3.markdown(metric_card("Test RMSE", money(metrics["rmse"])), unsafe_allow_html=True)
    m4.markdown(metric_card("Test R²", f"{metrics['r2']:.3f}"), unsafe_allow_html=True)

    sample = pd.DataFrame(bundle["test_sample"])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            sample, x="y_true", y="y_pred", title="Actual versus predicted (holdout sample)"
        )
        max_val = max(sample["y_true"].max(), sample["y_pred"].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Perfect fit",
                line=dict(color=BLUE, dash="dash"),
            )
        )
        st.plotly_chart(styled_figure(fig), use_container_width=True)
    with c2:
        fig = px.histogram(sample, x="residual", nbins=50, title="Residual distribution")
        fig.update_traces(marker_color=GOLD)
        st.plotly_chart(styled_figure(fig), use_container_width=True)

    fig = px.scatter(
        sample,
        x="y_pred",
        y="residual",
        title="Residuals versus predicted price",
        opacity=0.65,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=BLUE)
    st.plotly_chart(styled_figure(fig), use_container_width=True)

    importance = pd.DataFrame(bundle["feature_importance"]).head(15)
    fig = px.bar(
        importance.sort_values("mae_increase"),
        x="mae_increase",
        y="feature",
        orientation="h",
        title="Permutation importance (increase in MAE when shuffled)",
    )
    fig.update_traces(marker_color=GOLD)
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    st.caption(
        "Permutation importance measures predictive association on the holdout set. "
        "It is not evidence that a feature causes price to change."
    )

    split = bundle["split"]
    rand = bundle["random_split_diagnostic_metrics"]
    st.markdown("#### Best-model summary")
    st.write(
        f"Selected **{bundle['model_name']}** using chronological holdout MAE. "
        f"Train window {split['train_start']} to {split['train_end']} "
        f"({split['train_rows']:,} homes); test window {split['test_start']} to "
        f"{split['test_end']} ({split['test_rows']:,} homes)."
    )
    st.write(
        f"A random 80/20 diagnostic on a separately fitted clone scored "
        f"MAE {money(rand['mae'])} / R² {rand['r2']:.3f}. Random splits on this "
        f"one-year market are usually optimistic relative to true forward performance."
    )


def page_about(bundle: dict) -> None:
    render_hero(
        "About this platform",
        "A senior-style valuation system: leakage-aware features, chronological evaluation, and a production Streamlit interface.",
    )
    profile = bundle.get("data_profile", {})
    st.markdown("#### Project overview")
    st.write(
        "Aetherline Valuation estimates King County residential sale prices from property "
        "attributes. Training is isolated in `feature_engineering.py`. The Streamlit app "
        "only loads the serialized pipeline and never fits a model at startup."
    )
    st.markdown("#### Dataset")
    st.write(
        f"King County house sales (`kc_house_data.csv`): {profile.get('n_rows', 21613):,} raw records, "
        f"{profile.get('cleaned_rows', 0):,} after validation, {profile.get('n_zipcodes', 70)} zipcodes, "
        f"sales from {profile.get('date_min', '2014-05-02')} to {profile.get('date_max', '2015-05-27')}. "
        f"{profile.get('duplicate_ids', 177)} homes sold more than once; repeat sales are kept as "
        f"separate dated transactions. `id` is excluded from features."
    )
    st.markdown("#### Methodology")
    st.write(
        "Features include house age, renovation flags, years since renovation, living-to-lot ratio, "
        "basement percentage, bathrooms per bedroom, and sale month/year. Models: Ridge regression "
        "on log price (baseline), Random Forest, and XGBoost. Selection uses chronological 80/20 "
        "holdout MAE, with time-series cross-validation inside the training window. Targets are "
        "modeled in log space and inverted to USD."
    )
    st.markdown("#### Technology stack")
    st.write("Python, pandas, NumPy, scikit-learn, XGBoost, Joblib, Streamlit, Plotly.")
    st.markdown("#### Limitations")
    st.write(
        "- The label is a 2014–2015 transaction price, not a current asking price or appraisal.\n"
        "- Neighborhood comps (`sqft_living15`, `sqft_lot15`) are omitted because they are not "
        "available on the valuation form.\n"
        "- Prediction intervals are split-conformal on log residuals, not legal confidence bounds.\n"
        "- Feature importance and sensitivity plots are associative, not causal.\n"
        "- Extreme luxury homes remain high-error; MAE is not uniform across the price range."
    )


def main() -> None:
    st.set_page_config(
        page_title="Aetherline Valuation",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    st.sidebar.markdown("### Aetherline")
    st.sidebar.caption("King County residential intelligence")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Executive Dashboard",
            "House Price Prediction",
            "Market Analytics",
            "Model Performance",
            "About",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Model artifacts load from `models/best_model.joblib`. Training is offline.")

    try:
        market = load_market_data()
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load application assets: {exc}")
        st.stop()

    if page == "Executive Dashboard":
        page_dashboard(market)
    elif page == "House Price Prediction":
        page_predict(bundle, market)
    elif page == "Market Analytics":
        page_analytics(market)
    elif page == "Model Performance":
        page_performance(bundle)
    else:
        page_about(bundle)


if __name__ == "__main__":
    main()
