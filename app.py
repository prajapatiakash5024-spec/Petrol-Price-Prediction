import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Petrol Price Predictor",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: #0a0e1a;
        color: #e2e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f1628 !important;
        border-right: 1px solid #1e2d4a;
    }

    /* Hero header */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3.4rem;
        font-weight: 900;
        color: #fbbf24;
        text-shadow: 0 0 30px rgba(202, 138, 4, 0.5), 0 2px 8px rgba(0,0,0,0.5);
        line-height: 1.1;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
        -webkit-text-fill-color: #fbbf24;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        font-weight: 300;
        letter-spacing: 0.5px;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1a2234 100%);
        border: 1px solid #1e2d4a;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #ca8a04;
    }

    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ca8a04;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
    }

    /* Section headers */
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 900;
        color: #fbbf24;
        border-left: 5px solid #d97706;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
        text-shadow: 0 0 20px rgba(202, 138, 4, 0.4);
        letter-spacing: -0.3px;
    }

    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, #1a1100 0%, #1c1203 100%);
        border: 2px solid #ca8a04;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(202, 138, 4, 0.15);
    }

    .prediction-value {
        font-family: 'Syne', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        color: #ca8a04;
    }

    .prediction-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Info badge */
    .badge {
        display: inline-block;
        background: rgba(202, 138, 4, 0.15);
        border: 1px solid rgba(202, 138, 4, 0.4);
        color: #d97706;
        border-radius: 30px;
        padding: 0.3rem 1rem;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #1e2d4a, transparent);
        margin: 2rem 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0f1628;
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: #ca8a04 !important;
        color: white !important;
    }

    /* Slider */
    .stSlider [data-testid="stThumbValue"] {
        color: #ca8a04;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #ca8a04 0%, #a16207 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        width: 100%;
        transition: all 0.2s ease;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(202, 138, 4, 0.4);
    }

    /* DataFrame */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: #111827;
        border: 1px solid #1e2d4a;
        border-radius: 10px;
        color: #e2e8f0;
    }

    /* Success/info boxes */
    .stSuccess, .stInfo {
        border-radius: 12px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #ca8a04; }
</style>
""", unsafe_allow_html=True)


# ─── Load & Prepare Data ─────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_excel("Petrol Dataset June 23 2022 -- Version 2.csv.xlsx")
    df.columns = df.columns.str.strip()
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Price Per Liter (USD)", axis=1)
    y = df["Price Per Liter (USD)"]
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "mse": round(mean_squared_error(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "r2": round(r2_score(y_test, y_pred), 4),
    }

    return df, model, X, X_train, X_test, y_train, y_test, y_pred, metrics


df, model, X, X_train, X_test, y_train, y_test, y_pred, metrics = load_and_train()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:2.5rem; margin-bottom:0.5rem;'>⛽</div>
        <div style='font-family: Syne, sans-serif; font-weight:700; font-size:1.1rem; color:#ca8a04;'>PetrolIQ</div>
        <div style='font-size:0.72rem; color:#475569; text-transform:uppercase; letter-spacing:2px;'>ML Price Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#64748b; font-size:0.78rem; text-transform:uppercase; letter-spacing:1.5px;'>Navigation</p>", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🏠 Overview", "📊 Data Explorer", "🤖 Model & Metrics", "🔮 Live Predictor"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='background:#111827; border:1px solid #1e2d4a; border-radius:12px; padding:1rem;'>
        <p style='font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;'>Dataset Info</p>
        <p style='color:#e2e8f0; font-size:0.85rem; margin:0.2rem 0;'>📋 <b>{df.shape[0]}</b> countries</p>
        <p style='color:#e2e8f0; font-size:0.85rem; margin:0.2rem 0;'>📌 <b>{df.shape[1]}</b> features</p>
        <p style='color:#e2e8f0; font-size:0.85rem; margin:0.2rem 0;'>📅 June 2022</p>
        <p style='color:#e2e8f0; font-size:0.85rem; margin:0.2rem 0;'>🤖 Linear Regression</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;'>
        <span class='badge'>v1.0 · Mini Project</span>
    </div>
    """, unsafe_allow_html=True)


# ─── PAGES ────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════
#  PAGE 1: OVERVIEW
# ══════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div>
        <div class='hero-title'>Petrol Price<br>Intelligence</div>
        <div class='hero-subtitle'>Machine Learning–Powered Global Fuel Price Analysis & Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{df.shape[0]}</div>
            <div class='metric-label'>Countries Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>${df['Price Per Liter (USD)'].mean():.2f}</div>
            <div class='metric-label'>Avg Price / Liter</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>${df['Price Per Liter (USD)'].max():.2f}</div>
            <div class='metric-label'>Highest Price</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{metrics['r2']}</div>
            <div class='metric-label'>R² Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("<div class='section-header'>Global Petrol Price Distribution</div>", unsafe_allow_html=True)
        fig = px.histogram(
            df, x="Price Per Liter (USD)", nbins=30,
            color_discrete_sequence=["#ca8a04"],
            template="plotly_dark"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font_color="#94a3b8",
            bargap=0.05,
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>Project Context</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#94a3b8; font-size:0.9rem; line-height:1.8;'>
        Fuel prices are shaped by <b style='color:#d97706;'>geopolitical events</b>, crude oil supply, 
        and economic indicators. This project leverages <b style='color:#d97706;'>Linear Regression</b> 
        to model global petrol price patterns across <b style='color:#d97706;'>181 countries</b> 
        using June 2022 data — a period of significant market volatility due to the 
        Russia–Ukraine conflict and global supply chain disruptions.
        </div>
        <br>
        <div style='display:flex; gap:0.5rem; flex-wrap:wrap;'>
            <span class='badge'>Linear Regression</span>
            <span class='badge'>Scikit-Learn</span>
            <span class='badge'>Plotly</span>
            <span class='badge'>Pandas</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Top 15 Most Expensive Petrol Countries</div>", unsafe_allow_html=True)
    top15 = df.nlargest(15, "Price Per Liter (USD)")[["Country", "Price Per Liter (USD)", "GDP Per Capita ( USD )"]].reset_index(drop=True)
    fig2 = px.bar(
        top15, x="Price Per Liter (USD)", y="Country",
        orientation='h',
        color="Price Per Liter (USD)",
        color_continuous_scale=["#1e3a5f", "#ca8a04"],
        template="plotly_dark",
        hover_data=["GDP Per Capita ( USD )"]
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,22,40,0.8)",
        font_color="#94a3b8",
        xaxis=dict(gridcolor="#1e2d4a"),
        yaxis=dict(gridcolor="#1e2d4a"),
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 2: DATA EXPLORER
# ══════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown("<div class='hero-title' style='font-size:2rem;'>Data Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Explore and visualize the petrol dataset</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📈 Correlations", "🌍 Country Insights"])

    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            search = st.text_input("🔍 Search by Country", placeholder="e.g. India, Germany...")
        with col2:
            rows = st.selectbox("Show rows", [10, 25, 50, 100, 181], index=0)

        filtered = df[df["Country"].str.contains(search, case=False, na=False)] if search else df
        st.dataframe(
            filtered.head(rows).style.format({
                "Price Per Liter (USD)": "${:.2f}",
                "Price Per Gallon (USD)": "${:.2f}",
                "Price Per Liter (PKR)": "₨{:.2f}",
                "World Share": "{:.1%}",
                "GDP Per Capita ( USD )": "${:,.0f}",
            }),
            use_container_width=True, height=420
        )
        st.markdown(f"<p style='color:#64748b; font-size:0.8rem;'>Showing {min(rows, len(filtered))} of {len(filtered)} records</p>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='section-header'>Feature Correlation Heatmap</div>", unsafe_allow_html=True)
        numeric_df = df.select_dtypes(include=np.number).drop(columns=["S#"], errors='ignore')
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('#0f1628')
        ax.set_facecolor('#0f1628')
        sns.heatmap(
            corr, annot=True, fmt=".2f", ax=ax,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            linewidths=0.5, linecolor='#0a0e1a',
            annot_kws={"size": 8, "color": "white"}
        )
        ax.tick_params(colors='#94a3b8', labelsize=8)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        # Correlation with target
        st.markdown("<div class='section-header'>Correlation with Price Per Liter (USD)</div>", unsafe_allow_html=True)
        corr_target = corr["Price Per Liter (USD)"].drop("Price Per Liter (USD)").sort_values()
        fig3 = px.bar(
            x=corr_target.values, y=corr_target.index,
            orientation='h',
            color=corr_target.values,
            color_continuous_scale=["#ef4444", "#1e2d4a", "#ca8a04"],
            template="plotly_dark"
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font_color="#94a3b8",
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1e2d4a", title="Correlation Coefficient"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=320
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-header'>GDP vs Petrol Price</div>", unsafe_allow_html=True)
        fig4 = px.scatter(
            df, x="GDP Per Capita ( USD )", y="Price Per Liter (USD)",
            hover_name="Country",
            size="Daily Oil Consumption (Barrels)",
            color="Price Per Liter (USD)",
            color_continuous_scale=["#1e3a5f", "#ca8a04"],
            template="plotly_dark",
            size_max=40
        )
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font_color="#94a3b8",
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=400
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("<div class='section-header'>Price Distribution by Consumption Tier</div>", unsafe_allow_html=True)
        df_copy = df.copy()
        df_copy["Consumption Tier"] = pd.qcut(
            df_copy["Daily Oil Consumption (Barrels)"],
            q=3, labels=["Low", "Medium", "High"]
        )
        fig5 = px.box(
            df_copy, x="Consumption Tier", y="Price Per Liter (USD)",
            color="Consumption Tier",
            color_discrete_map={"Low": "#1e3a5f", "Medium": "#ca8a04", "High": "#fbbf24"},
            template="plotly_dark"
        )
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font_color="#94a3b8",
            showlegend=False,
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=320
        )
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 3: MODEL & METRICS
# ══════════════════════════════════════════════
elif page == "🤖 Model & Metrics":
    st.markdown("<div class='hero-title' style='font-size:2rem;'>Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Linear Regression — Training & Evaluation Results</div>", unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    metric_data = [
        ("MAE", metrics["mae"], "Mean Absolute Error"),
        ("MSE", metrics["mse"], "Mean Squared Error"),
        ("RMSE", metrics["rmse"], "Root Mean Squared Error"),
        ("R²", metrics["r2"], "R² Score (Accuracy)"),
    ]
    for col, (label, val, desc) in zip([col1, col2, col3, col4], metric_data):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
                <div style='font-size:0.7rem; color:#334155; margin-top:0.3rem;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("<div class='section-header'>Actual vs Predicted</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test.values, y=y_pred,
            mode='markers',
            marker=dict(color='#ca8a04', size=7, opacity=0.7, line=dict(color='#d97706', width=1)),
            name='Predictions'
        ))
        # Perfect prediction line
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode='lines',
            line=dict(color='#64748b', dash='dash', width=1.5),
            name='Perfect Fit'
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font_color="#94a3b8",
            xaxis=dict(title="Actual Price (USD)", gridcolor="#1e2d4a"),
            yaxis=dict(title="Predicted Price (USD)", gridcolor="#1e2d4a"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>Residuals Distribution</div>", unsafe_allow_html=True)
        residuals = y_test.values - y_pred
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=residuals, nbinsx=25,
            marker_color='#ca8a04', opacity=0.8,
            name='Residuals'
        ))
        fig2.add_vline(x=0, line_color="#64748b", line_dash="dash", line_width=1.5)
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,22,40,0.8)",
            font_color="#94a3b8",
            xaxis=dict(title="Residual (Actual − Predicted)", gridcolor="#1e2d4a"),
            yaxis=dict(title="Count", gridcolor="#1e2d4a"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>Top Feature Importances (Coefficients)</div>", unsafe_allow_html=True)
    coeff_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    }).reindex(pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
               .assign(abs_coef=lambda d: d["Coefficient"].abs())
               .sort_values("abs_coef", ascending=False).index).head(10)

    fig3 = px.bar(
        coeff_df.sort_values("Coefficient"),
        x="Coefficient", y="Feature",
        orientation='h',
        color="Coefficient",
        color_continuous_scale=["#ef4444", "#1e2d4a", "#ca8a04"],
        template="plotly_dark"
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,22,40,0.8)",
        font_color="#94a3b8",
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#1e2d4a"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=350
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Data split summary
    st.markdown("<div class='section-header'>Training Split Summary</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(X_train)}</div>
            <div class='metric-label'>Training Samples (70%)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(X_test)}</div>
            <div class='metric-label'>Test Samples (30%)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{X.shape[1]}</div>
            <div class='metric-label'>Input Features</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 4: LIVE PREDICTOR
# ══════════════════════════════════════════════
elif page == "🔮 Live Predictor":
    st.markdown("<div class='hero-title' style='font-size:2rem;'>Live Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Adjust the parameters below to predict petrol price per liter (USD)</div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown("<div class='section-header'>Input Parameters</div>", unsafe_allow_html=True)

        daily_oil = st.slider(
            "Daily Oil Consumption (Barrels)",
            int(df["Daily Oil Consumption (Barrels)"].min()),
            int(df["Daily Oil Consumption (Barrels)"].max()),
            int(df["Daily Oil Consumption (Barrels)"].median()),
            step=1000,
            help="Country's daily oil consumption in barrels"
        )
        world_share = st.slider(
            "World Share (%)",
            0.0, float(df["World Share"].max()),
            float(df["World Share"].median()),
            step=0.001,
            format="%.3f",
            help="Country's share of global oil consumption"
        )
        yearly_gallons = st.slider(
            "Yearly Gallons Per Capita",
            float(df["Yearly Gallons Per Capita"].min()),
            float(df["Yearly Gallons Per Capita"].max()),
            float(df["Yearly Gallons Per Capita"].median()),
            step=1.0
        )
        price_gallon = st.slider(
            "Price Per Gallon (USD)",
            float(df["Price Per Gallon (USD)"].min()),
            float(df["Price Per Gallon (USD)"].max()),
            float(df["Price Per Gallon (USD)"].median()),
            step=0.01
        )
        price_pkr = st.slider(
            "Price Per Liter (PKR)",
            float(df["Price Per Liter (PKR)"].min()),
            float(df["Price Per Liter (PKR)"].max()),
            float(df["Price Per Liter (PKR)"].median()),
            step=0.5
        )
        gdp = st.slider(
            "GDP Per Capita (USD)",
            int(df["GDP Per Capita ( USD )"].min()),
            int(df["GDP Per Capita ( USD )"].max()),
            int(df["GDP Per Capita ( USD )"].median()),
            step=100
        )
        gallons_buy = st.slider(
            "Gallons GDP Per Capita Can Buy",
            int(df["Gallons GDP Per Capita Can Buy"].min()),
            int(df["Gallons GDP Per Capita Can Buy"].max()),
            int(df["Gallons GDP Per Capita Can Buy"].median()),
            step=10
        )
        xtimes = st.slider(
            "xTimes Yearly Gallons Per Capita Buy",
            int(df["xTimes Yearly Gallons Per Capita Buy"].min()),
            int(df["xTimes Yearly Gallons Per Capita Buy"].max()),
            int(df["xTimes Yearly Gallons Per Capita Buy"].median()),
            step=1
        )

        predict_btn = st.button("⚡ Predict Petrol Price")

    with col_result:
        st.markdown("<div class='section-header'>Prediction Result</div>", unsafe_allow_html=True)

        if predict_btn:
            # Build input matching model features
            input_data = {
                "S#": [0],
                "Daily Oil Consumption (Barrels)": [daily_oil],
                "World Share": [world_share],
                "Yearly Gallons Per Capita": [yearly_gallons],
                "Price Per Gallon (USD)": [price_gallon],
                "Price Per Liter (PKR)": [price_pkr],
                "GDP Per Capita ( USD )": [gdp],
                "Gallons GDP Per Capita Can Buy": [gallons_buy],
                "xTimes Yearly Gallons Per Capita Buy": [xtimes],
            }
            input_df = pd.DataFrame(input_data)
            input_encoded = pd.get_dummies(input_df, drop_first=True)
            input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
            prediction = model.predict(input_encoded)[0]
            prediction = max(0, prediction)  # clamp negatives

            st.markdown(f"""
            <div class='prediction-box'>
                <div class='prediction-label'>Estimated Price Per Liter</div>
                <div class='prediction-value'>${prediction:.4f}</div>
                <div style='color:#64748b; font-size:0.8rem; margin-top:0.5rem;'>USD · June 2022 Pricing Model</div>
            </div>
            """, unsafe_allow_html=True)

            # Comparison
            avg = df["Price Per Liter (USD)"].mean()
            diff = prediction - avg
            diff_pct = (diff / avg) * 100

            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='font-size:1.5rem;'>${avg:.2f}</div>
                    <div class='metric-label'>Global Average</div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                color = "#ca8a04" if diff >= 0 else "#d97706"
                arrow = "▲" if diff >= 0 else "▼"
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='font-size:1.5rem; color:{color};'>{arrow} {abs(diff_pct):.1f}%</div>
                    <div class='metric-label'>vs Global Avg</div>
                </div>""", unsafe_allow_html=True)

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                number=dict(prefix="$", font=dict(color="#ca8a04", size=28)),
                gauge=dict(
                    axis=dict(range=[0, df["Price Per Liter (USD)"].max()],
                              tickcolor="#64748b", tickfont=dict(color="#64748b")),
                    bar=dict(color="#ca8a04"),
                    bgcolor="#0f1628",
                    borderwidth=1, bordercolor="#1e2d4a",
                    steps=[
                        dict(range=[0, avg * 0.7], color="#1e3a5f"),
                        dict(range=[avg * 0.7, avg * 1.3], color="#1a2234"),
                        dict(range=[avg * 1.3, df["Price Per Liter (USD)"].max()], color="#1c1203"),
                    ],
                    threshold=dict(line=dict(color="#fbbf24", width=2), thickness=0.75, value=avg)
                )
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8",
                margin=dict(l=20, r=20, t=20, b=10),
                height=220
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        else:
            st.markdown("""
            <div style='background:#0f1628; border:1px dashed #1e2d4a; border-radius:20px; padding:3rem; text-align:center;'>
                <div style='font-size:3rem; margin-bottom:1rem;'>🔮</div>
                <div style='color:#475569; font-size:0.95rem;'>Adjust the sliders on the left<br>and click <b style='color:#ca8a04;'>Predict Petrol Price</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Model info reminder
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#111827; border:1px solid #1e2d4a; border-radius:12px; padding:1rem;'>
            <p style='font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;'>Model Info</p>
            <p style='color:#94a3b8; font-size:0.82rem; margin:0.2rem 0;'>Algorithm: Linear Regression</p>
            <p style='color:#94a3b8; font-size:0.82rem; margin:0.2rem 0;'>R² Score: <b style='color:#ca8a04;'>{metrics['r2']}</b></p>
            <p style='color:#94a3b8; font-size:0.82rem; margin:0.2rem 0;'>MAE: <b style='color:#ca8a04;'>${metrics['mae']}</b></p>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding:1rem 0; color:#334155; font-size:0.78rem;'>
    ⛽ PetrolIQ · Petrol Price Prediction using Machine Learning · Mini Project 2024
</div>
""", unsafe_allow_html=True)
