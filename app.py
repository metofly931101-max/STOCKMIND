
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="STOCKMIND",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

COLORS = {
    "BUY": "#16866D",
    "WAIT": "#D5A13B",
    "AVOID": "#DC6262",
    "REVIEW": "#6387B5",
    "INSUFFICIENT_DATA": "#9CA3AF"
}

LABELS = {
    "BUY": "ผ่านเกณฑ์คัดกรอง",
    "WAIT": "รอราคา / พื้นฐาน",
    "AVOID": "ไม่ผ่านเกณฑ์",
    "REVIEW": "วิเคราะห์เฉพาะกลุ่ม",
    "INSUFFICIENT_DATA": "ข้อมูลไม่เพียงพอ"
}

# 2. LOAD DATA

@st.cache_data
def load_data(filename):
    return pd.read_csv(DATA_DIR / filename)

try:
    df = load_data(
        "stockmind_final_dataset.csv"
    )

    assessment = load_data(
        "model_assessment.csv"
    )

except Exception as e:
    st.error(f"Cannot load application data: {e}")
    st.stop()


def fmt(value, suffix=""):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}{suffix}"


# 3. SIDEBAR NAVIGATION

st.sidebar.title("STOCKMIND")

st.sidebar.caption(
    "Graham-Based Stock Analysis"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Market Overview",
        "Stock Screener",
        "Stock Analysis",
        "Stock Comparison",
        "Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Academic Research Project\n\n"
    "Graham Engine + Machine Learning"
)

st.sidebar.caption(
    "ข้อมูลจากการประมวลผลใน Colab "
    "ไม่ใช่ข้อมูลราคาหุ้นแบบ Real-time"
)

# 4. MARKET OVERVIEW

if page == "Market Overview":

    st.title("STOCKMIND")

    st.caption(
        "Intelligent Value Investing Platform"
    )

    st.subheader("SET100 Market Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Stocks", len(df))

    c2.metric(
        "BUY Candidates",
        int((df["Recommendation"] == "BUY").sum())
    )

    c3.metric(
        "WAIT Candidates",
        int((df["Recommendation"] == "WAIT").sum())
    )

    c4.metric(
        "ML Coverage",
        f'{df["ML_Predicted_Return_Pct"].notna().mean():.0%}'
    )

    summary = (
        df["Recommendation"]
        .value_counts()
        .rename_axis("Recommendation")
        .reset_index(name="Count")
    )

    left, right = st.columns(2)

    with left:

        fig = px.pie(
            summary,
            names="Recommendation",
            values="Count",
            hole=0.60,
            color="Recommendation",
            color_discrete_map=COLORS,
            title="Recommendation Distribution"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with right:

        fig = px.bar(
            summary,
            x="Recommendation",
            y="Count",
            color="Recommendation",
            color_discrete_map=COLORS,
            title="Stock Screening Results"
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            fig,
            width="stretch"
        )

    st.subheader("Stocks Passing All Criteria")

    st.dataframe(
        df.loc[
            df["Recommendation"] == "BUY",
            [
                "Ticker",
                "Price",
                "Graham_Number",
                "Screening_Buy_Price",
                "MOS_Pct",
                "Recommendation"
            ]
        ],
        hide_index=True,
        width="stretch"
    )

# 5. STOCK SCREENER

elif page == "Stock Screener":

    st.title("Stock Screener")

    c1, c2 = st.columns(2)

    with c1:

        sectors = st.multiselect(
            "Select Sector",
            sorted(df["Sector"].dropna().unique())
        )

    with c2:

        recommendations = st.multiselect(
            "Recommendation",
            list(COLORS.keys()),
            default=list(COLORS.keys())
        )

    search = st.text_input(
        "Search Stock Ticker"
    ).strip().upper()

    filtered = df.copy()

    if sectors:

        filtered = filtered[
            filtered["Sector"].isin(sectors)
        ]

    filtered = filtered[
        filtered["Recommendation"].isin(
            recommendations
        )
    ]

    if search:

        filtered = filtered[
            filtered["Ticker"].str.contains(
                search,
                regex=False,
                na=False
            )
        ]

    st.write(f"Found {len(filtered)} stocks")

    columns = [
        "Ticker",
        "Sector",
        "Price",
        "Graham_Number",
        "Screening_Buy_Price",
        "MOS_Pct",
        "Quality_Status",
        "Recommendation",
        "ML_Predicted_Return_Pct"
    ]

    st.dataframe(
        filtered[columns].round(4),
        hide_index=True,
        width="stretch"
    )

    st.download_button(
        "Download Screening Results",
        data=filtered.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="stockmind_screening.csv",
        mime="text/csv"
    )

# 6. INDIVIDUAL STOCK ANALYSIS

elif page == "Stock Analysis":

    st.title("Individual Stock Analysis")

    ticker = st.selectbox(
        "Select Stock",
        sorted(df["Ticker"].unique())
    )

    row = df.loc[
        df["Ticker"] == ticker
    ].iloc[0]

    recommendation = row["Recommendation"]

    st.header(ticker)

    st.caption(str(row["Sector"]))

    message = (
        f'{recommendation} — '
        f'{LABELS.get(recommendation, "")}'
    )

    if recommendation == "BUY":
        st.success(message)

    elif recommendation == "AVOID":
        st.error(message)

    elif recommendation == "WAIT":
        st.warning(message)

    else:
        st.info(message)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Stock Price",
        fmt(row["Price"], " THB")
    )

    c2.metric(
        "Graham Number",
        fmt(row["Graham_Number"], " THB")
    )

    c3.metric(
        "Screening Buy Price",
        fmt(row["Screening_Buy_Price"], " THB")
    )

    c4.metric(
        "Margin of Safety",
        fmt(row["MOS_Pct"], "%")
    )

    st.subheader("Valuation Analysis")

    valuation = pd.DataFrame({
        "Metric": [
            "Current Price",
            "Screening Buy Price",
            "Graham Number"
        ],
        "Price": [
            row["Price"],
            row["Screening_Buy_Price"],
            row["Graham_Number"]
        ]
    }).dropna()

    fig = px.bar(
        valuation,
        x="Price",
        y="Metric",
        orientation="h",
        color="Metric",
        text_auto=".2f"
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader("Financial Fundamentals")

    fundamentals = pd.DataFrame({
        "Financial Metric": [
            "EPS 2025",
            "Book Value Per Share",
            "Current Ratio",
            "Debt to Assets (%)",
            "EPS CAGR 3Y (%)",
            "P/E Ratio",
            "P/B Ratio"
        ],
        "Value": [
            row["EPS_2025"],
            row["BVS"],
            row["Current_Ratio"],
            row["Debt_to_Assets"] * 100,
            row["EPS_CAGR_3Y"] * 100,
            row["PE_Ratio"],
            row["PB_Ratio"]
        ]
    })

    st.dataframe(
        fundamentals.round(4),
        hide_index=True,
        width="stretch"
    )

    st.subheader("Recommendation Explanation")

    st.write(row["Recommendation_Reason"])

    st.write(
        "Required Price Drop:",
        fmt(
            row["Required_Price_Drop_Pct"],
            "%"
        )
    )

    st.caption(
        "ราคาคัดกรองเป็นผลจากสมมติฐาน "
        "ไม่ใช่ราคาเป้าหมายที่รับประกันผลตอบแทน"
    )

    st.divider()

    st.subheader("Machine Learning Analysis")

    c1, c2 = st.columns(2)

    c1.metric(
        "Predicted 12M Return",
        fmt(
            row["ML_Predicted_Return_Pct"],
            "%"
        )
    )

    c2.metric(
        "ML Signal",
        str(row["ML_Signal"])
    )

    st.write(
        "ML Data Date:",
        row["ML_As_Of_Date"]
        if pd.notna(row["ML_As_Of_Date"])
        else "N/A"
    )

    st.warning(
        "Experimental ML prediction only. "
        "Not used to determine BUY or SELL decisions."
    )

# 7. STOCK COMPARISON

elif page == "Stock Comparison":

    st.title("Stock Comparison")

    selected = st.multiselect(
        "Select up to 5 stocks",
        sorted(df["Ticker"].unique()),
        max_selections=5
    )

    if not selected:

        st.info("Please select stocks.")

    else:

        comparison = df[
            df["Ticker"].isin(selected)
        ].copy()

        columns = [
            "Ticker",
            "Price",
            "Graham_Number",
            "Screening_Buy_Price",
            "MOS_Pct",
            "PE_Ratio",
            "PB_Ratio",
            "Current_Ratio",
            "Recommendation",
            "ML_Predicted_Return_Pct"
        ]

        st.dataframe(
            comparison[columns].round(4),
            hide_index=True,
            width="stretch"
        )

        fig = px.bar(
            comparison,
            x="Ticker",
            y=[
                "Price",
                "Screening_Buy_Price",
                "Graham_Number"
            ],
            barmode="group",
            title="Stock Valuation Comparison"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        fig = px.bar(
            comparison,
            x="Ticker",
            y="MOS_Pct",
            color="Recommendation",
            color_discrete_map=COLORS,
            title="Margin of Safety Comparison"
        )

        fig.add_hline(
            y=30,
            line_dash="dash",
            annotation_text="MOS Threshold 30%"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

# 8. MODEL PERFORMANCE

elif page == "Model Performance":

    st.title("Machine Learning Performance")

    model = assessment.iloc[0]

    st.subheader(str(model["Model"]))

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Directional Accuracy",
        fmt(
            model["Directional_Accuracy_Pct"],
            "%"
        )
    )

    c2.metric(
        "MAE (percentage points)",
        fmt(model["MAE"] * 100)
    )

    c3.metric(
        "RMSE (percentage points)",
        fmt(model["RMSE"] * 100)
    )

    c4.metric(
        "R² Score",
        f'{model["R2"]:.4f}'
    )

    validation = load_data(
        "model_validation_results.csv"
    )

    testing = load_data(
        "final_test_results.csv"
    )

    st.subheader("Validation Results")

    st.dataframe(
        validation,
        hide_index=True,
        width="stretch"
    )

    fig = px.bar(
        validation,
        x="MAE",
        y="Model",
        orientation="h",
        title="Validation Model Comparison"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader("Final Test Results")

    st.dataframe(
        testing,
        hide_index=True,
        width="stretch"
    )

    st.subheader("Model Limitations")

    st.write(
        """
        - Uses historical price-based features.
        - Predicts future 12-month returns.
        - Low explanatory power (R²).
        - Potential survivorship bias.
        - Experimental information only.
        """
    )

# 9. FOOTER

st.divider()

st.caption(
    "STOCKMIND | Academic Project | "
    "Graham-Based Screening & Machine Learning"
)

st.caption(
    "For educational purposes only. "
    "Not personalized investment advice."
)
