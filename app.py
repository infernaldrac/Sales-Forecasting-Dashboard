
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Sales Forecasting Dashboard", page_icon="📈", layout="wide")

st.title("📈 Sales Forecasting Dashboard")
st.caption("AI & Data Science Internship Project")

possible_paths = [
    "Sample - Superstore.csv",
    "Dataset/Sample - Superstore.csv",
    "Superstore.csv",
    "Dataset/Superstore.csv"
]

df = None
for p in possible_paths:
    if Path(p).exists():
        df = pd.read_csv(p)
        break

if df is None:
    st.error("Dataset not found.")
    st.stop()

if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()

st.sidebar.header("Filters")

category = st.sidebar.selectbox(
    "Category",
    ["All"] + sorted(df["Category"].dropna().unique().tolist())
)

region = st.sidebar.selectbox(
    "Region",
    ["All"] + sorted(df["Region"].dropna().unique().tolist())
)

filtered_df = df.copy()

if category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == category]

if region != "All":
    filtered_df = filtered_df[filtered_df["Region"] == region]

c1,c2,c3,c4 = st.columns(4)

c1.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
c2.metric("Orders", filtered_df["Order ID"].nunique())
c3.metric("Customers", filtered_df["Customer ID"].nunique())
c4.metric("Average Sale", f"${filtered_df['Sales'].mean():,.2f}")

st.divider()

left,right = st.columns(2)

with left:
    st.subheader("Sales by Category")
    cat = filtered_df.groupby("Category",as_index=False)["Sales"].sum()
    st.plotly_chart(px.bar(cat,x="Category",y="Sales"), use_container_width=True)

with right:
    st.subheader("Sales by Region")
    reg = filtered_df.groupby("Region",as_index=False)["Sales"].sum()
    st.plotly_chart(px.bar(reg,x="Region",y="Sales"), use_container_width=True)

if "Order Date" in filtered_df.columns:
    monthly = filtered_df.groupby(pd.Grouper(key="Order Date",freq="M"))["Sales"].sum().reset_index()
    st.subheader("Monthly Sales Trend")
    st.plotly_chart(px.line(monthly,x="Order Date",y="Sales",markers=True), use_container_width=True)

col1,col2=st.columns(2)

with col1:
    st.subheader("Top 10 Products")
    top=filtered_df.groupby("Product Name",as_index=False)["Sales"].sum().sort_values("Sales",ascending=False).head(10)
    st.plotly_chart(px.bar(top,x="Sales",y="Product Name",orientation="h"),use_container_width=True)

with col2:
    st.subheader("Sales Distribution")
    st.plotly_chart(px.histogram(filtered_df,x="Sales",nbins=30),use_container_width=True)

st.subheader("Filtered Dataset")
st.dataframe(filtered_df,use_container_width=True)

csv=filtered_df.to_csv(index=False).encode()
st.download_button("📥 Download Filtered Data",csv,"filtered_data.csv","text/csv")

st.divider()
st.markdown("**Developed by Aarsh Pavashiya | AI & Data Science Internship Project**")

st.markdown("---")
st.subheader("Filtered Dataset")
st.dataframe(filtered_df)

st.markdown("---")
st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    filtered_df.groupby(
        pd.Grouper(
            key="Order Date",
            freq="M"
        )
    )["Sales"]
    .sum()
    .reset_index()
)

fig = px.line(
    monthly_sales,
    x="Order Date",
    y="Sales",
    markers=True,
    title="Monthly Sales Trend"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("📦 Sales by Category")

category_sales = (
    filtered_df.groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

fig = px.bar(
    category_sales,
    x="Category",
    y="Sales",
    color="Category",
    title="Category-wise Sales"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🌍 Sales by Region")

region_sales = (
    filtered_df.groupby("Region")["Sales"]
    .sum()
    .reset_index()
)

fig = px.pie(
    region_sales,
    names="Region",
    values="Sales",
    hole=0.4,
    title="Regional Sales Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("👥 Sales by Segment")

segment_sales = (
    filtered_df.groupby("Segment")["Sales"]
    .sum()
    .reset_index()
)

fig = px.bar(
    segment_sales,
    x="Segment",
    y="Sales",
    color="Segment",
    title="Customer Segment Sales"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🏆 Top 10 Products")

top_products = (
    filtered_df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(
    top_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    color="Sales",
    title="Top Selling Products"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("📊 Sales Distribution")

fig = px.histogram(
    filtered_df,
    x="Sales",
    nbins=30,
    title="Distribution of Sales"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🔥 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include="number")

fig = px.imshow(
    numeric_df.corr(),
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

csv = filtered_df.to_csv(index=False)

st.download_button(
    "⬇ Download Filtered Dataset",
    csv,
    "Filtered_Data.csv",
    "text/csv"
)

# ==========================
# Forecasting Section
# ==========================

from pathlib import Path

st.markdown("---")
st.header("🔮 Sales Forecasting")

results_folder = Path("results")

comparison = None
future_forecast = None
forecast_results = None

comparison_path = results_folder / "model_comparison.csv"
future_path = results_folder / "future_forecast.csv"
forecast_path = results_folder / "forecast_results.csv"

if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)

if future_path.exists():
    future_forecast = pd.read_csv(future_path)

if forecast_path.exists():
    forecast_results = pd.read_csv(forecast_path)
if comparison is not None:

    st.subheader("📊 Model Comparison")

    st.dataframe(comparison, use_container_width=True)

    metric = st.selectbox(
        "Select Metric",
        ["MAE","RMSE","MAPE"]
    )

    fig = px.bar(
        comparison,
        x="Model",
        y=metric,
        color="Model",
        text_auto=".2f",
        title=f"{metric} Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

if future_forecast is not None:

    st.markdown("---")

    st.subheader("📅 Future Sales Forecast")

    st.dataframe(
        future_forecast,
        use_container_width=True
    )

    date_col = future_forecast.columns[0]
    sales_col = future_forecast.columns[1]

    fig = px.line(
        future_forecast,
        x=date_col,
        y=sales_col,
        markers=True,
        title="Future Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

if forecast_results is not None:

    st.markdown("---")

    st.subheader("📈 Forecast Results")

    st.dataframe(
        forecast_results,
        use_container_width=True
    )

    numeric = forecast_results.select_dtypes(
        include="number"
    )

    if numeric.shape[1] >= 2:

        fig = px.scatter(
            forecast_results,
            x=numeric.columns[0],
            y=numeric.columns[1],
            title="Forecast Scatter Plot"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

st.subheader("⬇ Download Results")

if comparison is not None:

    st.download_button(
        "Download Model Comparison",
        comparison.to_csv(index=False),
        "model_comparison.csv",
        "text/csv"
    )

if future_forecast is not None:

    st.download_button(
        "Download Future Forecast",
        future_forecast.to_csv(index=False),
        "future_forecast.csv",
        "text/csv"
    )

if forecast_results is not None:

    st.download_button(
        "Download Forecast Results",
        forecast_results.to_csv(index=False),
        "forecast_results.csv",
        "text/csv"
    )

st.markdown("---")

st.header("📌 Business Summary")

st.success("""
✔ Prophet achieved the best forecasting performance.

✔ Technology is the highest revenue generating category.

✔ West Region contributes the highest sales.

✔ Sales peak during November and December.

✔ Business demand remains stable with no major anomalies.

✔ High-demand products should receive inventory priority.
""")

st.markdown("---")

st.markdown(
"""
### 👨‍💻 Developed by Aarsh Pavashiya

AI & Data Science Internship Project

Sales Forecasting & Business Intelligence Dashboard
"""
)