import streamlit as pl
import polars as pl
import plotly.express as px
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="QSAR Drug Target Analyzer",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 QSAR Drug Target Activity Analyzer")
st.markdown("""
**Welcome to the QSAR Data Dashboard.** This application is fully optimized to read binary Parquet datasets.
""")

st.sidebar.header("Setup & Filters")

# ----------------------------------------
# ULTRA-FAST DATA LOADING
# ----------------------------------------
@st.cache_data
def load_parquet_data(url):
    try:
        # Parquet files natively store data types and compress memory
        return pl.read_parquet(url)
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        return None

# CHANGE THIS URL to your new uploaded QSAR.parquet file link!
PARQUET_URL = "https://github.com/morlockmaimai-ui/Drug-Target-Activity-Analyzer-/releases/download/v1.0.0/QSAR.parquet"

with st.spinner("Streaming compressed Parquet dataset... This is 10x lighter on memory!"):
    df = load_parquet_data(PARQUET_URL)

if df is None:
    st.error("⚠️ Failed to load Parquet data. Please double-check your asset link format.")
    st.stop()

molecule_col = df.columns[0]
numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]

# Sidebar Filter
all_molecules = df[molecule_col].unique().to_list()
selected_molecule = st.sidebar.selectbox("Select a Specific Molecule to Focus On:", ["All"] + all_molecules)

if selected_molecule != "All":
    filtered_df = df.filter(pl.col(molecule_col) == selected_molecule)
else:
    filtered_df = df

# ----------------------------------------
# METRICS & KEY STATISTICS
# ----------------------------------------
st.header("📊 Key Statistics & Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Unique Molecules", len(all_molecules))
with col2:
    st.metric("Total Drug Targets Tested", len(numeric_cols))
with col3:
    st.metric("Dataset Rows", df.height)
with col4:
    st.metric("Current Filter View", selected_molecule)

with st.expander("View Data Summary"):
    st.dataframe(filtered_df.describe().to_pandas(), use_container_width=True)

# ----------------------------------------
# DATA VISUALIZATION (Memory Safe)
# ----------------------------------------
st.header("📈 Data Visualization & Distribution")
tab1, tab2 = st.tabs(["Univariate Analysis", "Bivariate Analysis"])

MAX_PLOT_ROWS = 15000
if filtered_df.height > MAX_PLOT_ROWS:
    plot_df = filtered_df.sample(n=MAX_PLOT_ROWS, seed=42)
    st.caption(f"⚠️ Plot dynamically downsampled to {MAX_PLOT_ROWS} rows for performance stability.")
else:
    plot_df = filtered_df

with tab1:
    st.subheader("Distribution of Primary Metric")
    if len(numeric_cols) > 0:
        primary_metric = st.selectbox("Select metric to plot distribution:", numeric_cols, index=0)
        fig_hist = px.histogram(
            plot_df.select([primary_metric]).to_pandas(), 
            x=primary_metric, 
            title=f"Distribution of {primary_metric}"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.subheader("Feature Relationships")
    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Select X-axis metric:", numeric_cols, index=0)
        y_axis = st.selectbox("Select Y-axis metric:", numeric_cols, index=min(1, len(numeric_cols)-1))
        
        fig_scatter = px.scatter(
            plot_df.select([molecule_col, x_axis, y_axis]).to_pandas(), 
            x=x_axis, 
            y=y_axis, 
            hover_name=molecule_col,
            title=f"{x_axis} vs {y_axis}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------------------------------
# DRUG RANKING SYSTEM
# ----------------------------------------
st.header("🏆 Drug Targets Ranked By Activity Score")

long_df_pl = (
    filtered_df.unpivot(index=molecule_col, on=numeric_cols, variable_name="Drug", value_name="Score")
    .drop_nulls(subset=["Score"])
    .sort([molecule_col, "Score"], descending=[False, True])
)

long_df_pl = long_df_pl.with_columns(
    pl.int_range(1, pl.len() + 1).over(molecule_col).alias("Rank")
)

if filtered_df.height < 5000:
    drug_wide_pl = (
        long_df_pl.with_columns(pl.format("Rank_{}", pl.col("Rank")).alias("Rank_Str"))
        .pivot(on="Rank_Str", index=molecule_col, values="Drug")
    )
    orig_order = filtered_df.select(molecule_col)
    drug_wide_pl = orig_order.join(drug_wide_pl, on=molecule_col, how="left")
    
    st.download_button(
        label="📥 Download Ranked Drugs CSV",
        data=drug_wide_pl.write_csv().encode('utf-8'),
        file_name="qsar_ranked_drugs.csv",
        mime="text/csv"
    )
else:
    st.warning("⚠️ Full calculation table blocked because your active workspace data is too large. Filter your molecule focus on the sidebar to unlock download features.")

# ----------------------------------------
# INTERACTIVE MOLECULE QUERY FORM
# ----------------------------------------
st.header("🔍 Molecule Activity Inspector")
selected_inspect = st.selectbox("Inspect specific molecule details:", all_molecules)

molecule_data_pd = long_df_pl.filter(pl.col(molecule_col) == selected_inspect).sort("Rank").to_pandas()

if not molecule_data_pd.empty:
    fig_bar = px.bar(
        molecule_data_pd,
        x="Score", y="Drug",
        orientation='h', color="Score", text="Rank",
        title=f"Drug Target Activity Hierarchy for Molecule: {selected_inspect}",
        color_continuous_scale="Viridis"
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)
    st.table(molecule_data_pd[["Rank", "Drug", "Score"]])
