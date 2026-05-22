import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration FIRST
st.set_page_config(
    page_title="QSAR Drug Target Analyzer",
    page_icon="🧬",
    layout="wide"
)

# ----------------------------------------
# DATA LOADING (Optimized Caching)
# ----------------------------------------
@st.cache_data
def load_optimized_data():
    # Reads the lightweight parquet files instead of raw heavy CSV
    # These files should be uploaded to your GitHub repository along with the script
    df = pd.read_parquet("qsar_clean.parquet")
    long_df = pd.read_parquet("qsar_long.parquet")
    drug_wide = pd.read_parquet("qsar_ranked.parquet")
    return df, long_df, drug_wide

try:
    df, long_df, drug_wide = load_optimized_data()
except FileNotFoundError:
    st.error("⚠️ Optimized Parquet data files not found. Please ensure 'qsar_clean.parquet', 'qsar_long.parquet', and 'qsar_ranked.parquet' are in your repository.")
    st.stop()

molecule_col = df.columns[0]
numeric_cols = df.select_dtypes(include="number").columns.tolist()
all_molecules = df[molecule_col].unique().tolist()

# ----------------------------------------
# TITLE & INTRODUCTION
# ----------------------------------------
st.title("🧬 QSAR Drug Target Activity Analyzer")
st.markdown("""
**Welcome to the QSAR Data Dashboard.** This application is designed to analyze Quantitative Structure-Activity Relationship (QSAR) datasets, specifically focusing on how different molecules interact with various drug targets.
""")

# Sidebar Filters
st.sidebar.header("Setup & Filters")
selected_molecule = st.sidebar.selectbox("Select a Specific Molecule to Focus On:", ["All"] + all_molecules)

# Filter main dataset based on selection
filtered_df = df if selected_molecule == "All" else df[df[molecule_col] == selected_molecule]

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
    st.metric("Dataset Rows", df.shape[0])
with col4:
    st.metric("Current Filter View", selected_molecule)

with st.expander("View Raw Cleaned Data Summary"):
    st.dataframe(filtered_df.describe(), use_container_width=True)

# ----------------------------------------
# DATA VISUALIZATION
# ----------------------------------------
st.header("📈 Data Visualization & Distribution")
tab1, tab2 = st.tabs(["Univariate Analysis", "Bivariate Analysis"])

with tab1:
    st.subheader("Distribution of Primary Metric")
    if len(numeric_cols) > 0:
        primary_metric = st.selectbox("Select metric to plot distribution:", numeric_cols, index=0)
        fig_hist = px.histogram(
            filtered_df, 
            x=primary_metric, 
            marginal="rug",  
            title=f"Distribution of {primary_metric}",
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("No numeric columns available for distribution analysis.")

with tab2:
    st.subheader("Feature Relationships")
    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Select X-axis metric:", numeric_cols, index=0)
        y_axis = st.selectbox("Select Y-axis metric:", numeric_cols, index=min(1, len(numeric_cols)-1))
        
        fig_scatter = px.scatter(
            filtered_df, 
            x=x_axis, 
            y=y_axis, 
            hover_name=molecule_col,
            title=f"{x_axis} vs {y_axis}",
            color_discrete_sequence=['#ff7f0e']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Need at least 2 numeric features to display a scatter relationship.")

# ----------------------------------------
# DRUG RANKING SYSTEM (Pre-calculated fast display)
# ----------------------------------------
st.header("🏆 Drug Targets Ranked By Activity Score")
st.markdown("This section orders the tested drug targets from highest score (Rank 1) to lowest score for each molecule.")

# Display rankings based on selection
if selected_molecule != "All":
    display_ranking = drug_wide[drug_wide[molecule_col] == selected_molecule]
else:
    display_ranking = drug_wide

st.dataframe(display_ranking, use_container_width=True)

# Download button uses pre-calculated wide dataframe directly
@st.cache_data
def convert_df(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')

csv_data = convert_df(drug_wide)
st.download_button(
    label="📥 Download Complete Ranked Drugs CSV",
    data=csv_data,
    file_name="qsar_ranked_drugs.csv",
    mime="text/csv"
)

# ----------------------------------------
# INTERACTIVE MOLECULE QUERY FORM
# ----------------------------------------
st.header("🔍 Molecule Activity Inspector")
st.markdown("Select a specific molecule below to see a sorted breakdown of its ideal drug targets alongside their exact calculated values.")

selected_inspect = st.selectbox("Inspect specific molecule details:", all_molecules)
molecule_data = long_df[long_df[molecule_col] == selected_inspect].sort_values("Rank")

if not molecule_data.empty:
    fig_bar = px.bar(
        molecule_data.head(20), # Limit to top 20 for faster visualization rendering
        x="Score",
        y="Drug",
        orientation='h',
        color="Score",
        text="Rank",
        title=f"Top Drug Target Activity Hierarchy for Molecule: {selected_inspect}",
        color_continuous_scale="Viridis"
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.table(molecule_data)
else:
    st.info("No active data entries found for this molecule.")
