import streamlit as st
from data_loader import load_data
from visualizations import create_executive_dashboard, create_thematic_area_tab
from forecasting import create_forecasting_tab

# Set page config
st.set_page_config(page_title="Kenya GDP Analysis Dashboard", layout="wide")

# Define thematic areas and their sectors (updated to match cleaned column names)
THEMATIC_AREAS = {
    "Primary Industries": [
        "Agriculture", "Mining & Quarrying", "Construction", "Electricity & Water Supply"
    ],
    "Secondary Industries": [
        "Manufacturing", "Electricity & Water Supply", "Construction", "Transportation And Storage"
    ],
    "Tertiary Industries": [
        "Wholesale And Retail Trade", "Accommodation And Food Service Activities",
        "Transportation And Storage", "Information And Communication",
        "Financial And Insurance", "Real Estate", "Professional, Admin & Support Services",
        "Public Administration", "Health", "Education"
    ],
    "Government and Public Services": [
        "Public Administration", "Education", "Health", "Professional, Admin & Support Services",
        "Other Services"
    ],
    "Digital Economy and Communication": [
        "Information And Communication", "Financial And Insurance",
        "Professional, Admin & Support Services"
    ],
    "Macroeconomic Indicators": [
        "All Industries At Basic Prices", "Taxes On Product", "Gdp At Market Prices",
        "Gdp, Seasonally Adjusted", "Fisim"
    ]
}

# Main app
def main():
    st.title("Kenya Real Quarterly GDP Analysis (2013–2024)")

    # Load data
    df = load_data()
    if df is None:
        st.write("Please ensure the Excel file has columns 'Year' and 'Quarter' and try again.")
        return

    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Thematic Area selection
    thematic_area = st.sidebar.selectbox("Select Thematic Area", list(THEMATIC_AREAS.keys()))
    
    # Years selection with "Select All" option
    years = sorted(df['Year'].unique())
    select_all_years = st.sidebar.checkbox("Select All Years", value=False)
    if select_all_years:
        selected_years = years
    else:
        selected_years = st.sidebar.multiselect("Select Years", years, default=years[-3:])
    
    # Sectors selection with "Select All" option
    select_all_sectors = st.sidebar.checkbox("Select All Sectors", value=False)
    if select_all_sectors:
        selected_sectors = THEMATIC_AREAS[thematic_area]
    else:
        selected_sectors = st.sidebar.multiselect(
            "Select Sectors", THEMATIC_AREAS[thematic_area], default=THEMATIC_AREAS[thematic_area][:2]
        )

    # Quarter selection for interactivity
    quarters = sorted(df['Year_Quarter'].unique())
    if 'selected_quarter' not in st.session_state:
        st.session_state.selected_quarter = quarters[-1]  # Default to latest quarter
    selected_quarter = st.sidebar.selectbox(
        "Highlight Quarter (Affects All Charts)",
        quarters,
        index=quarters.index(st.session_state.selected_quarter),
        key="quarter_selector"
    )
    st.session_state.selected_quarter = selected_quarter

    # Filter data
    filtered_df = df[df['Year'].isin(selected_years)]
    if not selected_sectors:
        selected_sectors = THEMATIC_AREAS[thematic_area]

    # Tabs for navigation
    tabs = ["Executive Dashboard"] + list(THEMATIC_AREAS.keys()) + ["Forecasting", "Data Preview"]
    selected_tab = st.tabs(tabs)

    # Executive Dashboard
    with selected_tab[0]:
        create_executive_dashboard(filtered_df, selected_quarter)

    # Thematic Area Tabs
    for i, area in enumerate(THEMATIC_AREAS.keys(), 1):
        with selected_tab[i]:
            create_thematic_area_tab(area, filtered_df, selected_sectors, selected_quarter)

    # Forecasting Tab
    with selected_tab[-2]:  # Second to last tab (Forecasting)
        create_forecasting_tab(df)

    # Data Preview Tab
    with selected_tab[-1]:
        st.header("Data Preview")
        st.write(filtered_df)
        st.subheader("Summary Statistics")
        st.write(filtered_df.describe())

if __name__ == "__main__":
    main()