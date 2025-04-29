import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

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

# Function to load data
@st.cache_data
def load_data():
    try:
        # Try reading Excel file
        try:
            df = pd.read_excel(
                "Real Quarterly GDP, 2013 -2024.....xlsx",
                sheet_name=0,  # First sheet
                skiprows=1,    # Skip title row
                header=0       # Use second row as header
            )
            # Stop reading at metadata
            df = df.loc[:df.index[df['Year'].isna() & df['Quarter'].isna()].min() - 1]
        except FileNotFoundError:
            st.warning("Excel file not found. Trying CSV fallback...")
            df = pd.read_csv("2025-04-29T06-46_export.csv")
        
        # Clean column names: strip spaces, remove backticks, fix typos
        df.columns = (df.columns.str.strip()
                      .str.replace('`', '')
                      .str.replace('Public Admnistration', 'Public Administration')
                      .str.title())
        
        # Check for required columns
        required_cols = ['Year', 'Quarter']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            similar_cols = {col: [c for c in df.columns if col.lower() in c.lower()] for col in missing_cols}
            st.write("Possible similar columns:", similar_cols)
            return None
        
        # Forward-fill Year column
        df['Year'] = df['Year'].fillna(method='ffill')
        
        # Convert Year to integer
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype(int)
        
        # Create a Year_Quarter column for plotting
        df['Year_Quarter'] = df['Year'].astype(str) + ' ' + df['Quarter']
        
        # Drop rows with missing Quarter or mostly missing data (e.g., Q4 2024)
        df = df.dropna(subset=['Quarter'])
        df = df.dropna(thresh=10)  # Keep rows with at least 10 non-NA values
        
        # Ensure numeric columns are properly typed
        numeric_cols = df.columns.drop(['Year', 'Quarter', 'Year_Quarter'])
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Create a datetime column for potential sorting (not used in plots)
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Quarter'].str.replace('Q', '') + '-01')
        
        # Sort by Date to ensure chronological order
        df = df.sort_values('Date')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to generate insights
def generate_insights(df, thematic_area, selected_sectors):
    insights = []
    if thematic_area == "Primary Industries":
        insights.append("Agriculture shows strong seasonality, with Q3 typically lower due to harvest cycles.")
        insights.append("Construction and Mining sectors exhibit boom/bust cycles tied to infrastructure projects.")
    elif thematic_area == "Secondary Industries":
        insights.append("Manufacturing growth correlates with Electricity & Water Supply, indicating energy dependency.")
        insights.append("Transportation and Storage reflect infrastructure bottlenecks in peak quarters.")
    elif thematic_area == "Tertiary Industries":
        insights.append("Service sectors like Financial and Insurance drive GDP growth, with consistent contributions.")
        insights.append("Accommodation and Food Services show volatility tied to tourism seasons.")
    elif thematic_area == "Government and Public Services":
        insights.append("Public Administration provides stable GDP contributions, with Education and Health growing steadily.")
    elif thematic_area == "Digital Economy and Communication":
        insights.append("Information and Communication sector shows rapid growth, reflecting Kenya's digital economy expansion.")
    elif thematic_area == "Macroeconomic Indicators":
        insights.append("GDP at Market Prices shows steady growth, with Taxes on Product contributing significantly.")
        insights.append("FISIM (Financial Intermediation) impacts GDP negatively, reflecting banking costs.")
    return insights

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
    tabs = ["Executive Dashboard"] + list(THEMATIC_AREAS.keys()) + ["Data Preview"]
    selected_tab = st.tabs(tabs)

    # Executive Dashboard
    with selected_tab[0]:
        st.header("Executive Dashboard")
        st.write("Key GDP Trends and Highlights")
        # Line chart with highlighted quarter
        fig_gdp = go.Figure()
        fig_gdp.add_trace(go.Scatter(
            x=filtered_df['Year_Quarter'],
            y=filtered_df['Gdp At Market Prices'],
            mode='lines+markers',
            name='GDP at Market Prices',
            line=dict(color='blue'),
            opacity=0.3
        ))
        # Highlight selected quarter
        highlight_df = filtered_df[filtered_df['Year_Quarter'] == selected_quarter]
        if not highlight_df.empty:
            fig_gdp.add_trace(go.Scatter(
                x=highlight_df['Year_Quarter'],
                y=highlight_df['Gdp At Market Prices'],
                mode='markers',
                name=f'Highlighted: {selected_quarter}',
                marker=dict(color='red', size=12, symbol='circle'),
                opacity=1
            ))
        fig_gdp.update_layout(
            title="GDP at Market Prices (2013–2024)",
            xaxis_title="Quarter",
            yaxis_title="GDP (KES Million)",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig_gdp, use_container_width=True, key="executive_gdp_line_chart")
        st.write("**Highlight**: GDP growth has been steady, with a notable increase in service sectors like Information and Communication.")

    # Thematic Area Tabs
    for i, area in enumerate(THEMATIC_AREAS.keys(), 1):
        with selected_tab[i]:
            st.header(area)
            st.write("**Insights**")
            insights = generate_insights(filtered_df, area, selected_sectors)
            for insight in insights:
                st.write(f"- {insight}")

            # Charts
            if area == "Primary Industries":
                # Line Chart: GDP trends per sector with highlighted quarter
                fig_line = go.Figure()
                for sector in selected_sectors:
                    fig_line.add_trace(go.Scatter(
                        x=filtered_df['Year_Quarter'],
                        y=filtered_df[sector],
                        mode='lines',
                        name=sector,
                        opacity=0.3
                    ))
                    # Highlight selected quarter
                    highlight_df = filtered_df[filtered_df['Year_Quarter'] == selected_quarter]
                    if not highlight_df.empty:
                        fig_line.add_trace(go.Scatter(
                            x=highlight_df['Year_Quarter'],
                            y=highlight_df[sector],
                            mode='markers',
                            name=f'{sector} ({selected_quarter})',
                            marker=dict(size=12, symbol='circle'),
                            opacity=1,
                            showlegend=False
                        ))
                fig_line.update_layout(
                    title="Sector GDP Trends Over Time",
                    xaxis_title="Quarter",
                    yaxis_title="GDP (KES Million)",
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_line, use_container_width=True, key=f"{area}_line_chart")

                # Area Chart: Cumulative contribution with highlighted quarter
                fig_area = go.Figure()
                for sector in selected_sectors:
                    fig_area.add_trace(go.Scatter(
                        x=filtered_df['Year_Quarter'],
                        y=filtered_df[sector],
                        mode='lines',
                        stackgroup='one',
                        name=sector,
                        opacity=0.3
                    ))
                    # Highlight selected quarter
                    highlight_df = filtered_df[filtered_df['Year_Quarter'] == selected_quarter]
                    if not highlight_df.empty:
                        fig_area.add_trace(go.Scatter(
                            x=highlight_df['Year_Quarter'],
                            y=highlight_df[sector],
                            mode='markers',
                            name=f'{sector} ({selected_quarter})',
                            marker=dict(size=12, symbol='circle'),
                            opacity=1,
                            showlegend=False
                        ))
                fig_area.update_layout(
                    title="Cumulative Sector Contribution",
                    xaxis_title="Quarter",
                    yaxis_title="GDP (KES Million)",
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_area, use_container_width=True, key=f"{area}_area_chart")

                # Bar Chart: Sector contributions with highlighted quarter
                st.subheader(f"Sector Contributions Across Quarters")
                bar_layout = st.radio(
                    "Bar Chart Layout", ["Stacked", "Clustered"], index=0, key=f"bar_layout_{area}"
                )
                fig_bar = go.Figure()
                for sector in selected_sectors:
                    if bar_layout == "Stacked":
                        fig_bar.add_trace(go.Bar(
                            x=filtered_df['Year_Quarter'],
                            y=filtered_df[sector],
                            name=sector,
                            opacity=0.3 if filtered_df['Year_Quarter'].iloc[0] != selected_quarter else 1
                        ))
                    else:
                        fig_bar.add_trace(go.Bar(
                            x=filtered_df['Year_Quarter'],
                            y=filtered_df[sector],
                            name=sector,
                            opacity=0.3 if filtered_df['Year_Quarter'].iloc[0] != selected_quarter else 1
                        ))
                # Highlight selected quarter
                highlight_df = filtered_df[filtered_df['Year_Quarter'] == selected_quarter]
                if not highlight_df.empty:
                    for sector in selected_sectors:
                        if bar_layout == "Stacked":
                            fig_bar.add_trace(go.Bar(
                                x=highlight_df['Year_Quarter'],
                                y=highlight_df[sector],
                                name=f'{sector} ({selected_quarter})',
                                opacity=1,
                                showlegend=False
                            ))
                        else:
                            fig_bar.add_trace(go.Bar(
                                x=highlight_df['Year_Quarter'],
                                y=highlight_df[sector],
                                name=f'{sector} ({selected_quarter})',
                                opacity=1,
                                showlegend=False
                            ))
                fig_bar.update_layout(
                    title=f"Sector Contributions (All Quarters)",
                    xaxis_title="Quarter",
                    yaxis_title="GDP (KES Million)",
                    barmode='stack' if bar_layout == "Stacked" else 'group',
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_bar, use_container_width=True, key=f"{area}_bar_chart")

            elif area == "Secondary Industries":
                # Dual-axis Chart: Manufacturing vs Electricity (static)
                if "Manufacturing" in selected_sectors and "Electricity & Water Supply" in selected_sectors:
                    fig, ax1 = plt.subplots()
                    ax1.set_xlabel('Quarter')
                    ax1.set_ylabel('Manufacturing (KES Million)', color='tab:blue')
                    ax1.plot(filtered_df['Year_Quarter'], filtered_df['Manufacturing'], color='tab:blue')
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Electricity & Water Supply (KES Million)', color='tab:orange')
                    ax2.plot(filtered_df['Year_Quarter'], filtered_df['Electricity & Water Supply'], color='tab:orange')
                    plt.title("Manufacturing vs Electricity & Water Supply")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)

                # Heatmap: Quarter-year performance (static)
                pivot_data = filtered_df.pivot_table(index='Year', columns='Quarter', values=selected_sectors[0] if selected_sectors else "Manufacturing")
                fig_heatmap, ax = plt.subplots()
                sns.heatmap(pivot_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
                plt.title(f"{selected_sectors[0] if selected_sectors else 'Manufacturing'} by Quarter and Year")
                st.pyplot(fig_heatmap)

                # Bar Chart: Sector contributions with highlighted quarter
                st.subheader(f"Sector Contributions Across Quarters")
                bar_layout = st.radio(
                    "Bar Chart Layout", ["Stacked", "Clustered"], index=0, key=f"bar_layout_{area}"
                )
                fig_bar = go.Figure()
                for sector in selected_sectors:
                    if bar_layout == "Stacked":
                        fig_bar.add_trace(go.Bar(
                            x=filtered_df['Year_Quarter'],
                            y=filtered_df[sector],
                            name=sector,
                            opacity=0.3 if filtered_df['Year_Quarter'].iloc[0] != selected_quarter else 1
                        ))
                    else:
                        fig_bar.add_trace(go.Bar(
                            x=filtered_df['Year_Quarter'],
                            y=filtered_df[sector],
                            name=sector,
                            opacity=0.3 if filtered_df['Year_Quarter'].iloc[0] != selected_quarter else 1
                        ))
                # Highlight selected quarter
                highlight_df = filtered_df[filtered_df['Year_Quarter'] == selected_quarter]
                if not highlight_df.empty:
                    for sector in selected_sectors:
                        if bar_layout == "Stacked":
                            fig_bar.add_trace(go.Bar(
                                x=highlight_df['Year_Quarter'],
                                y=highlight_df[sector],
                                name=f'{sector} ({selected_quarter})',
                                opacity=1,
                                showlegend=False
                            ))
                        else:
                            fig_bar.add_trace(go.Bar(
                                x=highlight_df['Year_Quarter'],
                                y=highlight_df[sector],
                                name=f'{sector} ({selected_quarter})',
                                opacity=1,
                                showlegend=False
                            ))
                fig_bar.update_layout(
                    title=f"Sector Contributions (All Quarters)",
                    xaxis_title="Quarter",
                    yaxis_title="GDP (KES Million)",
                    barmode='stack' if bar_layout == "Stacked" else 'group',
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_bar, use_container_width=True, key=f"{area}_bar_chart")

            elif area == "Tertiary Industries":
                # Stacked Area Chart with highlighted quarter
                fig_stacked = go.Figure()
                for sector in selected_sectors:
                    fig_stacked.add_trace(go.Scatter(
                        x=filtered_df['Year_Quarter'],
                        y=filtered_df[sector],
                        mode='lines',
                        stackgroup='one',
                        name=sector,
                        opacity=0.3
                    ))
                    # Highlight selected quarter
                    highlight_df = filtered_df[filtered_df['Year_Quarter'] == selected_quarter]
                    if not highlight_df.empty:
                        fig_stacked.add_trace(go.Scatter(
                            x=highlight_df['Year_Quarter'],
                            y=highlight_df[sector],
                            mode='markers',
                            name=f'{sector} ({selected_quarter})',
                            marker=dict(size=12, symbol='circle'),
                            opacity=1,
                            showlegend=False
                        ))
                fig_stacked.update_layout(
                    title="Growth of Service Sectors",
                    xaxis_title="Quarter",
                    yaxis_title="GDP (KES Million)",
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_stacked, use_container_width=True, key=f"{area}_stacked_area_chart")

                # Treemap: Proportion of sectors
                latest_quarter = filtered_df.iloc[-1]
                treemap_data = pd.DataFrame({
                    "Sector": selected_sectors,
                    "Value": [latest_quarter[sector] for sector in selected_sectors]
                })
                fig_treemap = px.treemap(
                    treemap_data, path=['Sector'], values='Value',
                    title="Proportion of Service Sectors in Latest Quarter"
                )
                st.plotly_chart(fig_treemap, use_container_width=True, key=f"{area}_treemap")

                # Bar Chart: Sector contributions with highlighted quarter
                st.subheader(f"Sector Contributions Across Quarters")
                bar_layout = st.radio(
                    "Bar Chart Layout", ["Stacked", "Clustered"], index=0, key=f"bar_layout_{area}"
                )
                fig_bar = go.Figure()
                for sector in selected_sectors:
                    if bar_layout == "Stacked":
                        fig_bar.add_trace(go.Bar(
                            x=filtered_df['Year_Quarter'],
                            y=filtered_df[sector],
                            name=sector,
                            opacity=0.3 if filtered_df['Year_Quarter'].iloc[0] != selected_quarter else 1
                        ))
                    else:
                        fig_bar.add_trace(go.Bar(
                            x=filtered_df['Year_Quarter'],
                            y=filtered_df[sector],
                            name=sector,
                            opacity=0.3 if filtered_df['Year_Quarter'].iloc[0] != selected_quarter else 1
                        ))
                # Highlight selected quarter
                highlight_df = filtered_df[filtered_df['Year_Quarter'] == selected_quarter]
                if not highlight_df.empty:
                    for sector in selected_sectors:
                        if bar_layout == "Stacked":
                            fig_bar.add_trace(go.Bar(
                                x=highlight_df['Year_Quarter'],
                                y=highlight_df[sector],
                                name=f'{sector} ({selected_quarter})',
                                opacity=1,
                                showlegend=False
                            ))
                        else:
                            fig_bar.add_trace(go.Bar(
                                x=highlight_df['Year_Quarter'],
                                y=highlight_df[sector],
                                name=f'{sector} ({selected_quarter})',
                                opacity=1,
                                showlegend=False
                            ))
                fig_bar.update_layout(
                    title=f"Sector Contributions (All Quarters)",
                    xaxis_title="Quarter",
                    yaxis_title="GDP (KES Million)",
                    barmode='stack' if bar_layout == "Stacked" else 'group',
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_bar, use_container_width=True, key=f"{area}_bar_chart")

            elif area == "Government and Public Services":
                # Pie Chart: Public services share
                latest_quarter = filtered_df.iloc[-1]
                pie_data = pd.DataFrame({
                    "Sector": selected_sectors,
                    "Value": [latest_quarter[sector] for sector in selected_sectors]
                })
                fig_pie = px.pie(
                    pie_data, names='Sector', values='Value',
                    title="Public Services Share in Latest Quarter"
                )
                st.plotly_chart(fig_pie, use_container_width=True, key=f"{area}_pie_chart")

                # Bar Chart: Sector contributions with highlighted quarter
                st.subheader(f"Sector Contributions Across Quarters")
                bar_layout = st.radio(
                    "Bar Chart Layout", ["Stacked", "Clustered"], index=0, key=f"bar_layout_{area}"
                )
                fig_bar = go.Figure()
                for sector in selected_sectors:
                    if bar_layout == "Stacked":
                        fig_bar.add_trace(go.Bar(
                            x=filtered_df['Year_Quarter'],
                            y=filtered_df[sector],
                            name=sector,
                            opacity=0.3 if filtered_df['Year_Quarter'].iloc[0] != selected_quarter else 1
                        ))
                    else:
                        fig_bar.add_trace(go.Bar(
                            x=filtered_df['Year_Quarter'],
                            y=filtered_df[sector],
                            name=sector,
                            opacity=0.3 if filtered_df['Year_Quarter'].iloc[0] != selected_quarter else 1
                        ))
                # Highlight selected quarter
                highlight_df = filtered_df[filtered_df['Year_Quarter'] == selected_quarter]
                if not highlight_df.empty:
                    for sector in selected_sectors:
                        if bar_layout == "Stacked":
                            fig_bar.add_trace(go.Bar(
                                x=highlight_df['Year_Quarter'],
                                y=highlight_df[sector],
                                name=f'{sector} ({selected_quarter})',
                                opacity=1,
                                showlegend=False
                            ))
                        else:
                            fig_bar.add_trace(go.Bar(
                                x=highlight_df['Year_Quarter'],
                                y=highlight_df[sector],
                                name=f'{sector} ({selected_quarter})',
                                opacity=1,
                                showlegend=False
                            ))
                fig_bar.update_layout(
                    title=f"Sector Contributions (All Quarters)",
                    xaxis_title="Quarter",
                    yaxis_title="GDP (KES Million)",
                    barmode='stack' if bar_layout == "Stacked" else 'group',
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_bar, use_container_width=True, key=f"{area}_bar_chart")

            elif area == "Digital Economy and Communication":
                # Scatter Plot: ICT vs GDP
                if "Information And Communication" in selected_sectors:
                    fig_scatter = px.scatter(
                        filtered_df, x="Information And Communication", y="Gdp At Market Prices",
                        title="ICT vs GDP at Market Prices",
                        labels={"Information And Communication": "ICT (KES Million)", "Gdp At Market Prices": "GDP (KES Million)"}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True, key=f"{area}_scatter_chart")

                # Bar Chart: Sector contributions with highlighted quarter
                st.subheader(f"Sector Contributions Across Quarters")
                bar_layout = st.radio(
                    "Bar Chart Layout", ["Stacked", "Clustered"], index=0, key=f"bar_layout_{area}"
                )
                fig_bar = go.Figure()
                for sector in selected_sectors:
                    if bar_layout == "Stacked":
                        fig_bar.add_trace(go.Bar(
                            x=filtered_df['Year_Quarter'],
                            y=filtered_df[sector],
                            name=sector,
                            opacity=0.3 if filtered_df['Year_Quarter'].iloc[0] != selected_quarter else 1
                        ))
                    else:
                        fig_bar.add_trace(go.Bar(
                            x=filtered_df['Year_Quarter'],
                            y=filtered_df[sector],
                            name=sector,
                            opacity=0.3 if filtered_df['Year_Quarter'].iloc[0] != selected_quarter else 1
                        ))
                # Highlight selected quarter
                highlight_df = filtered_df[filtered_df['Year_Quarter'] == selected_quarter]
                if not highlight_df.empty:
                    for sector in selected_sectors:
                        if bar_layout == "Stacked":
                            fig_bar.add_trace(go.Bar(
                                x=highlight_df['Year_Quarter'],
                                y=highlight_df[sector],
                                name=f'{sector} ({selected_quarter})',
                                opacity=1,
                                showlegend=False
                            ))
                        else:
                            fig_bar.add_trace(go.Bar(
                                x=highlight_df['Year_Quarter'],
                                y=highlight_df[sector],
                                name=f'{sector} ({selected_quarter})',
                                opacity=1,
                                showlegend=False
                            ))
                fig_bar.update_layout(
                    title=f"Sector Contributions (All Quarters)",
                    xaxis_title="Quarter",
                    yaxis_title="GDP (KES Million)",
                    barmode='stack' if bar_layout == "Stacked" else 'group',
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_bar, use_container_width=True, key=f"{area}_bar_chart")

            elif area == "Macroeconomic Indicators":
                # Bar Chart: Contribution to GDP in Latest Quarter
                latest_quarter = filtered_df.iloc[-1]
                waterfall_data = [
                    {"Sector": "All Industries", "Value": latest_quarter["All Industries At Basic Prices"]},
                    {"Sector": "Taxes On Product", "Value": latest_quarter["Taxes On Product"]},
                    {"Sector": "Fisim", "Value": latest_quarter["Fisim"]},
                    {"Sector": "Gdp At Market Prices", "Value": latest_quarter["Gdp At Market Prices"]}
                ]
                bar_data = pd.DataFrame(waterfall_data)
                fig_bar = px.bar(
                    bar_data, x='Sector', y='Value',
                    title="Contribution to GDP in Latest Quarter",
                    labels={"Value": "GDP (KES Million)"},
                    color='Sector'
                )
                st.plotly_chart(fig_bar, use_container_width=True, key=f"{area}_bar_chart")

    # Data Preview Tab
    with selected_tab[-1]:
        st.header("Data Preview")
        st.write(filtered_df)
        st.subheader("Summary Statistics")
        st.write(filtered_df.describe())

if __name__ == "__main__":
    main()