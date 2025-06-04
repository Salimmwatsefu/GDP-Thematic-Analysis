import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression

def interpret_statistical_measures(mae, mse, rmse, mape, z_value, p_value, sector):
    """
    Generate interpretations for statistical measures.
    Args:
        mae: Mean Absolute Error
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
        mape: Mean Absolute Percentage Error
        z_value: Z-Value
        p_value: P-Value
        sector: Name of the sector being analyzed
    Returns:
        List of interpretation strings
    """
    interpretations = []
    
    # MAE Interpretation
    interpretations.append(f"**Error Metrics Interpretation for {sector}:**")
    interpretations.append(f"- The Mean Absolute Error (MAE) of {mae:,.0f} KES Million indicates the average magnitude of prediction errors. "
                         f"This means our predictions typically deviate by {mae:,.0f} KES Million from actual values.")
    
    # RMSE vs MAE Interpretation
    if rmse > mae:
        interpretations.append(f"- The Root Mean Squared Error (RMSE) of {rmse:,.0f} KES Million is larger than the MAE, "
                             "suggesting the presence of some larger prediction errors or outliers in the data.")
    else:
        interpretations.append(f"- The Root Mean Squared Error (RMSE) of {rmse:,.0f} KES Million is close to the MAE, "
                             "indicating consistent prediction errors without major outliers.")
    
    # MAPE Interpretation
    if mape < 10:
        accuracy_level = "excellent"
    elif mape < 20:
        accuracy_level = "good"
    elif mape < 30:
        accuracy_level = "acceptable"
    else:
        accuracy_level = "poor"
    
    interpretations.append(f"- The Mean Absolute Percentage Error (MAPE) of {mape:.1f}% indicates {accuracy_level} prediction accuracy. "
                         f"This means our predictions are typically off by {mape:.1f}% of the actual value.")
    
    # Z-Value Interpretation
    if abs(z_value) > 2:
        z_interpretation = "significantly different from"
    elif abs(z_value) > 1:
        z_interpretation = "moderately different from"
    else:
        z_interpretation = "close to"
    
    interpretations.append(f"- The Z-Value of {z_value:.2f} indicates that the latest observation is {z_interpretation} "
                         "the historical mean in terms of standard deviations.")
    
    # P-Value Interpretation
    if p_value < 0.01:
        significance = "strong"
    elif p_value < 0.05:
        significance = "moderate"
    elif p_value < 0.1:
        significance = "weak"
    else:
        significance = "no"
    
    interpretations.append(f"- The P-Value of {p_value:.4f} suggests {significance} statistical significance in the trend. "
                         f"{'This indicates a meaningful pattern in the data.' if significance != 'no' else 'This suggests the variations might be random.'}")
    
    return interpretations

def generate_statistical_summary(data, sectors, chart_type="time-series"):
    """
    Generate statistical summary for given data and sectors.
    Args:
        data: Pandas DataFrame or Series (filtered_df for charts, or treemap/pie/bar data for snapshots).
        sectors: List of sector names or single sector (string).
        chart_type: 'time-series' for full stats, 'snapshot' for Mean/Median only.
    Returns:
        Markdown string with statistical summary.
    """
    stats = []
    stats.append("**Statistical Summary**:")

    if isinstance(sectors, str):
        sectors = [sectors]

    for sector in sectors:
        if sector not in data.columns and not isinstance(data, pd.Series):
            continue
        # Extract data for the sector
        if isinstance(data, pd.Series):
            sector_data = data
        elif chart_type == "snapshot" and isinstance(data, pd.DataFrame) and "Value" in data.columns:
            sector_data = data["Value"]  # For treemap, pie, or bar data
        else:
            sector_data = data[sector].dropna()

        if sector_data.empty or len(sector_data) == 0:
            stats.append(f"- **{sector}**: No data available")
            continue

        if chart_type == "time-series":
            # Basic statistics
            mean_val = sector_data.mean()
            std_val = sector_data.std()
            median_val = sector_data.median()
            min_val = sector_data.min()
            max_val = sector_data.max()

            # Error metrics and statistical tests
            if len(sector_data) > 1:
                # Calculate trend using simple linear regression
                X = np.arange(len(sector_data)).reshape(-1, 1)
                y = sector_data.values
                model = LinearRegression()
                model.fit(X, y)
                predicted = model.predict(X)

                # Mean Absolute Error
                mae = np.mean(np.abs(y - predicted))
                
                # Mean Squared Error
                mse = np.mean((y - predicted) ** 2)
                
                # Root Mean Squared Error
                rmse = np.sqrt(mse)
                
                # Mean Absolute Percentage Error
                mape = np.mean(np.abs((y - predicted) / y)) * 100
                
                # Z-Value for the latest observation
                z_value = (sector_data.iloc[-1] - mean_val) / std_val if std_val != 0 else 0
                
                # P-Value from t-test against the mean
                _, p_value = scipy_stats.ttest_1samp(sector_data, sector_data.mean())

            # Percentage change (first to last)
            first_val = sector_data.iloc[0]
            last_val = sector_data.iloc[-1]
            pct_change = ((last_val - first_val) / first_val) * 100 if first_val != 0 else np.nan

            stats.append(f"- **{sector}**:")
            stats.append(f"  - **Historical Mean**: {mean_val:,.0f} KES Million")
            stats.append(f"  - **Historical Standard Deviation**: {std_val:,.0f} KES Million")
            stats.append(f"  - **Historical Median**: {median_val:,.0f} KES Million")
            stats.append(f"  - **Historical Min**: {min_val:,.0f} KES Million")
            stats.append(f"  - **Historical Max**: {max_val:,.0f} KES Million")
            stats.append(f"  - **Percentage Change**: {'+' if pct_change >= 0 else ''}{pct_change:.1f}%" if not np.isnan(pct_change) else "  - **Percentage Change**: Not applicable")
            
            if len(sector_data) > 1:
                stats.append("\n  **Error and Statistical Measures:**")
                stats.append(f"  - **Mean Absolute Error**: {mae:,.0f} KES Million")
                stats.append(f"  - **Mean Squared Error**: {mse:,.0f} KES Million")
                stats.append(f"  - **Root Mean Squared Error**: {rmse:,.0f} KES Million")
                stats.append(f"  - **Mean Absolute Percentage Error**: {mape:.1f}%")
                stats.append(f"  - **Z-Value (Latest)**: {z_value:.2f}")
                stats.append(f"  - **P-Value**: {p_value:.4f}")
                
                # Add interpretations
                stats.append("\n")
                interpretations = interpret_statistical_measures(mae, mse, rmse, mape, z_value, p_value, sector)
                stats.extend(interpretations)
                stats.append("\n")

        elif chart_type == "snapshot":
            # Minimal stats for snapshot charts
            mean_val = sector_data.mean()
            median_val = sector_data.median()
            stats.append(f"- **{sector}**:")
            stats.append(f"  - **Mean Contribution**: {mean_val:,.0f} KES Million")
            stats.append(f"  - **Median Contribution**: {median_val:,.0f} KES Million")

    return "\n".join(stats)

def create_executive_dashboard(filtered_df, selected_quarter):
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
    
    # Statistical summary for line chart
    stats_summary = generate_statistical_summary(filtered_df, "Gdp At Market Prices", chart_type="time-series")
    st.markdown(stats_summary)
    
    st.write("**Highlight**: GDP growth has been steady, with a notable increase in service sectors like Information and Communication.")

def create_thematic_area_tab(area, filtered_df, selected_sectors, selected_quarter):
    st.header(area)
    st.write("**Insights**")
    from insights import generate_insights
    insights = generate_insights(filtered_df, area, selected_sectors)
    for insight in insights:
        st.write(f"- {insight}")

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
        
        # Statistical summary for line chart (grouped for all sectors)
        stats_summary = generate_statistical_summary(filtered_df, selected_sectors, chart_type="time-series")
        st.markdown(stats_summary)

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
        
        # Statistical summary for bar chart
        stats_summary = generate_statistical_summary(filtered_df, selected_sectors, chart_type="time-series")
        st.markdown(stats_summary)

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
            
            # Statistical summary for dual-axis chart
            stats_summary = generate_statistical_summary(filtered_df, ["Manufacturing", "Electricity & Water Supply"], chart_type="time-series")
            st.markdown(stats_summary)

        # Heatmap: Quarter-year performance (static)
        pivot_data = filtered_df.pivot_table(index='Year', columns='Quarter', values=selected_sectors[0] if selected_sectors else "Manufacturing")
        fig_heatmap, ax = plt.subplots()
        sns.heatmap(pivot_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
        plt.title(f"{selected_sectors[0] if selected_sectors else 'Manufacturing'} by Quarter and Year")
        st.pyplot(fig_heatmap)
        
        # Statistical summary for heatmap (snapshot)
        stats_summary = generate_statistical_summary(filtered_df, selected_sectors[0] if selected_sectors else "Manufacturing", chart_type="snapshot")
        st.markdown(stats_summary)

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
        
        # Statistical summary for bar chart
        stats_summary = generate_statistical_summary(filtered_df, selected_sectors, chart_type="time-series")
        st.markdown(stats_summary)

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
        
        # Statistical summary for stacked area chart
        stats_summary = generate_statistical_summary(filtered_df, selected_sectors, chart_type="time-series")
        st.markdown(stats_summary)

        # Treemap: Proportion of sectors
        latest_quwatermark = True
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
        
        # Statistical summary for treemap (snapshot)
        stats_summary = generate_statistical_summary(treemap_data, "Value", chart_type="snapshot")
        st.markdown(stats_summary)

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
        
        # Statistical summary for bar chart
        stats_summary = generate_statistical_summary(filtered_df, selected_sectors, chart_type="time-series")
        st.markdown(stats_summary)

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
        
        # Statistical summary for pie chart (snapshot)
        stats_summary = generate_statistical_summary(pie_data, "Value", chart_type="snapshot")
        st.markdown(stats_summary)

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
        
        # Statistical summary for bar chart
        stats_summary = generate_statistical_summary(filtered_df, selected_sectors, chart_type="time-series")
        st.markdown(stats_summary)

    elif area == "Digital Economy and Communication":
        # Scatter Plot: ICT vs GDP
        if "Information And Communication" in selected_sectors:
            fig_scatter = px.scatter(
                filtered_df, x="Information And Communication", y="Gdp At Market Prices",
                title="ICT vs GDP at Market Prices",
                labels={"Information And Communication": "ICT (KES Million)", "Gdp At Market Prices": "GDP (KES Million)"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key=f"{area}_scatter_chart")
            
            # Statistical summary for scatter plot
            stats_summary = generate_statistical_summary(filtered_df, ["Information And Communication", "Gdp At Market Prices"], chart_type="time-series")
            st.markdown(stats_summary)

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
        
        # Statistical summary for bar chart
        stats_summary = generate_statistical_summary(filtered_df, selected_sectors, chart_type="time-series")
        st.markdown(stats_summary)

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
        
        # Statistical summary for bar chart (snapshot)
        stats_summary = generate_statistical_summary(bar_data, "Value", chart_type="snapshot")
        st.markdown(stats_summary)

