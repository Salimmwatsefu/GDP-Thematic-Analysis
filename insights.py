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