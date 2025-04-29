import streamlit as st
import pandas as pd

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
        
        # Create a datetime column for potential sorting and forecasting
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Quarter'].str.replace('Q', '') + '-01')
        
        # Sort by Date to ensure chronological order
        df = df.sort_values('Date')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None