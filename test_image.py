import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import calendar
import numpy as np
# Set professional matplotlib styling with high resolution
#plt.style.use('vayuchat.mplstyle')
df = pd.read_csv("AQ_met_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
states_df = pd.read_csv("states_data.csv")
ncap_df = pd.read_csv("ncap_funding_data.csv")
# df is pandas DataFrame with air quality data from India. Data frequency is daily from 2017 to 2024. The data has the following columns and data types:
# Unnamed: 0                int64
# Timestamp        datetime64[ns]
# State                    object
# City                     object
# Station                  object
# site_id                  object
# Year                      int64
# PM2.5 (µg/m³)           float64
# PM10 (µg/m³)            float64
# NO (µg/m³)              float64
# NO2 (µg/m³)             float64
# NOx (ppb)               float64
# NH3 (µg/m³)             float64
# SO2 (µg/m³)             float64
# CO (mg/m³)              float64
# Ozone (µg/m³)           float64
# AT (°C)                 float64
# RH (%)                  float64
# WS (m/s)                float64
# WD (deg)                float64
# RF (mm)                 float64
# TOT-RF (mm)             float64
# SR (W/mt2)              float64
# BP (mmHg)               float64
# VWS (m/s)               float64
# dtype: object
# states_df is a pandas DataFrame of state-wise population, area and whether state is union territory or not of India.
# state               object
# population           int64
# area (km2)           int64
# isUnionTerritory      bool
# dtype: object
# ncap_df is a pandas DataFrame of funding given to the cities of India from 2019-2022, under The National Clean Air Program (NCAP).
# S. No.                                 int64
# state                                 object
# city                                  object
# Amount released during FY 2019-20    float64
# Amount released during FY 2020-21    float64
# Amount released during FY 2021-22    float64
# Total fund released                  float64
# Utilisation as on June 2022          float64
# dtype: object
# Question: Compare the wind speed and PM2.5 levels during Delhi’s most polluted week (highest PM2.5) in December 2024 with the previous 15 days and the following 15 days on a time series plot.
# Generate code to answer the question and save result in 'answer' variable
# If creating a plot, save it with a unique filename and store the filename in 'answer'
# If returning text/numbers, store the result directly in 'answer'


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import numpy as np

# Ensure data is loaded
if df.empty:
    answer = "No data available"
else:
    try:
        # Filter for Delhi in December 2024
        df_delhi = df[
            (df['City'].str.contains('Delhi', case=False, na=False)) &
            (df['Timestamp'].dt.year == 2024) &
            (df['Timestamp'].dt.month == 12)
        ].copy()
        df_delhi = df_delhi.dropna(subset=['PM2.5 (µg/m³)', 'WS (m/s)'])
        df_delhi = df_delhi.sort_values('Timestamp')

        # Need at least a full week to compute rolling mean
        if len(df_delhi) < 7:
            answer = "Insufficient data"
        else:
            # Compute 7‑day rolling mean of PM2.5
            df_delhi['PM2.5_roll7'] = df_delhi['PM2.5 (µg/m³)'].rolling(window=7, min_periods=7).mean()
            # Identify the window with the highest mean PM2.5
            max_idx = df_delhi['PM2.5_roll7'].idxmax()
            max_end_date = df_delhi.loc[max_idx, 'Timestamp']
            max_start_date = max_end_date - pd.Timedelta(days=6)

            # Define extended window: 15 days before start and 15 days after end
            ext_start = max_start_date - pd.Timedelta(days=15)
            ext_end = max_end_date + pd.Timedelta(days=15)

            # Filter data for the extended period
            mask = (df_delhi['Timestamp'] >= ext_start) & (df_delhi['Timestamp'] <= ext_end)
            df_plot = df_delhi.loc[mask].copy()

            if df_plot.empty or len(df_plot) < 30:
                answer = "Insufficient data"
            else:
                # Plot time series
                plt.figure(figsize=(9, 6))
                ax1 = plt.gca()
                sns.lineplot(data=df_plot, x='Timestamp', y='PM2.5 (µg/m³)', ax=ax1,
                             label='PM2.5 (µg/m³)', color='tab:red')
                ax1.set_ylabel('PM2.5 (µg/m³)', color='tab:red')
                ax1.tick_params(axis='y', labelcolor='tab:red')

                ax2 = ax1.twinx()
                sns.lineplot(data=df_plot, x='Timestamp', y='WS (m/s)', ax=ax2,
                             label='Wind Speed (m/s)', color='tab:blue')
                ax2.set_ylabel('Wind Speed (m/s)', color='tab:blue')
                ax2.tick_params(axis='y', labelcolor='tab:blue')

                plt.title('Delhi – PM2.5 and Wind Speed around Most Polluted Week (Dec 2024)')
                plt.xlabel('Date')
                plt.tight_layout()

                # Save plot
                filename = f"plot.png"
                plt.savefig(filename, dpi=1200, bbox_inches='tight', facecolor='white')
                plt.close()

                answer = filename
    except Exception as e:
        answer = "Unable to complete analysis with available data"