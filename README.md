### Names: Benineza Moise 
### ID: 26464

-----------------------------
# Big Data Capstone Project
## Part I: Problem Definition & Planning
### I. Sector Selection
Transportation

### II. Problem Statement
This project aims to analyze Citi Bike usage patterns in New York City to answer key questions:
+ What are the peak usage times for bike sharing?
+ Which stations are most popular and why?
+ How far do riders typically travel?
+ What factors influence bike usage (day of week, weather, etc.)?
+ Can we predict high-demand periods to optimize bike distribution?

### III. Dataset Identification
Dataset Title: Citi Bike Trip Data (bike.csv)
Source Link: [Citi Bike System Data]([https://example.com](https://citibikenyc.com/system-data))
Number of Rows and Columns: [To be filled after loading data]
Data Structure: Structured (CSV)
Data Status: Requires Preprocessing

## Part II: Python Analytics Tasks
### 1. Clean the Dataset
```
# 1. DATA CLEANING
def clean_data(df):
    # Handle missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    df = df.dropna(subset=['started_at', 'ended_at', 'start_lat', 'start_lng', 'end_lat', 'end_lng'])
    
    # Convert time formats
    def convert_time(time_str):
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:  # mm:ss format
                    return int(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:  # hh:mm:ss format
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            return float(time_str)
        except:
            return np.nan
    
    df['started_at'] = df['started_at'].apply(convert_time)
    df['ended_at'] = df['ended_at'].apply(convert_time)
    
    # Calculate duration and filter outliers
    df['duration'] = df['ended_at'] - df['started_at']
    df = df[(df['duration'] > 60) & (df['duration'] < 3600*24)]
    
    # Calculate distance and filter outliers
    df['distance_km'] = df.apply(lambda x: haversine(x['start_lat'], x['start_lng'], 
                                                    x['end_lat'], x['end_lng']), axis=1)
    df = df[df['distance_km'] < 50]
    
    # Encode categorical variables
    df['member_casual'] = df['member_casual'].map({'member': 1, 'casual': 0})
    
    # Calculate speed and filter outliers
    df['speed_kmh'] = df['distance_km'] / (df['duration'] / 3600)
    df = df[(df['speed_kmh'] > 5) & (df['speed_kmh'] < 30)]
    
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    
    return df

cleaned_df = clean_data(df)
print("\nCleaning complete. Final shape:", cleaned_df.shape)
```
**OUTPUT**
<img width="255" height="287" alt="Image" src="https://github.com/user-attachments/assets/fae81b59-befc-474e-897e-03efe9e60661" /> <img width="339" height="371" alt="Image" src="https://github.com/user-attachments/assets/cc8bb522-24d1-48e4-951c-5cc5769df190" />
