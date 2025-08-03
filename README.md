### Names: Benineza Moise 
### ID: 26464

-----------------------------
# Big Data Capstone Project
# PART 1: Problem Definition & Planning
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

# PART 2: Python Analytics Tasks
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
### 2. EXPLORATORY DATA ANALYSIS (EDA)
* Descriptive statistics
<img width="1068" height="485" alt="Image" src="https://github.com/user-attachments/assets/86deafff-2916-4c92-812e-ecbe9991603a" />

* Visualize distributions and relationships among variables
<img width="1067" height="357" alt="Image" src="https://github.com/user-attachments/assets/cd70abf1-788a-42b2-aa75-e83e813981b4" />
<img width="879" height="498" alt="Image" src="https://github.com/user-attachments/assets/cd10ecd2-3be0-4180-8c01-9e43240628fd" />

### 3. Applying a Machine Learning or Clustering Model
* I chose classification as the suitable model & Train the model on the dataset
<img width="784" height="513" alt="Image" src="https://github.com/user-attachments/assets/0e9ab7f2-6172-4e5c-8484-1327c822ccf8" />

### 4. Evaluating the Model
* I chose silhouette score as the evaluation metrics
<img width="883" height="492" alt="Image" src="https://github.com/user-attachments/assets/d8600f10-a1e8-408a-8b20-9b86d0826e0e" />

### 6. Incorporate Innovation
<img width="776" height="494" alt="Image" src="https://github.com/user-attachments/assets/9ca0652b-89d2-49bc-b466-64a94d19ebad" />

# PART 3 Power BI Dashboard Tasks
### Problem & Insights
Analyzing ride durations, distances, stations, speed and purpose of ride covered by Citi Bikes.

**Purpose:** Analyzing bikes ridden in the city what purpose they were ridden for, where they were ridden at and if the rides were by members or casuals 
### Incorporate Interactivity with the Appropriate Visuals
<img width="761" height="438" alt="Image" src="https://github.com/user-attachments/assets/22c42205-70c4-4cb9-892d-4969b0199558" />
