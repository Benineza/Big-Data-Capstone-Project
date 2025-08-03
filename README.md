### Names: Benineza Moise 
### ID: 26464

-----------------------------
# Big Data Capstone Project
# PART 1: Problem Definition & Planning
### I. Sector Selection
Transportation Sector

### II. Problem Statement
This project aims to analyze Citi Bike usage patterns in New York City to answer key questions:
+ What are the peak usage times for bike sharing?
+ Which stations are most popular and why?
+ How far do riders typically travel?
+ What factors influence bike usage (day of week, weather, etc)
+ Can we predict high demand periods to optimize bike distribution?

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
**OUTPUT:**

<img width="255" height="287" alt="Image" src="https://github.com/user-attachments/assets/fae81b59-befc-474e-897e-03efe9e60661" /> <img width="339" height="371" alt="Image" src="https://github.com/user-attachments/assets/cc8bb522-24d1-48e4-951c-5cc5769df190" />
### 2. EXPLORATORY DATA ANALYSIS (EDA)
* Descriptive statistics
```
def perform_eda(df):
    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(df[['duration', 'distance_km', 'speed_kmh']].describe())
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Duration distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['duration']/60, bins=50, kde=True)
    plt.title('Trip Durations (minutes)')
    
    # Distance distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['distance_km'], bins=50, kde=True)
    plt.title('Trip Distances (km)')
    
    # Rider type distribution
    plt.subplot(2, 2, 3)
    df['member_casual'].value_counts().plot(kind='bar')
    plt.title('Member vs Casual Riders')
    plt.xticks([0, 1], ['Casual', 'Member'], rotation=0)
    
    # Bike type distribution
    plt.subplot(2, 2, 4)
    df['rideable_type'].value_counts().plot(kind='bar')
    plt.title('Bike Type Distribution')
    
    plt.tight_layout()
    plt.show()
```
**OUTPUT:**

<img width="1068" height="485" alt="Image" src="https://github.com/user-attachments/assets/86deafff-2916-4c92-812e-ecbe9991603a" />

* Visualize distributions and relationships among variables
<img width="1067" height="357" alt="Image" src="https://github.com/user-attachments/assets/cd70abf1-788a-42b2-aa75-e83e813981b4" />
<img width="879" height="498" alt="Image" src="https://github.com/user-attachments/assets/cd10ecd2-3be0-4180-8c01-9e43240628fd" />

### 3. Applying a Machine Learning or Clustering Model
* I chose classification as the suitable model & Train the model on the dataset
```
# CLASSIFICATION MODEL (Member vs Casual Prediction)
# Prepare data
def prepare_data(df):
    features = ['duration', 'distance_km', 'speed_kmh', 'start_lat', 'start_lng', 'end_lat', 'end_lng']
    X = df[features]
    y = df['member_casual']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

X_train, X_test, y_train, y_test = prepare_data(cleaned_df)

# Train model
def train_classification_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_classification_model(X_train, y_train)

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    
    features = ['duration', 'distance_km', 'speed_kmh', 'start_lat', 'start_lng', 'end_lat', 'end_lng']
    importances = model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.title('Feature Importance')
    plt.show()

evaluate_model(model, X_test, y_test)
```
**OUTPUT:**

<img width="784" height="513" alt="Image" src="https://github.com/user-attachments/assets/0e9ab7f2-6172-4e5c-8484-1327c822ccf8" />

### 4. Evaluating the Model
* I chose silhouette score as the evaluation metrics
```
# CLUSTERING ANALYSIS
def perform_clustering(df):
    # 6.1 Prepare data
    features = ['duration', 'distance_km', 'speed_kmh']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal clusters
    range_n_clusters = [2, 3, 4, 5, 6]
    best_score = -1
    best_n = 2
    
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        print(f"For n_clusters = {n_clusters}, silhouette score is {silhouette_avg}")
        
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n = n_clusters
    
    print(f"\nOptimal number of clusters: {best_n} with score {best_score}")
    
    # Final clustering
    kmeans = KMeans(n_clusters=best_n, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualize clusters
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='distance_km', y='duration', hue='cluster', data=df, palette='viridis')
    plt.title('Clusters by Distance/Duration')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='speed_kmh', y='distance_km', hue='cluster', data=df, palette='viridis')
    plt.title('Clusters by Speed/Distance')
    
    plt.tight_layout()
    plt.show()
    
    # Cluster analysis
    print("\nCluster Characteristics:")
    print(df.groupby('cluster').agg({
        'duration': 'mean',
        'distance_km': 'mean',
        'speed_kmh': 'mean',
        'member_casual': 'mean'
    }))
    
    return df

clustered_df = perform_clustering(cleaned_df)
```
**OUTPUT:**

<img width="883" height="492" alt="Image" src="https://github.com/user-attachments/assets/d8600f10-a1e8-408a-8b20-9b86d0826e0e" />

### 6. Incorporate Innovation
```
# 6. INNOVATIVE ANALYSIS: TRIP PURPOSE INFERENCE
def infer_trip_purposes(df):
    # Calculate CBD distances
    times_square_lat, times_square_lon = 40.7580, -73.9855
    
    df['start_dist_to_cbd'] = df.apply(
        lambda x: haversine(x['start_lat'], x['start_lng'], times_square_lat, times_square_lon), axis=1)
    df['end_dist_to_cbd'] = df.apply(
        lambda x: haversine(x['end_lat'], x['end_lng'], times_square_lat, times_square_lon), axis=1)
    
    # Infer purposes
    conditions = [
        (df['start_dist_to_cbd'] < 2) & (df['end_dist_to_cbd'] < 2),
        (df['start_dist_to_cbd'] < 2) & (df['end_dist_to_cbd'] >= 2),
        (df['start_dist_to_cbd'] >= 2) & (df['end_dist_to_cbd'] < 2),
        (df['start_dist_to_cbd'] >= 2) & (df['end_dist_to_cbd'] >= 2)
    ]
    
    choices = ['CBD circulation', 'CBD departure', 'CBD arrival', 'Non-CBD trip']
    df['inferred_purpose'] = np.select(conditions, choices)
    
    # Analyze purposes
    print("\nTrip Purposes by Rider Type:")
    print(pd.crosstab(df['inferred_purpose'], df['member_casual']))
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.countplot(x='inferred_purpose', hue='member_casual', data=df)
    plt.title('Trip Purposes by Rider Type')
    plt.xticks(rotation=45)
    plt.show()
    
    return df

cleaned_df = infer_trip_purposes(cleaned_df)
```
**OUTPUT:**

<img width="776" height="494" alt="Image" src="https://github.com/user-attachments/assets/9ca0652b-89d2-49bc-b466-64a94d19ebad" />

### Creating the final clean dataset
```
cleaned_df.to_csv('bike_cleaned.csv', index=False)
```

# PART 3 Power BI Dashboard Tasks
### Problem & Insights
Analyzing ride durations, distances, stations, speed and purpose of ride covered by Citi Bikes.

**Purpose:** Analyzing bikes ridden in the city what purpose they were ridden for, where they were ridden at and if the rides were by members or casuals 
### Incorporate Interactivity with the Appropriate Visuals
<img width="761" height="438" alt="Image" src="https://github.com/user-attachments/assets/22c42205-70c4-4cb9-892d-4969b0199558" />
<img width="762" height="419" alt="Image" src="https://github.com/user-attachments/assets/004e9c27-6ddf-4d57-a2da-7a04a71a3632" />

### Innovative Features (DAX formulas & Custom Tooltip)
***Average Speed for Members***
```
Avg_Speed_Member = AVERAGEX(FILTER('bike_cleaned', 'bike_cleaned'[member_casual] = 1), 'bike_cleaned'[speed_kmh])
```
***Total Distance by Casuals***
```
Total_Distance_Casual = CALCULATE(SUM('bike_cleaned'[distance_km]), 'bike_cleaned'[member_casual] = 0)
```

