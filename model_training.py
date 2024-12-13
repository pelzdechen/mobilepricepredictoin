import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

# Step 1: Reading the dataset
data = pd.read_csv('static/mobile_price_classification.csv')

# Remove duplicates
data = data.drop_duplicates()

# Print sample data for inspection
print("Sample Data:")
print(data.head())

# Step 2: Problem statement definition
print(f"Dataset Shape: {data.shape}")
print(f"Dataset Columns: {data.columns}")
# Target Variable is 'Price', Dependent Variable
target_variable = 'Price'

# Independent variables: All other features except 'Price'
independent_variables = data.columns[data.columns != target_variable].tolist()

# Step 3: Visualizing the distribution of the Target variable
sns.histplot(data[target_variable], kde=True)
plt.title(f"Distribution of Target Variable ({target_variable})")
plt.xlabel(target_variable)
plt.ylabel("Frequency")
plt.show()

# Step 4: Basic Data Exploration
print("Basic Data Info:")
print(data.info())

# Check for data types and missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Step 5: Visual Exploratory Data Analysis (EDA)
# Categorical features for bar plots
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Continuous features for histograms
continuous_features = data.select_dtypes(exclude=['object']).columns.tolist()

# Visualize categorical variables
for feature in categorical_features:
    sns.countplot(x=data[feature])
    plt.title(f"Countplot for {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.show()

# Visualize continuous variables with histograms
for feature in continuous_features:
    sns.histplot(data[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

# Step 6: Outlier Analysis (Boxplot)
for feature in continuous_features:
    sns.boxplot(x=data[feature])
    plt.title(f"Box Plot for {feature}")
    plt.xlabel(feature)
    plt.show()

# Step 7: Missing Values Analysis & Handling
# Impute missing values for continuous features with the median
for feature in continuous_features:
    data[feature].fillna(data[feature].median(), inplace=True)

# For categorical features, impute missing values with the mode
for feature in categorical_features:
    data[feature].fillna(data[feature].mode()[0], inplace=True)

# Step 8: Feature Selection (Correlation Analysis)
plt.figure(figsize=(10, 8))
sns.heatmap(data[continuous_features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Step 9: Statistical Feature Selection using ANOVA (Continuous vs Categorical)
# ANOVA for categorical features vs continuous target variable
for feature in categorical_features:
    # Perform ANOVA
    group_data = [data[data[feature] == category][target_variable] for category in data[feature].unique()]
    f_stat, p_val = stats.f_oneway(*group_data)
    print(f"ANOVA for {feature} - F-statistic: {f_stat:.2f}, p-value: {p_val:.4f}")

# Step 10: Selecting final features for the machine learning model
# Select the features for model building
selected_features = continuous_features + categorical_features
X = data[selected_features]
y = data[target_variable]

# Step 11: Data Conversion to Numeric Values for Machine Learning
# Convert categorical features to numeric using get_dummies
X = pd.get_dummies(X, drop_first=True)

# Step 12: Train/Test Split and Standardization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data (important for models like KNN and Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 13: Investigating Multiple Regression Algorithms
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
    "KNN Regressor": KNeighborsRegressor(),
    "SVR": SVR()
}

# Initialize dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R^2": r2}
    print(f"{name} - MSE: {mse:.2f}, R^2: {r2:.2f}")

# Step 14: Select the Best Model
best_model_name = max(results, key=lambda x: results[x]["R^2"])
print(f"Best model: {best_model_name} with R^2: {results[best_model_name]['R^2']:.2f}")

# Train and save the best model
best_model = models[best_model_name]
best_model.fit(X_train_scaled, y_train)

# Saving the best model (optional)
import joblib
joblib.dump(best_model, 'best_model.pkl')
