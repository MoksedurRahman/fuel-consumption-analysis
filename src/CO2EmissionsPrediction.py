# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('../data/FuelConsumption.csv')

# Initial data inspection
print("Data shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nData info:")
print(data.info())
print("\nSummary statistics:")
print(data.describe(include='all'))

# Drop leakage and non-useful columns
drop_cols = [
    "MODEL",  # High cardinality, behaves like an ID
    "FUELCONSUMPTION_CITY",
    "FUELCONSUMPTION_HWY",
    "FUELCONSUMPTION_COMB",  # This is likely a typo in the original code
    "FUELCONSUMPTION_COMB",
    "FUELCONSUMPTION_COMB_MPG",  # This is likely a typo in the original code
    "FUELCONSUMPTION_COMB_MPG"
]
data = data.drop(columns=[col for col in drop_cols if col in data.columns])

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# EDA - Exploratory Data Analysis
plt.figure(figsize=(15, 10))

# Numeric features distribution
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
for i, col in enumerate(numeric_features):
    plt.subplot(3, 3, i+1)
    sns.histplot(data[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Categorical features distribution
categorical_features = data.select_dtypes(include=['object']).columns
for col in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=data, x=col)
    plt.title(col)
    plt.xticks(rotation=45)
    plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = data.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.show()

# Pairplot for key numerical features
sns.pairplot(data=data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'CO2EMISSIONS']])
plt.show()

# Feature Engineering
# Create new features that might be relevant
data['FUEL_EFFICIENCY'] = (data['ENGINESIZE'] / data['CYLINDERS'])  # Hypothetical efficiency metric
data['VEHICLE_WEIGHT_TO_POWER'] = data['VEHICLEWEIGHT'] / (data['ENGINESIZE'] * data['CYLINDERS'])

# Handle categorical variables - we'll use one-hot encoding in the pipeline later
print("\nUnique values in categorical features:")
for col in categorical_features:
    print(f"{col}: {data[col].nunique()} unique values")

# Prepare data for modeling
X = data.drop(columns=['CO2EMISSIONS'])
y = data['CO2EMISSIONS']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Feature selection and modeling pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_regression, k='all')),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Define models to try
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'LinearRegression': LinearRegression(),
    'SVR': SVR()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    pipeline.set_params(regressor=model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2
    }
    
    print(f"\n{name} Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")
    print(f"R2 Score: {r2:.2f}")

# Compare model performance
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)

# Feature importance for the best model (RandomForest in this case)
best_model = pipeline.named_steps['regressor']
feature_names = numeric_features.tolist()

# Get feature names from one-hot encoding
if len(categorical_features) > 0:
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_names.extend(cat_feature_names)

# Get feature importances
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# Hyperparameter tuning for the best model (RandomForest)
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best score (negative MSE):", grid_search.best_score_)

# Evaluate the tuned model
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTuned Model Performance:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {np.sqrt(mse):.2f}")
print(f"R2 Score: {r2:.2f}")

# Save the best model
import joblib
joblib.dump(best_pipeline, 'co2_emissions_model.pkl')

# Load the model for future use
# loaded_model = joblib.load('co2_emissions_model.pkl')