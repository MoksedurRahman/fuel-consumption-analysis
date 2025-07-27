# %% [markdown]
# 🧠 8 Steps of Machine Learning

# %% [markdown]
# ✅ Full 9-Step ML Pipeline  
# 
# Step	---  Name	---  Purpose  
# 
# 0️⃣	Import Libraries:	Load Python packages for data, ML, and plotting  
# 1️⃣	Data Gathering:	Collect and import data  
# 2️⃣	Data Pre-processing:	Clean and prepare data  
# 3️⃣	Feature Engineering:	Enhance or select important features  
# 4️⃣	Choosing Model:	Pick the best ML algorithm  
# 5️⃣	Training Model:	Train model on training dataset  
# 6️⃣	Test/Evaluate Model:	Assess model performance  
# 7️⃣	Parameter Tuning:	Improve model with hyperparameter tuning  
# 8️⃣	Prediction:	Use the trained model to predict unseen data  

# %% [markdown]
# 🧠 0️⃣ Import Libraries

# %%
# Data Handling

import numpy as np
import matplotlib.pyplot as plt
#pip install pandas
import pandas as pd

# Visualization

import matplotlib.pyplot as plt
#pip install seaborn
import seaborn as sns
%matplotlib inline

# Machine Learning
#pip install scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# 1️⃣ Data Gathering  
# 
# Collect relevant data from various sources like:  
# 
# CSV files (e.g., FuelConsumption.csv)

# %%
# Data Loading
df = pd.read_csv("../data/FuelConsumption.csv")

# %% [markdown]
# 2️⃣ Data Preprocessing  
# 
# Clean and format the data:  
# 
# Handle missing values  
# 
# Remove duplicates  
# 
# Convert data types  
# 
# Normalize or scale features  

# %%
df.head()

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# 3️⃣ Feature Engineering   
# 
# -- Transform raw data into meaningful features:  
# 
# -- Select important variables  
# 
# -- Create new features (ratios, interactions)  
# 
# -- Encode categorical variables (e.g., One-Hot Encoding)  
# 
# -- Reduce dimensions (e.g., PCA)  

# %%
df.drop(columns=['MODELYEAR','MAKE','MODEL','TRANSMISSION'],inplace=True)

# %%
df['FUELTYPE'].value_counts()

# %%
df['VEHICLECLASS'].value_counts()

# %%
df.drop(columns=['FUELTYPE','VEHICLECLASS'],inplace=True)

# %%
df.head()

# %%
df.hist()

# %%
df.hist('ENGINESIZE')

# %%
df.hist('CYLINDERS')

# %%
df.hist('FUELCONSUMPTION_HWY')

# %%
df.hist('FUELCONSUMPTION_COMB')

# %%
df.hist('FUELCONSUMPTION_COMB_MPG')

# %% [markdown]
# Feature Selection

# %%
df.corr()

# %%
sns.heatmap(df.corr(), annot=True)

# %%
df.drop(columns=['FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG'], inplace=True)

# %%
sns.heatmap(df.corr(), annot=True)

# %%
df.head()

# %%
feature = df.drop(columns=['CO2EMISSIONS'])
feature

# %%
target = df[['CO2EMISSIONS']]
target

# %% [markdown]
# 4️⃣ Choosing a Model  
# 
# Choose the appropriate algorithm:  
# 
# -- Regression: Predicting continuous values (e.g., CO2 Emissions)  
# 
# -- Classification: Predicting categories (e.g., spam/ham)  
# 
# -- Clustering: Grouping data (e.g., KMeans)  

# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# %% [markdown]
# 5️⃣ Training the Model  
# 
# -- Fit the model to your training data.

# %%
# Spliting Data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(feature, target, test_size=0.25, random_state=1)

# %%
xtrain.shape, xtest.shape

# %%
ytrain.shape, ytest.shape

# %%
# Choosing Model

plt.scatter(feature.ENGINESIZE ,target.values)
plt.title("EngineSize vs Co2Emission")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSION")
plt.show()

# %%
# Modeling
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xtrain[['ENGINESIZE']], ytrain.values.ravel())

# %%
# Value of Theta 0
model.intercept_

# %%
model.coef_

# %% [markdown]
# 6️⃣ Model Testing / Evaluation  
# 
# -- Evaluate how well the model performs:  
# 
# -- Metrics: MAE, MSE, RMSE, R² for regression  
# 
# -- Accuracy, Precision, Recall, F1 for classification  

# %%
# Prediction
ypred = model.predict(xtest[['ENGINESIZE']])
xtest[['ENGINESIZE']].iloc[0], ypred[0], ytest.values[0]

# %%
ytest.values[0] - ypred[0] 

# %%
# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("Absoulute Error: ", mean_absolute_error(ytest, ypred))
print("Mean Squarred Error: ", mean_squared_error(ytest, ypred))
print("R2 Score: ", r2_score(ytest, ypred))

# %%
model2 = LinearRegression()
model2.fit(xtrain[['CYLINDERS']], ytrain.values.ravel())
# Prediction
ypred = model2.predict(xtest[['CYLINDERS']])

print("Absoulute Error: ", mean_absolute_error(ytest, ypred))
print("Mean Squarred Error: ", mean_squared_error(ytest, ypred))
print("R2 Score: ", r2_score(ytest, ypred))

# %% [markdown]
# 7️⃣ Parameter Tuning   
# 
# Optimize model parameters using:  
# 
# -- Grid Search  
# 
# -- Random Search  
# 
# -- Cross-validation  

# %%


# %% [markdown]
# 8️⃣ Prediction  
# 
# Use the trained model to make future predictions.


