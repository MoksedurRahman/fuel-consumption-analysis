# fuel-consumption-analysis
Fuel Efficiency Analysis and Visualization using Python


# 🚗 Fuel Efficiency Analysis and Visualization using Python

## 📊 Overview
This project analyzes the `FuelConsumption.csv` dataset to explore how engine size, cylinders, and fuel consumption affect CO2 emissions. It includes data cleaning, visualizations, and a regression model to predict emissions using Python.

## Dataset
- Source: Natural Resources Canada (FuelConsumption.csv)
- Features: Engine Size, Cylinders, Fuel Consumption (City, Hwy, Combined), CO₂ Emissions

## Objectives
- Understand feature relationships and trends.
- Identify key contributors to CO₂ emissions.
- Provide visual insights using charts.

## 🧰 Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (for regression)

## 📁 Folder Structure

fuel-consumption-analysis/  
│
├── data/  
│     └── FuelConsumption.csv         # Original dataset  
│
├── notebooks/  
│   └── fuel-consumption-analysis.ipynb         # Main analysis notebook  
│
├── images/  
│   └── *.png                       # Exported plots  
│
├── src/  
│   └── fuel-consumption-analysis.py    # Script to load and preprocess data  
│
├── README.md                       # Project description and summary  
├── requirements.txt                # List of dependencies  
└── .gitignore                      # Ignore unnecessary files  


## 🚀 How to Run

1. Clone this repo
2. Install dependencies:  
```bash
pip install -r requirements.txt
