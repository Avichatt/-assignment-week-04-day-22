import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load data again for Part B
df = pd.read_csv('insurance.csv')

# Preprocessing categorical bits as numbers for analysis
le = LabelEncoder()
df['smoker_n'] = le.fit_transform(df['smoker'])
df['sex_n'] = le.fit_transform(df['sex'])
df['region_n'] = le.fit_transform(df['region'])

# 1. Feature Analysis
print("--- Part 1: Feature Analysis ---")
# Select only numeric for correlation
numeric_df = df[['age', 'bmi', 'children', 'charges', 'smoker_n', 'sex_n', 'region_n']]
corr_matrix = numeric_df.corr()

print("Correlation with 'charges':")
print(corr_matrix['charges'].sort_values(ascending=False))

print("\nInterpretation:")
print("Smoker status has the highest impact on charges, followed by age and bmi.")

# 2. Improve model performance
print("\n--- Part 2: Improve Model Performance ---")

# Old features (from Part A basics)
X_old = df[['age', 'children']]
y = df['charges']

# New features (including smoker status which we saw has high correlation)
X_improved = df[['age', 'bmi', 'smoker_n']]

# Split and train both
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_old, y, test_size=0.2, random_state=42)
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_improved, y, test_size=0.2, random_state=42)

model_o = LinearRegression().fit(X_train_o, y_train_o)
model_i = LinearRegression().fit(X_train_i, y_train_i)

mse_old = mean_squared_error(y_test_o, model_o.predict(X_test_o))
mse_improved = mean_squared_error(y_test_i, model_i.predict(X_test_i))

print(f"Old Features MSE: {mse_old:.2f}")
print(f"Improved Features MSE: {mse_improved:.2f}")
print(f"Improvement: {mse_old - mse_improved:.2f} less error!")

# 3. Explain Feature Selection
print("\n--- Part 3: Why Feature Selection Matters ---")
explanation = """
Feature selection impacts both regression and classification by:
- Reducing Overfitting: Removing noise (useless features) helps models learn the real patterns.
- Accuracy: Focuses the model on features that really cause the result (like smoking status for insurance cost).
- Speed: Fewer columns mean faster training and less memory used.
- Simplicity: It is easier to explain the model (Age and BMI cause price) rather than using every single detail.
"""
print(explanation)
