import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Identify ML Problem Type
print("--- Part 1: Identify ML Problem Type ---")
print("Dataset: Medical Cost Personal Dataset (insurance.csv)")
print("ML Type: Supervised Learning")
print("Justification: The dataset contains input features (age, sex, bmi, etc.) and known target labels like 'charges' and 'smoker'.")
print("Tasks:")
print(" - Regression: Predicting 'charges' (a continuous numeric value).")
print(" - Classification: Predicting 'smoker' status (a categorical value: yes/no).")
print("\n")

# 2. Data Handling with Pandas
print("--- Part 2: Data Handling with Pandas ---")
# Load dataset
df = pd.read_csv('insurance.csv')
print(f"Dataset loaded with {len(df)} rows.")

# Handle missing values (Check if there are any)
print("Checking for missing values:")
print(df.isnull().sum())
# If there were missing values, we could use: df = df.dropna() or df.fillna(df.mean())
# For this dataset, usually there are no nulls, but let's be safe.
df = df.dropna() 

# Select relevant features for basic models
# We need to convert categorical text to numbers for sklearn
le = LabelEncoder()
df['sex_n'] = le.fit_transform(df['sex'])
df['smoker_n'] = le.fit_transform(df['smoker'])
df['region_n'] = le.fit_transform(df['region'])

# 3. Regression Task
print("\n--- Part 3: Regression Task ---")
# Target: charges (continuous)
X_reg = df[['age', 'bmi', 'children', 'smoker_n']]
y_reg = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred_reg = reg_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_reg)
print(f"Regression Target: charges")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# 4. Classification Task
print("\n--- Part 4: Classification Task ---")
# Target: smoker (categorical)
X_clf = df[['age', 'bmi', 'children', 'charges']]
y_clf = df['smoker_n']

X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train, y_train)
y_pred_clf = clf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred_clf)
print(f"Classification Target: smoker")
print(f"Accuracy: {acc * 100:.2f}%")

# 5. Comparison
print("\n--- Part 5: Comparison ---")
comparison = {
    "Metric Type": ["Output Type", "Use Cases", "Evaluation Metrics"],
    "Regression": ["Continuous (Numeric)", "Predicting prices, scores, costs", "MSE, RMSE, R-squared"],
    "Classification": ["Categorical (Labels)", "Predicting Yes/No, Spam/Not Spam", "Accuracy, Precision, Recall"]
}
print(pd.DataFrame(comparison))
