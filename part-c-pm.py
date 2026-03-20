import pandas as pd

# Q1: Types of Machine Learning
print("--- Q1: Types of Machine Learning ---")
ml_types_text = """
1. Supervised Learning: Learning with labels (input and output are provided).
   Example: Predicting house price (Regression) or identifying spam email (Classification).
2. Unsupervised Learning: Learning patterns from data without labels.
   Example: Grouping customers by their shopping behavior (Clustering).
3. Reinforcement Learning: Learning through trial and error using rewards and penalties.
   Example: Teaching an AI to play chess or drive a car.
"""
print(ml_types_text)

# Q2: Coding Task (Using Pandas)
print("\n--- Q2: Coding Task ---")
df = pd.read_csv('insurance.csv')

# Conditionally filter where 'smoker' is 'yes'
smokers_df = df[df['smoker'] == 'yes']

# Compute average charges for smokers
avg_charges_smokers = smokers_df['charges'].mean()

print(f"Number of smokers in dataset: {len(smokers_df)}")
print(f"Average charges for smokers: ${avg_charges_smokers:.2f}")

# Q3: Difference between Regression and Classification
print("\n--- Q3: Regression vs Classification ---")
diff_text = """
The main difference is the 'Output':
- Regression output is continuous and quantitative.
  Example: Weight, Temperature, Age, Price, etc. 
  Model Example: Linear Regression.
- Classification output is categorical and discrete.
  Example: Yes/No, Cat/Dog, Red/Blue, Pass/Fail, etc.
  Model Example: Logistic Regression, Decision Trees.
Evaluation:
- Regression uses metrics like Mean Squared Error (MSE) or R-squared.
- Classification uses metrics like Accuracy, F1-Score, or Confusion Matrix.
"""
print(diff_text)
