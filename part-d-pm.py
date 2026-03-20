prompt_used = """
Prompt AI: "Explain types of machine learning, regression, and classification with Python examples using Pandas."
"""


ai_output_response = """
AI RESPONSE:
ML Types:
1. Supervised: Data has labels (Regression/Classification).
2. Unsupervised: Data has no labels (Clustering).
3. Reinforcement: Learning by trial and error.

Code Example:
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

# Load sample
data = {'feature': [1, 2, 3, 4], 'target_reg': [10, 20, 30, 40], 'target_clf': [0, 1, 0, 1]}
df = pd.DataFrame(data)

# Regression Example
model_reg = LinearRegression().fit(df[['feature']], df['target_reg'])
print(f"Regression Prediction: {model_reg.predict([[5]])}")

# Classification Example
model_clf = LogisticRegression().fit(df[['feature']], df['target_clf'])
print(f"Classification Prediction: {model_clf.predict([[5]])}")
"""


evaluation_report = """
--- Evaluation of AI Output ---
- Concepts Explained correctly?
  YES. The definition of ML types and the separation of regression vs. classification are correct. 
  It uses the term labels correctly.
- Code runnable and meaningful?
  YES. It uses a small dictionary to create a Pandas DataFrame (df = pd.DataFrame(data)) 
  and demonstrates LinearRegression and LogisticRegression correctly. 
  It correctly uses fit and predict.
- Usefulness:
  The example is perfect for a beginner because it shows all parts in a few lines.
"""

print("--- Part 1: Prompt Doc ---")
print(prompt_used)
print("\n--- Part 2: AI Output ---")
print(ai_output_response)
print("\n--- Part 3: Evaluation ---")
print(evaluation_report)
