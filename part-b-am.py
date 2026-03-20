import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('sales_data.csv')

# Part 1. Grouped analysis: Group by Region and compute means
print("--- Part 1. Grouped analysis: Mean Profit by Region ---")
region_means = df.groupby('Region')['Profit'].mean().reset_index()
print(region_means)


# Part 2. Visualize grouped results using bar plot
plt.figure(figsize=(10, 6))
plt.bar(region_means['Region'], region_means['Profit'], color='teal')
plt.title('Mean Profit by Region')
plt.xlabel('Region')
plt.ylabel('Average Profit')
plt.show()


# Part 3. Compare two numerical features using KDE/Line
plt.figure(figsize=(12, 6))
sns.kdeplot(df['Sales'], label='Sales', color='blue', fill=True, alpha=0.3)
sns.kdeplot(df['Profit'], label='Profit', color='red', fill=True, alpha=0.5)
plt.title('KDE Comparison: Sales vs Profit')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.show()

# Interpretation: 
# The Sales distribution is spread across a much wider range (100 to 5000) 
# while the Profit is more concentrated at lower values as it's a fraction of Sales.


# Part 4. Insights from visualizations:
"""
Insights:
- The bar plot shows whether some regions are more profitable on average than others.
- The KDE comparison shows the scale difference between Sales and Profit.
- In our synthetic dataset, the distribution is relatively uniform across regions.
"""
