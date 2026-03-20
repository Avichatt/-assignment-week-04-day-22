import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Used for KDE plots as they look much better

# Load dataset
df = pd.read_csv('sales_data.csv')

# Part 1. Data Selection using Pandas
print("--- Part 1. Data Selection using loc and iloc ---")

# (a) Use loc (label-based selection)
# Example 1: Selecting a specific row by index and a column
print("\nloc Example 1: Profit of row 5")
print(df.loc[5, 'Profit'])

# Example 2: Selecting multiple rows and specific columns
print("\nloc Example 2: First 3 rows with Sales and Region")
print(df.loc[0:2, ['Sales', 'Region']]) 

# Example 3: Slicing rows and columns
print("\nloc Example 3: Rows 10 to 12, Category to Region")
print(df.loc[10:12, 'Category':'Region'])

# (b) Use iloc (index-based selection)
# Example 1: Selecting a specific element
print("\niloc Example 1: Value at row 0, column 0")
print(df.iloc[0, 0])

# Example 2: Slicing rows and columns by integer index
print("\niloc Example 2: Rows 5 to 7, first 3 columns")
print(df.iloc[5:8, 0:3])

# Example 3: Using negative indexing
print("\niloc Example 3: Last 2 rows, last 2 columns")
print(df.iloc[-2:, -2:])


# Part 2. Filtering Data
print("\n--- Part 2. Filtering Data ---")

# Filter rows based on multiple conditions (e.g., Sales > 4000 and Category is 'Electronics')
high_sales_electronics = df[(df['Sales'] > 4000) & (df['Category'] == 'Electronics')]
print(f"Number of high-sale electronics transactions: {len(high_sales_electronics)}")

# Extract subset: Profit > 1000
high_profit_data = df[df['Profit'] > 1000]
print(f"Rows with Profit > 1000: {len(high_profit_data)}")


# Part 3. Descriptive Statistics
print("\n--- Part 3. Descriptive Statistics ---")
print(df.describe())

"""
Manual Interpretation:
- Mean (Sales): approx 2500 - indicating the average transaction size.
- Std (Profit): indicates the variability or spread of profits across transactions.
- Min (Sales): 100 - the lowest value in our generated dataset.
- Max (Sales): 4999 - the highest value in our generated dataset.
"""


# Part 4. Histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Sales'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sales')
plt.xlabel('Sales Value')
plt.ylabel('Frequency')
plt.show() # In a real environment, save figures or use interactive plots.

# Interpretation of Histogram:
# The distribution looks fairly uniform because we generated it with random values. 
# It shows have a wide spread from 100 to 5000.


# Part 5. Bar Plot (Categorical vs Numerical)
# We can aggregate first: Total Sales by Category
category_sales = df.groupby('Category')['Sales'].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(category_sales['Category'], category_sales['Sales'], color='orange')
plt.title('Total Sales by Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.show()

# Interpretation: 
# Compares categories like Electronics, Fashion, etc. to see which performed best overall.


# Part 6. Line Chart (Trend over index)
# We can sort by date and plot the trend of Profit
df_sorted = df.sort_values(by='Date').reset_index()
plt.figure(figsize=(10, 6))
plt.plot(df_sorted.index, df_sorted['Profit'], color='green', linewidth=0.5)
plt.title('Profit Trend over Time (Index-based)')
plt.xlabel('Time Step')
plt.ylabel('Profit')
plt.show()


# Part 7. KDE Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Profit'], fill=True, color='red')
plt.title('KDE Plot of Profit')
plt.xlabel('Profit')
plt.ylabel('Density')
plt.show()

# Interpretation of KDE:
# Compared to the histogram, the KDE provides a smooth curve representing the probability 
# distribution, highlighting the peaks and density more clearly.
