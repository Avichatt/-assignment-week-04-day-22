import pandas as pd


df = pd.read_csv('sales_data.csv')

# Part C — Interview Ready


print("--- Q1: loc vs iloc ---")
print("""
Answer:
- loc is label-defined selection. You can use the explicit row/column names.
- Example: df.loc[10:20, 'Category'] selects rows 10 to 20 and the 'Category' column.
- Note that loc includes the last element (it's inclusive on BOTH sides of a slice).

- iloc is integer-defined selection (index labels are ignored).
- Example: df.iloc[10:20, 2] selects the 11th through 20th rows and the 3rd column.
- Note that iloc follows standard Python slicing rules: it does NOT include the end index.
""")



print("\n--- Q2: Filtering rows with value > average (Sales) ---")
average_sales = df['Sales'].mean()
print(f"Average Sales: {average_sales:.2f}")

higher_than_average_rows = df[df['Sales'] > average_sales]
print(f"Number of rows with higher-than-average sales: {len(higher_than_average_rows)}")
print(higher_than_average_rows.head())


# Q3 — What is the purpose of describe()? What insights can we get from it?
print("\n--- Q3: Purpose of describe() ---")
print("""
Answer:
- describe() provides summary statistics for all numerical columns in your DataFrame.
- We can get insights such as:
    1. Mean - the average value of each column.
    2. Std (Standard Deviation) - how spread out the data points are.
    3. Min / Max - the range of the values.
    4. 25%, 50%, 75% quartiles - help to identify data distribution and outliers.
    5. Count - to see if any values are missing across columns.
""")
