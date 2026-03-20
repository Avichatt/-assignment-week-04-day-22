# Part D — AI-Augmented Task

# 1. Prompt AI: 
# "Explain how to perform data analysis using Pandas and visualization using Matplotlib with examples."

# 2. Document prompt and output:
prompt = "Explain how to perform data analysis using Pandas and visualization using Matplotlib with examples."

output = """
Data analysis using Pandas and visualization with Matplotlib is a foundational skill in data science. 
Here's a quick guide:

1. Data Analysis with Pandas:
Use `df = pd.read_csv('data.csv')` to load your data.
You can use `df.head()` to see the first few rows, `df.info()` for column types, 
and `df.describe()` for statistics.
Filtering: `df[df['age'] > 30]` returns all rows where age is over 30.
Aggregation: `df.groupby('city')['salary'].mean()` calculates the average salary per city.

2. Visualization with Matplotlib:
First, import the library: `import matplotlib.pyplot as plt`.
Create a basic line plot: `plt.plot(df['index'], df['value'])` and `plt.show()`.
Create a bar plot: `plt.bar(names, scores)`.
Add labels and a title: `plt.xlabel('X-axis')`, `plt.ylabel('Y-axis')`, `plt.title('My Plot')`.

Example Code:
import pandas as pd
import matplotlib.pyplot as plt

# Loading data
data = {'name': ['A', 'B', 'C'], 'age': [25, 30, 35], 'score': [88, 92, 95]}
df = pd.DataFrame(data)

# Mean age
print("Mean Age:", df['age'].mean())

# Plotting
plt.bar(df['name'], df['score'])
plt.title('Scores by Name')
plt.show()
"""

print(f"--- Prompt ---\n{prompt}\n")
print(f"--- AI Output ---\n{output}\n")


# 3. Evaluate:
print("--- Evaluation ---")
"""
Evaluation Summary:
- Are plots correct? 
  Yes, the Matplotlib example uses the standard bar plot function correctly 
  with appropriate title and display calls.
- Is the explanation meaningful? 
  Yes, it covers loading, basic analysis like head/info/describe, filtering, 
  and grouping before showing how to visualize with plot/bar and adding labels.
"""
