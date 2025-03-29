# Define the path to the CSV file
csv_path = '/Users/kavith/Projects/3data/data/sample_sales.csv'

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv(csv_path)

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Group by date and sum the quantity sold
sales_by_date = df.groupby('date')['quantity'].sum().reset_index()

# Sort by quantity in descending order
sales_by_date_sorted = sales_by_date.sort_values('quantity', ascending=False)

# Get the date(s) with the highest number of sales
max_quantity = sales_by_date_sorted['quantity'].max()
top_dates = sales_by_date_sorted[sales_by_date_sorted['quantity'] == max_quantity]

# Print the results
print("Date(s) with the most number of sales:")
print(top_dates)
print(f"\nHighest quantity sold on a single date: {max_quantity} units")

# Create a bar chart to visualize sales by date
plt.figure(figsize=(12, 6))
plt.bar(sales_by_date['date'].dt.strftime('%Y-%m-%d'), sales_by_date['quantity'])
plt.title('Number of Items Sold by Date')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight the top date(s)
for date in top_dates['date']:
    plt.axvline(x=date.strftime('%Y-%m-%d'), color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()