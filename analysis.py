# Define the path to the CSV file
csv_path = '/Users/kavith/Projects/3data/data/sample_sales.csv'

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv(csv_path)

# Group by product_name and sum the sales
product_sales = df.groupby('product_name')['sales'].sum().reset_index()

# Sort products by total sales in descending order
top_products = product_sales.sort_values('sales', ascending=False).head(3)

# Print the top 3 products by total sales
print("Top 3 Products by Total Sales:")
print(top_products)

# Create a bar chart to visualize the top products
plt.figure(figsize=(10, 6))
plt.bar(top_products['product_name'], top_products['sales'])
plt.title('Top 3 Products by Total Sales')
plt.xlabel('Product')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()