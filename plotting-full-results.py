import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
# Replace 'your_file.csv' with the actual name of your file
df = pd.read_csv('simulation_results_20240601.csv')

# Convert the Date column to datetime objects for proper time-series plotting
df['Date'] = pd.to_datetime(df['Date'])

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each revenue stream
plt.plot(df['Date'], df['Total Revenue'].cumsum(), label='Total Revenue', linewidth=2)
plt.plot(df['Date'], df['DA Revenue'].cumsum(), label='DA Revenue', linestyle='--')
plt.plot(df['Date'], df['RT Revenue'].cumsum(), label='RT Revenue', linestyle=':')

# Add labels and title
plt.title('Revenue Streams Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)

# Improve readability
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save and show the plot
plt.savefig('revenue_plot.png')
plt.show()