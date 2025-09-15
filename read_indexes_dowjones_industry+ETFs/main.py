import pandas as pd

# Read each CSV file
etfs = pd.read_csv("ETFs.csv")                     # Long Name, Symbol
mag7 = pd.read_csv("mag7.csv")                     # Long Name, Symbol
dow = pd.read_csv("dowjones_ind_index.csv")        # Long Name, Symbol
myetfs = pd.read_csv("myETFs.csv")                 # Long Name, Symbol

# Clean data: drop rows where Symbol is missing
etfs = etfs[etfs['Symbol'].notna() & (etfs['Symbol'] != '')]
mag7 = mag7[mag7['Symbol'].notna() & (mag7['Symbol'] != '')]
dow = dow[dow['Symbol'].notna() & (dow['Symbol'] != '')]
myetfs = myetfs[myetfs['Symbol'].notna() & (myetfs['Symbol'] != '')]

# Concatenate all lists
combined = pd.concat([etfs, mag7, dow, myetfs], ignore_index=True)

# Remove duplicates by Symbol, keep the first occurrence
combined = combined.drop_duplicates(subset='Symbol', keep='first')

# Save to CSV
combined.to_csv('Combined_ETFs.csv', index=False)

