import pandas as pd

# Load the summarizer output file from the 'Data' sheet
# Use header=4 (5th row, 0-indexed) and usecols to select D to AW columns

df = pd.read_excel(
    "Private Network Tracker - Masterfile for support team 1Q25 - 23-April-25_vOL.xlsx",
    sheet_name="Data",
    header=4,
    usecols="D:AW"
)

# Print the columns of the dataframe to verify
print(df.columns.tolist())

tech_cols = [
    'Private 5G', 'Private LTE', 'CBRS', 'MulteFire',
    'Network Slicing', 'Fixed Wireless Access (FWA)'
]

use_case_cols = ['IoT', 'Enterprise workforce']
add_tech_cols = ['Edge computing', 'Slice', 'AI', 'Other']

# Convert ticked (e.g., 1) to 1, everything else (including NaN) to 0 for all relevant columns
for col in tech_cols + use_case_cols + add_tech_cols:
    df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)

# Create 'Technology' column as a concatenation of ticked techs
df['Technology'] = df[tech_cols].apply(
    lambda row: ', '.join([col for col, val in row.items() if val == 1]), axis=1
)

# Create 'Use cases' column as a concatenation of ticked use case columns
df['Use cases'] = df[use_case_cols].apply(
    lambda row: ', '.join([col for col, val in row.items() if val == 1]), axis=1
)

# Create 'Additional technologies' column as a concatenation of ticked additional tech columns
df['Additional technologies'] = df[add_tech_cols].apply(
    lambda row: ', '.join([col for col, val in row.items() if val == 1]), axis=1
)

# Now you can proceed with further processing, e.g. dropna, save, etc.
print(df[tech_cols + use_case_cols + add_tech_cols + ['Technology', 'Use cases', 'Additional technologies']].head(10))
df.to_csv("training_data_for_classifier.csv", index=False)