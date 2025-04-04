from census import Census
import requests
from us import states
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================================================================
# Data Extraction
# ======================================================================================================================
# Prepare for the API call
api_key = "652ba082c38062b1b88301e1f2070a36aa21ff92"
year = 2023
variables_of_interest = ['SEX', 'AGEP', 'PINCP']
var_string = ','.join(variables_of_interest)
var_string = "&".join(var_string.rsplit(",", 1)) # substitute the last "," with "&"

# Define the API url
acs1 = f'https://api.census.gov/data/{year}/acs/acs1/pums?get={var_string}&key={api_key}'
acs1_var_link = f'https://api.census.gov/data/{year}/acs/acs1/pums/variables.json'

print(acs1)

# Make the API call
acs1_data = requests.get(acs1).json()
acs1_var = requests.get(acs1_var_link).json()

# Convert JSON list to a DataFrame
df = pd.DataFrame(acs1_data[1:], columns=acs1_data[0])

# Convert the variable names to a DataFrame
var_df = pd.DataFrame(acs1_var['variables']).T.reset_index().sort_values(by='index').reset_index(drop=True)
# Customize var_df to contain only the relevant variables
var_list = var_df[var_df['index'].isin(variables_of_interest)]

print(df.head())

# ======================================================================================================================
# Data Cleaning
# ======================================================================================================================
# convert the data types into numeric
df = df.apply(pd.to_numeric, errors='coerce')
print(df.dtypes)

# rename the columns
df['SEX'] = df['SEX'].replace({1: 'Male', 2: 'Female'})
df['SEX'] = pd.Categorical(df['SEX'], categories=['Male', 'Female'], ordered=True)

# remove missing values by replacing -19999 income with NaN
df['PINCP'] = df['PINCP'].replace(-19999, pd.NA)
print(df['PINCP'].isna().sum())
df = df.dropna(subset=['PINCP'])

print(f'Maximum income: {df["PINCP"].max()}; Minimum income: {df["PINCP"].min()}')
print(f'Maximum age: {df["AGEP"].max()}; Minimum age: {df["AGEP"].min()}')

# ======================================================================================================================
# Plot the data
# ======================================================================================================================
# plot the income as a function of age
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='AGEP', y='PINCP', hue='SEX', estimator='mean', errorbar=('ci', 99))
plt.title('Income as a function of Age')
plt.xticks(range(df["AGEP"].min(), df["AGEP"].max(), 5))
plt.xlim(df["AGEP"].min(), df["AGEP"].max())
plt.xlabel('Age')
plt.ylabel('Income')
plt.savefig('income.png', dpi=600)
plt.show()


