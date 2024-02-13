

# %%
#Import YAML, argparse, matplotlib, logging

import yaml
import matplotlib.pyplot as plt
import argparse
import logging

# %%
#Loading pandas

import pandas as pd

#display all columns
pd.set_option("display.max_columns", None)

# %% NEW
#Argparse code:
import argparse
import sys

parser = argparse.ArgumentParser(description='DineSafe TO trends: Establishments and Fines')
parser.add_argument('--title', '-t', type=str, help='Plot title')
parser.add_argument('--output_file', '-o', type=str, help='Output plot filename')
parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose mode')

# Parse the arguments
args = parser.parse_args()

# Access the arguments using args.title and args.output_file
print("Title:", args.title)
print("Output file:", args.output_file)
print("Verbose mode:", args.verbose)


logging.basicConfig(
    handlers=(logging.StreamHandler(), logging.FileHandler('DineSafeTO.log')), 
    level=logging.INFO,
    )

dataset_url = 'https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/ea1d6e57-87af-4e23-b722-6c1f5aa18a8d/resource/815aedb5-f9d7-4dcd-a33a-4aa7ac5aac50/download/Dinesafe.csv'

try:
    dine_safe_TO = pd.read_csv(dataset_url)
    logging.info(f'Successfully loaded {dataset_url}')
except Exception as e:
    logging.error('Error loading dataset from {dataset_url}:{e}')
    raise e

# %% NEW
#Config files
config_files = ['systemconfig.yml', 'jobconfig.yml']
config = {}

for this_config_file in config_files:
    with open(this_config_file, 'r') as yamlfile:
        this_config = yaml.safe_load(yamlfile)
        config.update(this_config)


# %%
#PREVIOUSLY
# Getting started
# Task 1: Load the data to a single DataFrame
#dine_safe_TO = pd.read_csv('/Users/cnblackwell/Desktop/DSI_Materials/python_data/Dinesafe.csv') 

#        
# Task 1: Load the data to a single DataFrame, using the config files
dine_safe_TO = pd.read_csv(config['dataset'])

# %%
# Task 2: Profile the DataFrame
# Showing column names,using the info() method.     

dine_safe_TO.info()


# %%
# Data types when loaded
print('The data types are:\n', 
      dine_safe_TO.dtypes)

# %%
# This shows the NaNs in each columns

dine_safe_TO.isna().sum()

# %%
# Shape of the DataFrame
print("The shape of the DataFrame:\n", 
      dine_safe_TO.shape)

# %%
# Task 3: Generate summary statistics


# #rename all columns to lower case
dine_safe_TO = dine_safe_TO.rename(columns=str.lower)


# %%
dine_safe_TO.head() # Now, all columns are in lower case

# %%
# For numeric columns: What are the max, min, mean, and median?

dine_safe_TO.describe()

# %%
# Of note, there are non-numeric (text) columns in this Data Frame
    # For text columns: What is the most common value? How many unique values are there?
    # For these questions, we can use the describe(include='object') method.
    # After running this code, we can see how many unique values ('unique') and the most common values in each column.

dine_safe_TO.describe(include='object')

# %%
# Task 4: Rename one or more columns in the DataFrame
# We will rename the columns so that we replace spaces with underscores as well as convert all letters to lowercase.

def clean_names(string):
    return string.lower().replace(' ', '_')

print(list(dine_safe_TO)) # prints the original column names in a list

dine_safe_TO = dine_safe_TO.rename(columns=clean_names)

print(list(dine_safe_TO)) # prints the new, renamed column names in a list


# %%
# Task 4a: To rename one column, we can also do the following:

print(list(dine_safe_TO)) # original column list

dine_safe_TO = dine_safe_TO.rename(columns={'rec_#':'record_number'})

print(list(dine_safe_TO)) # notice that 'rec_#' has been renamed to 'record_number'

# %%
# Task 5: Select a single column and find its unique values

dine_safe_TO['establishment_type'].unique()


# %%
# Task 6: Select a single text/categorical column and find the counts of its values.

dine_safe_TO['infraction_details'].value_counts()

# %%
# Task 7: Convert the data type of at least one of the columns. 
# If all columns are typed correctly, convert one to str and back.

dine_safe_TO['inspection_date'] = pd.to_datetime(dine_safe_TO['inspection_date'])
dine_safe_TO['inspection_date']

# this will produce an output showing that 'inspection_date' is now in the datetime64 format.

# %%
# Convert column 'establishment_name' from object to string
     #this shows that the 'establishment_name'column is an 'object' type in its original form

dine_safe_TO['establishment_name']

# %%
# Conversion:

dine_safe_TO['establishment_name'] = dine_safe_TO['establishment_name'].astype("string")

# %%
dine_safe_TO['establishment_name'] # now the type has been changed to 'string' type.

# %%
# Convert the same column above back to object

dine_safe_TO['establishment_name'] = dine_safe_TO['establishment_name'].astype("object")

# %%
dine_safe_TO['establishment_name'] #shows the column is now an 'object' type.

# %%
# Task 8: Write the DataFrame to a different file format

dine_safe_TO.to_excel('../dine_safe_TO_modified.xlsx', index=False)

# %%
#Data Wrangling

#Load the Excel sheet from above.
dsTO = pd.read_excel('../dine_safe_TO_modified.xlsx', sheet_name=None)

# %%
# Task 1 :Create a column derived from an existing one. Some possibilities:
# Bin a continuous variable
# Extract a date or time part (e.g. hour, month, day of week)
# Assign a value based on the value in another column (e.g. TTC line number based on line values in the subway delay data)
# Replace text in a column (e.g. replacing occurrences of "Street" with "St.")


df_DineSafe = pd.DataFrame()

for sheet_name, sheet_contents in dsTO.items():
   df_DineSafe = pd.concat([df_DineSafe, sheet_contents],
                              axis=0,
                              ignore_index=True)

df_DineSafe


# %%
# Create a 'city' column to the left of 'establishment_address' column

df_DineSafe.insert(7,'city','Toronto') #adds 'city' column with the value 'Toronto' 
df_DineSafe

# %%
#Task 2: Remove one or more columns
#In this sample, we will remove the 'inspection_id' column

df_DineSafe=df_DineSafe.drop('inspection_id',axis=1)

df_DineSafe # note that the 'inspection_id' column has been removed




# %%
# Task 3: Extract a subset of columns and rows to a new DataFrame

# using the query() method:

subset_df_DineSafe = df_DineSafe.query('severity.isna()').head()

subset_df_DineSafe #Notice that the 'severity' column shows the first 5 values 

# %%
# using the .loc() method
# this will show the same result as above but using the .loc() method

subset_df_DineSafe = df_DineSafe.loc[df_DineSafe['severity'].isna()].head()

subset_df_DineSafe 

# %%
# Task 4 : Investigate null values
#Create and describe a DataFrame containing records with NaNs in any column

new_df_DineSafe = df_DineSafe.loc[df_DineSafe['infraction_details'].isna()].head()

new_df_DineSafe


#If it makes sense to drop records with NaNs in certain columns from the original DataFrame, do so.

# %%
# Create and describe a DataFrame containing records with NaNs in a subset of columns
# This will show the NaN's in 'infraction_details' with the subset columns 'establishment_name',
# 'inspection_date', and 'outcome'.

new_df_DineSafe = df_DineSafe.loc[df_DineSafe['infraction_details'].isna(),
                                  ['establishment_name', 'inspection_date', 'outcome', 
                                   'infraction_details']].head()

new_df_DineSafe




# %%
# Grouping and aggregating

# Task 1: Use groupby() to split your data into groups based on one of the columns.

est_type_DineSafe = df_DineSafe.groupby('establishment_type')




# %%
# How much were each establishment type fined based on the data?

est_type_DineSafe['amount_fined'].sum()

# %%
# Let's look at the size of establish_type column
est_type_DineSafe.size()

# %%
# This will show the summary the total number of each establishment type, and the
# corresponding amount of fines on a specific inspection date.

# Convert date first, to fix its format:
df_DineSafe['inspection_date']=pd.to_datetime(df_DineSafe['inspection_date'])

# Then use groupby() and agg() methods: 
df_DineSafe_summary = (df_DineSafe
                       .groupby('inspection_date')
                       .agg(est_type=('establishment_type','max'),
                            num_establishment=('establishment_type','count'),
                            total_fines=('amount_fined','sum')))
df_DineSafe_summary
                                         

# %%
# Save the processed file to a new csv file

df_DineSafe_summary.to_csv('../df_DineSafe_summary.csv',index=False)

# %%
# This is a jupyter-specific "magic" command to render plots in-line
%matplotlib inline
import matplotlib.pyplot as plt

# Let's plot the Dine Safe Summary data below:

df_DineSafe_summary.plot(subplots=True)

# %%

# PREVIOUSLY

df_DineSafe_summary.plot()

#adds title
plt.title('No. of Establishments with Fines ($) vs. Inspection Date')
#adds x-axis label  
plt.xlabel('Inspection Date') 
#adds y-axis label 
plt.ylabel('Amount in number or $')
#adds legend , with proper positioning
plt.legend(['No. of Establishments','Total Fines in $CAD'],
           bbox_to_anchor=(1,1),loc='upper left')
#adds grid with an alpha value of 0.8
plt.grid(alpha=0.8)

#Ta-dah!
plt.show()

# NEW 
# Plot and visualize using config and argparse, incorporated

plt.scatter(dine_safe_TO['amount_fined'], dine_safe_TO['inspection_date'], color=config['plot_config']['color'])
plt.title(args.title)
plt.ylabel(config['plot_config']['ylabel'])  # adds x-axis label
plt.xlabel(config['plot_config']['xlabel'])  # adds y-axis label
plt.grid(alpha=0.8)  # adds grid with an alpha value of 0.8
plt.show()  # Show the plot


#Save plot image to file as PNG
plt.savefig(f'{args.output_file}.png')

# %%


# %% [markdown]
# - END - cnb


