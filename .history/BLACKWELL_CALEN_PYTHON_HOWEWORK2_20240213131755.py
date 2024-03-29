#
import yaml
import argparse
import matplotlib.pyplot as plt
import pandas as pd

#Argparse code:
import argparse
import sys

parser = argparse.ArgumentParser(description='DineSafe TO trends: Establishments and Fines')
parser.add_argument('--title', '-t', type=str, help='Plot title')
parser.add_argument('--output_file', '-o', type=str, help='Output plot filename',default="DineSafe_TO_Analysis")

# Parse only known arguments, ignoring any unrecognized arguments
args, _ = parser.parse_known_args()

# Access the arguments using args.title and args.output_file
print("Title:", args.title)
print("Output file:", args.output_file)


#Config files
config_files = ['systemconfig.yml', 'jobconfig.yml']
config = {}

for this_config_file in config_files:
    with open(this_config_file, 'r') as yamlfile:
        this_config = yaml.safe_load(yamlfile)
        config.update(this_config)


# Load the data to a single DataFrame, using the config files
dine_safe_TO = pd.read_csv(config['dataset'])

# Let's combine them and ensure that the plot includes labels, a title, a grid, and a legend:

plt.scatter(dine_safe_TO['amount_fined'],dine_safe_TO['inspection_date'])
plt.title(args.title)
#adds x-axis label  
plt.ylabel(config['plot_config']['ylabel']) 
#adds y-axis label 
plt.xlabel(config['plot_config']['xlabel'])
#adds legend , with proper positioning
#adds grid with an alpha value of 0.8
plt.savefig(f'{args.output_file}.png')

# - END - cnb


