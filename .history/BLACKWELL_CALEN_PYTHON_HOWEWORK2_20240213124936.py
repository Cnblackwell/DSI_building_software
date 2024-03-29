#
import yaml
import argparse
import matplotlib.pyplot as plt
import pandas as pd

#Argparse code:

parser = argparse.ArgumentParser(description='DineSafe TO trends: Establishments and Fines')
parser.add_argument('--title', '-t', type=str, help='Plot title')
parser.add_argument('--output_file', '-o', type=str, help='Output plot filename')
args = parser.parse_args()


#Config files
config_files = ['systemconfig.yml', 'jobconfig.yml']
config = {}

for this_config_file in config_files:
    with open(this_config_file, 'r') as yamlfile:
        this_config = yaml.safe_load(yamlfile)
        config.update(this_config)


# %%

# Load the data to a single DataFrame, using the config files
dine_safe_TO = pd.read_csv(config['dataset'])



# Let's combine them and ensure that the plot includes labels, a title, a grid, and a legend:

def plot_scatter(df,yvar,xvar,title):
    plt.scatter(df[yvar],df[xvar])
    plt.title(title)

plt.scatter(dine_safe_TO['amount_fined'],dine_safe_TO['inspection_date'])
plt.title(args.title)
#adds x-axis label  
plt.xlabel(config['plot_config']['xlabel']) 
#adds y-axis label 
plt.ylabel(config['plot_config']['ylabel'])

plt.savefig(f'{args.output_file}.png')


# - END - cnb


