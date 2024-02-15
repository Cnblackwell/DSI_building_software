
#Loading pandas

import pandas as pd

#display all columns
pd.set_option("display.max_columns", None)

#Original code from Python Assignment 2
dine_safe_TO = pd.read_csv('/Users/cnblackwell/Desktop/myproject/DSI_building_robust_software/building_software_homework/Dinesafe.csv')
dine_safe_TO = dine_safe_TO.rename(columns={'Rec_#':'record_number'})


#Refactored
import pandas as pd
import pytest

dine_safe_TO_url = 'https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/ea1d6e57-87af-4e23-b722-6c1f5aa18a8d/resource/815aedb5-f9d7-4dcd-a33a-4aa7ac5aac50/download/Dinesafe.csv'


dine_safe_TO.columns #show the CVS file columns


def load_and_rename(dine_safe_TO):
    dine_safe_TO = pd.read_csv(dine_safe_TO_url)
    if 'Rec #' not in dine_safe_TO.columns:
        raise ValueError('Rec # column not found, please try again!')
    dine_safe_TO.rename({'Rec #': 'record_number'}, inplace=True)

    return dine_safe_TO

dine_safe_TO = load_and_rename('refactored_data.csv')


