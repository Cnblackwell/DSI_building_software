import pytest

def test_load_data():
    path_url = 'https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/ea1d6e57-87af-4e23-b722-6c1f5aa18a8d/resource/815aedb5-f9d7-4dcd-a33a-4aa7ac5aac50/download/Dinesafe.csv'
    dine_safe_TO = load_data(path_url)
    assert isinstance(dine_safe_TO, pd.DataFrame)
