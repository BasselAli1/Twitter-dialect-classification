import pandas as pd

dataset = pd.read_csv('preprocessed_data.csv')

dataset['dialect_number'] = dataset['dialect'].factorize()[0]
outputs = dict(zip(dataset['dialect_number'], dataset['dialect']))
country_codes = {'IQ':'Iraq',
'BH':'Bahrain',
'KW':'Kuwait',
'SA':'Saudi Arabia',
'AE':'United Arab Emirates',
'OM':'Oman',
'QA':'Qatar',
'YE':'Yemen',
'SY':'Syrian Arab Republic',
'JO':'Jordan',
'PL':'Palestinian',
'LB':'Lebanon',
'EG':'Egypt',
'SD':'Sudan',
'LY':'Libya',
'TN':'Tunisia',
'DZ':'Algeria',
'MA':'Morocco'}