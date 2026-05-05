import pandas as pd

df = pd.read_csv('jack_poly_eg_082824.csv')

tr_list = []


for i in df.index:
    td = []
    
