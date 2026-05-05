import pandas as pd

df = pd.read_csv('pbe_eg.csv')


df['id'] = range(len(df))

df.to_csv('pbe_eg_id.csv',index=False)