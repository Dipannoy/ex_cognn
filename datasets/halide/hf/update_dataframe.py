import pandas as pd

df = pd.read_csv('hf_seen_dp_hf_id.csv')

df['filename'] = df['filename']+ '.cif'

df.to_csv('hf_seen_dp_hf_id.csv',index=False)
