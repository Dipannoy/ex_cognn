import pandas as pd
import os
import csv
df = pd.read_csv('jack_poly_eg_082824.csv')

tr_list = []

# Directory path
directory = './poly_eg_bcast_graphs_th0.5_train_val_test'

# List all filenames in the directory
filenames = os.listdir(directory)
header = ['filename','label','id']
id=0
for i in df.index:
    td = []
    if(df['filename'][i].split('.')[0]+'_bcast.bin' in filenames):
        td.append(df['filename'][i])
        td.append(df['label'][i])
        td.append(id)
        id=id+1
        tr_list.append(td)

with open('jack_poly_eg_mod_graph.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(tr_list)



    
