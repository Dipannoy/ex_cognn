import pickle
import pandas as pd
import re
import csv
from sklearn.metrics import mean_absolute_error as mae

with open('train_oxides_4_predictions.pkl', 'rb') as file:
    data_train = pickle.load(file)

with open('val_oxides_4_predictions.pkl', 'rb') as file:
    data_val = pickle.load(file)

with open('test_oxides_4_predictions.pkl', 'rb') as file:
    data_test = pickle.load(file)

with open('ef_9760.pickle', 'rb') as file:
    data_all = pickle.load(file)


# Now, 'data' contains the contents of the pickle file

data_all_dict = {}
tr_list=[]
for d in data_all:
    data_all_dict[d['jid']] = d['Ef']

for d in data_train:
    td=[]
    real = data_all_dict[d]
    pred = data_train[d].item()
    err = mae([real],[pred])
    td.append(err)
    td.append(d)
    td.append('train')
    tr_list.append(td)

for d in data_val:
    td=[]
    real = data_all_dict[d]
    pred = data_val[d].item()
    err = mae([real],[pred])
    td.append(err)
    td.append(d)
    td.append('train')
    tr_list.append(td)
 
for d in data_test:
    td=[]
    real = data_all_dict[d]
    pred = data_test[d].item()
    err = mae([real],[pred])
    td.append(err)
    td.append(d)
    td.append('test')
    tr_list.append(td)

header = ['error', 'compound','sample']


with open('aabbx6_ef_error_per_compound_4.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerow(header) 

    write.writerows(tr_list) 

