import pickle
import pandas as pd
import re
import csv
from sklearn.metrics import mean_absolute_error as mae

with open('train_poly_struct_poly_eg_2_predictions.pkl', 'rb') as file:
    data_train = pickle.load(file)

with open('test_poly_struct_poly_eg_2_predictions.pkl', 'rb') as file:
    data_test = pickle.load(file)

with open('jack_poly_eg_poly_struct.pickle', 'rb') as file:
    data_all = pickle.load(file)


# Now, 'data' contains the contents of the pickle file

data_all_dict = {}
tr_list=[]
#print(data_all[0])
for d in data_all:
    data_all_dict[d['jid']] = d['poly_Eg']

#print(data_train)

for d in data_train:
    #td=[]
    real = data_all_dict[d]
    pred = data_train[d].item()
    err = mae([real],[pred])
    if '_' in d:
        d = d.replace('_','')
    #print(d)
    ellist = re.findall('[A-Z][^(A-Z)(0-9)]*', d)
    #print(ellist)
    for e in ellist:
        td=[]
        td.append(e)
        td.append(err)
        td.append('train')
        tr_list.append(td)
 
for d in data_test:
    #td=[]
    real = data_all_dict[d]
    pred = data_test[d].item()
    err = mae([real],[pred])
    if '_' in d:
        d = d.replace('_','')
    ellist = re.findall('[A-Z][^(A-Z)(0-9)]*', d)
    for	e in ellist:
       	td=[]
       	td.append(e)
       	td.append(err)
       	td.append('test')
       	tr_list.append(td)

header = ['element','error','sample']


with open('jack_poly_struct_poly_eg_sm_0210_element_wise_error.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerow(header) 

    write.writerows(tr_list) 
