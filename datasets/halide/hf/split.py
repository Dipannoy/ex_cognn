import pandas as pd
import csv

df = pd.read_csv('oxide_hf.csv')



# df_len = len(df.index)

# tr_list = []
# id = 1

# header = ['compound','label','id']
# for i in df.index:
#     tr=[]
#     tr.append(df['file_name'][i].split('.')[0])
#     tr.append(df['Ef'][i])
#     tr.append(id)
#     id = id+1
#     tr_list.append(tr)


# with open('hf_seen_dp_hf_id.csv', 'w') as f: 
#     write = csv.writer(f) 
#     write.writerow(header) 

#     write.writerows(tr_list) 

tr_list = []


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.90)
test_2, val = train_test_split(test,test_size=0.05)

# print(train)


for i in val.index:
    tr=[]
    tr.append(val['id'][i])
    tr.append(val['compound'][i])
    
    tr_list.append(tr)


with open('./10_5_85/val_ids.csv', 'w') as f: 
    write = csv.writer(f) 
    # write.writerow(header) 

    write.writerows(tr_list) 


tr_list = []
for i in train.index:
    tr=[]
    tr.append(train['id'][i])
    tr.append(train['compound'][i])
    
    tr_list.append(tr)


with open('./10_5_85/train_ids.csv', 'w') as f: 
    write = csv.writer(f) 
    # write.writerow(header) 

    write.writerows(tr_list) 


tr_list = []

for i in test_2.index:
    tr=[]
    tr.append(test_2['id'][i])
    tr.append(test_2['compound'][i])
    
    tr_list.append(tr)


with open('./10_5_85/test_ids.csv', 'w') as f: 
    write = csv.writer(f) 
    # write.writerow(header) 

    write.writerows(tr_list) 
