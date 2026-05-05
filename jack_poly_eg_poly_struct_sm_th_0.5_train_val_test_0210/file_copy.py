import pickle
import os
import shutil



# Copying the file along with metadata
# Open the pickle file in binary read mode
with open('train_poly_struct_poly_eg_2_predictions.pkl', 'rb') as file:
    data_train = pickle.load(file)
with open('test_poly_struct_poly_eg_2_predictions.pkl', 'rb') as file:
    data_test = pickle.load(file)
with open('val_poly_struct_poly_eg_2_predictions.pkl', 'rb') as file:
    data_val = pickle.load(file)

# # Now 'data' contains the deserialized object
# print(data)

#print(len(data_train))
#print(len(data_test))
#print(len(data_val))


for d in data_val:
    source = './val_poly_struct_poly_eg_2_graphs/'+d+'_bcast.bin'
    destination = './poly_eg_bcast_graphs_th0.5_train_val_test/'
    shutil.copy2(source, destination)

for d in data_train:
    source = './train_poly_struct_poly_eg_2_graphs/'+d+'_bcast.bin'
    destination = './poly_eg_bcast_graphs_th0.5_train_val_test/'
    shutil.copy2(source, destination)

for d in data_test:
    source = './test_poly_struct_poly_eg_2_graphs/'+d+'_bcast.bin'
    destination = './poly_eg_bcast_graphs_th0.5_train_val_test/'
    shutil.copy2(source, destination)

