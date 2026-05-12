import pickle
import time
from tqdm import tqdm
from pymatgen.core import Structure, Lattice, Element
import numpy as np
import os
import pandas as pd
from collections import OrderedDict

def create_crystal_graphs(files, df):
    #struct_edges = {}
    data_list = []
    df_files = df['filename'].tolist()
    for i in tqdm(range(len(df_files))):
        if df_files[i]+'.cif' in files:
            file_path = os.path.join(folder_path, df_files[i]+'.cif')
            with open(file_path, 'r') as f:
                cif = f.read()
                struct = Structure.from_str(cif, 'cif')
                data = {}
                data['jid'] = df_files[i][:-4]
                data['system'] = df_files[i][:-4]
                d = OrderedDict()
                s = struct.as_dict()
                d['lattice_mat'] = s['lattice']['matrix']
                d['coords'] = [site['abc'] for site in s['sites']]
                d['elements'] = [site['species'][0]['element'] for site in s['sites']]
                d['abc'] = [s['lattice']['a'], s['lattice']['b'], s['lattice']['c']]
                d['angles'] = [s['lattice']['alpha'], s['lattice']['beta'], s['lattice']['gamma']]
                d['cartesian'] = False
                d['props'] = ['']*len(d['elements'])
                data['atoms'] = d
                data['nf'] = [site.species.elements[0].number for site in struct.sites]
                data['Ef'] = float(df[df['filename'] == df_files[i]]['label'].values)
                natoms = len(d['elements'])
                data['extra_atom_features.atomic_radius'] = [Element(z).atomic_radius for z in d['elements']]
                # data['extra_atom_features.atomic_orbitals'] = [list(Element(z).atomic_orbitals.values())[-1] for z in d['elements']]
                
                data['global_features.lattice'] = d['lattice_mat']
                data_list.append(data)
    return data_list

folder = '../datasets/jarvis/'
folder_path = folder + 'jarvis_all_cifs'
files = os.listdir(folder_path)



df_perov = pd.read_csv(folder + 'jarvis_lg_ef.csv')
data_list = create_crystal_graphs(files, df_perov)
print(len(data_list))

with open('./jarvis_lg_cell_ef.pickle', 'wb') as handle:
    pickle.dump(data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('DONE')



