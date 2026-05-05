from pymatgen.io.cif import CifWriter
from pymatgen.core.structure import Structure
import csv

# Replace "your_file.xyz" with the path to your XYZ file
# structure = Structure.from_file("all_relaxed_rs_10atoms.xyz")

# print(structure)

from ase.io import read, write

# Replace "your_file.xyz" with the path to your XYZ file
# atoms = read("all_relaxed_rs_10atoms.xyz")

# # print(atoms)
# write("output.cif", atoms)

tr=[]
tr_list=[]
header = ['file_name','label']

atoms = read("poly_structs_dft_082224.xyz", index=':')


for i in atoms:
    # atoms = read("all_relaxed_rs_10atoms.xyz", index=i)
    # print(i)
    # print(atoms.get_chemical_formula())
    # tr=[]
    # print(i.get_chemical_formula(mode='reduce'))
    # formula = i.formula

    formula = i.info['formula']
    #print(formula)
    tr.append(formula)
    tr.append(0)
    tr_list.append(tr)

    write('jack_poly_cif_082224/'+formula+'.cif', images=i, format='cif')
