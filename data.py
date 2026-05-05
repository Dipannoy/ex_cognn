"""Jarvis-dgl data loaders and DGLGraph utilities."""

import random
from pathlib import Path
from typing import Optional
import time

# from typing import Dict, List, Optional, Set, Tuple

import os
import torch
import dgl
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from jarvis.core.atoms import Atoms
from graphs import Graph, StructureDataset

# from jarvis.core.graphs import Graph, StructureDataset
from jarvis.db.figshare import data as jdata
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson

# from sklearn.pipeline import Pipeline
import pickle as pk

# from sklearn.decomposition import PCA  # ,KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv
# use pandas progress_apply
tqdm.pandas()


def load_dataset(
    name: str = "dft_3d",
    target=None,
    limit: Optional[int] = None,
    classification_threshold: Optional[float] = None,
):
    """Load jarvis data."""
    d = jdata(name)
    data = []
    for i in d:
        if i[target] != "na" and not math.isnan(i[target]):
            if classification_threshold is not None:
                if i[target] <= classification_threshold:
                    i[target] = 0
                elif i[target] > classification_threshold:
                    i[target] = 1
                else:
                    raise ValueError(
                        "Check classification data type.",
                        i[target],
                        type(i[target]),
                    )
            data.append(i)
    d = data
    if limit is not None:
        d = d[:limit]
    # d = pd.DataFrame(d)
    # d = d.replace("na", np.nan)
    return d


# np.mean(mean_absolute_deviation(x,axis=0))
def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def load_graphs(
    df: pd.DataFrame,
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    cachedir: Optional[Path] = None,
    use_canonize: bool = False,
    extra_feat_names = [],
    trim_graph_dir = None,
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """

    def atoms_to_graph(atoms, jid, **kwargs):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        return Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
            # extra_atom_features=extra_atom_features,
            # global_features=global_features,
            # extra_features=extra_features,
            jid=jid,
            trim_graph_dir=trim_graph_dir,
            extra_feat_kwargs=kwargs,
        )

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{neighbor_strategy}.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        graphs, labels = dgl.load_graphs(str(cachefile))
    else:
        # for idx, data in tqdm(df.iterrows()):
        #     atoms = data["atoms"]
        #     structure = (
        #         Atoms.from_dict(atoms) if isinstance(atoms, dict) else atoms
        #     )
        #     g = Graph.atom_dgl_multigraph(
        #         structure,
        #         cutoff=cutoff,
        #         cutoff_extra=cutoff_extra,
        #         atom_features="atomic_number",
        #         max_neighbors=max_neighbors,
        #         compute_line_graph=False,
        #         use_canonize=use_canonize,
        #         neighbor_strategy=neighbor_strategy,
        #         id=data[id_tag],
        #     )
            
        #     if "extra_features" in data:
        #         natoms = len(atoms["elements"])
        #         g.ndata["extra_features"] = torch.tensor(
        #             [data["extra_features"] for n in range(natoms)]
        #         ).type(torch.get_default_dtype())
        #     graphs.append(g)
        pass

        # extras = ["extra_atom_features", "global_features", "extra_features"]
        extra_feat_names = [extra for extra in extra_feat_names if extra in df.keys()]
        if len(extra_feat_names) > 0:
            graphs = df.progress_apply(lambda row: atoms_to_graph(
                                                        row["atoms"], row["jid"],
                                                        **{extra: row[extra] for extra in extra_feat_names}
                                                        ), axis=1).tolist()
        else:
            graphs = df.progress_apply(lambda row: atoms_to_graph(
                                                        row["atoms"], row["jid"],
                                                        ), axis=1).tolist()
        
        if cachefile is not None:
            dgl.save_graphs(str(cachefile), graphs.tolist())

    return graphs


def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    # np.random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    # shuffle consistently with https://github.com/txie-93/cgcnn/data.py
    # i.e. shuffle the index in place with standard library random.shuffle
    # first obtain only valid indices

    # test_size = round(N * 0.2)

    # full train/val test split
    # ids = ids[::-1]
    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


def get_torch_dataset(
    dataset,
    id_tag="jid",
    target="",
    target_atomwise="",
    target_grad="",
    target_stress="",
    neighbor_strategy="",
    atom_features="",
    use_canonize="",
    name="",
    line_graph="",
    cutoff=8.0,
    max_neighbors=12,
    classification=False,
    output_dir=".",
    tmp_name="dataset",
    extra_feat_names=[],
    trim_graph_dir = None,
):
    """Get Torch Dataset."""
    df = pd.DataFrame(dataset)
    try:
        feats = [k for k in df.columns if "global_features" in k]
        for feat in feats:
            df["extra_features."+feat.replace("global_features.", "")] = df[feat]
    except Exception as e:
        print(f"DF ERROR: {e}")
    # df['natoms']=df['atoms'].apply(lambda x: len(x['elements']))
    vals = np.array([data[target] for data in dataset])  # df[target].values
    print("data range", np.max(vals), np.min(vals))
    f = open(os.path.join(output_dir, tmp_name + "_data_range"), "w")
    line = "Max=" + str(np.max(vals)) + "\n"
    f.write(line)
    line = "Min=" + str(np.min(vals)) + "\n"
    f.write(line)
    f.close()

    graphs = load_graphs(
        df,
        name=name,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        extra_feat_names=extra_feat_names,
        trim_graph_dir=trim_graph_dir,
    )
    data = StructureDataset(
        df,
        graphs,
        target=target,
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        atom_features=atom_features,
        line_graph=line_graph,
        id_tag=id_tag,
        classification=classification,
    )
    return data


def get_train_val_loaders(
    datasets,
    target: str = "formation_energy_peratom",
    target_atomwise: str = "",
    target_grad: str = "",
    target_stress: str = "",
    atom_features: str = "cgcnn",
    neighbor_strategy: str = "k-nearest",
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size: int = 5,
    standardize: bool = False,
    line_graph: bool = True,
    split_seed: int = 123,
    workers: int = 0,
    pin_memory: bool = True,
    save_dataloader: bool = False,
    filename: str = "sample",
    id_tag: str = "jid",
    use_canonize: bool = False,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    classification_threshold: Optional[float] = None,
    target_multiplication_factor: Optional[float] = None,
    standard_scalar_and_pca=False,
    keep_data_order=False,
    output_features=1,
    output_dir=None,
    extra_feat_names=[],
    trim_graph_dir=None,
):

    print(output_dir)
    if not os.path.exists(output_dir):
        print(output_dir)
        os.makedirs(output_dir)
        print('directory created')

    if standard_scalar_and_pca:
        y_data = [i[target] for i in dataset_train]
        # pipe = Pipeline([('scale', StandardScaler())])
        if not isinstance(y_data[0], list):
            print("Running StandardScalar")
            y_data = np.array(y_data).reshape(-1, 1)
        sc = StandardScaler()

        sc.fit(y_data)
        print("Mean", sc.mean_)
        print("Variance", sc.var_)
        try:
            print("New max", max(y_data))
            print("New min", min(y_data))
        except Exception as exp:
            print(exp)
            pass
        # pc = PCA(n_components=output_features)
        # pipe = Pipeline(
        #    [
        #        ("scale", StandardScaler()),
        #        ("reduce_dims", PCA(n_components=output_features)),
        #    ]
        # )
        pk.dump(sc, open(os.path.join(output_dir, "sc.pkl"), "wb"))
        # pc = PCA(n_components=10)
        # pc.fit(y_data)
        # pk.dump(pc, open("pca.pkl", "wb"))

    train_data = get_torch_dataset(
        dataset=datasets['train'],
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name="oxides",
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="train_data",
        extra_feat_names=extra_feat_names,
        trim_graph_dir = trim_graph_dir,
    )
    val_data = get_torch_dataset(
        dataset=datasets['val'],
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name="oxides",
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="val_data",
        extra_feat_names=extra_feat_names,
        trim_graph_dir = trim_graph_dir,
    )
    test_data = get_torch_dataset(
        dataset=datasets['test'],
        id_tag=id_tag,
        atom_features=atom_features,
        target=target,
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        name="oxides",
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        classification=classification_threshold is not None,
        output_dir=output_dir,
        tmp_name="test_data",
        extra_feat_names=extra_feat_names,
        trim_graph_dir = trim_graph_dir,
    )

    collate_fn = train_data.collate
    if line_graph:
        collate_fn = train_data.collate_line_graph

    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    
    if save_dataloader:
        torch.save(train_loader, train_sample)
        torch.save(val_loader, val_sample)
        torch.save(test_loader, test_sample)
    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))

    if "more_test" in datasets.keys():
        more_test_data = get_torch_dataset(
            dataset=datasets['more_test'],
            id_tag=id_tag,
            atom_features=atom_features,
            target=target,
            target_atomwise=target_atomwise,
            target_grad=target_grad,
            target_stress=target_stress,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
            name="oxides",
            line_graph=line_graph,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            classification=classification_threshold is not None,
            output_dir=output_dir,
            tmp_name="more_test_data",
            extra_feat_names=extra_feat_names,
        )
    
        more_test_loader = DataLoader(
            more_test_data,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=workers,
            pin_memory=pin_memory,
        )      
    
        print("n_more_test:", len(more_test_loader.dataset))
        return (
            train_loader,
            val_loader,
            test_loader,
            more_test_loader,
        )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_loader.dataset.prepare_batch,
    )


def get_extra_feat_names(config):
    
    def name_list(feat, feat_type):
        names = []
        if feat is not None:
            if isinstance(feat, int):
                if feat > 0:
                    names = [feat_type]
            elif isinstance(feat, DictConfig):
                names = [feat_type+"."+str(k) for k in list(feat.keys())]
        return names

    extra_atom_features = name_list(config.extra_atom_features, "extra_atom_features")
    global_features = name_list(config.global_features, "global_features")
    extra_features = name_list(config.extra_features, "extra_features")
    extra_feat_names = extra_atom_features + global_features + extra_features
    
    return extra_feat_names


'''
def create_split_mp_poly(root_dir, target_file):
    df = pd.read_csv(root_dir+'/'+target_file)
    df2 = pd.read_csv(root_dir+'/'+'jack_poly_eg_082824.csv')

    tr_list = []
    train, test = train_test_split(df, test_size=0.1)
    # test_2, val = train_test_split(test,test_size=0.5)
    header = ['','jid']
    for i in df2.index:
        tr=[]
        tr.append(df2['id'][i])
        tr.append(df2['filename'][i].split('.')[0])
        #tr.append(val['filename'][i])

        
        tr_list.append(tr)


    with open(root_dir+'/'+'test_ids.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(header) 

        write.writerows(tr_list) 


    tr_list = []
    #header = ['id','compound']
    for i in train.index:
        tr=[]
        tr.append(train['id'][i])
        tr.append(train['filename'][i].split('.')[0])
        #tr.append(train['filename'][i])

        
        tr_list.append(tr)


    with open(root_dir+'/'+'train_ids.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(header) 

        write.writerows(tr_list) 


    tr_list = []

    for i in test.index:
        tr=[]
        tr.append(test['id'][i])
        tr.append(test['filename'][i].split('.')[0])
        #tr.append(test_2['filename'][i])

        
        tr_list.append(tr)

#header = ['','jid']
    with open(root_dir+'/'+'val_ids.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(header) 

        write.writerows(tr_list)


def create_split_mp_poly(root_dir, target_file,target_file2):
    df = pd.read_csv(root_dir+'/'+target_file)
    df2 = pd.read_csv(root_dir+'/'+target_file2)

    tr_list = []
    train, test = train_test_split(df, test_size=0.1)

    header = ['','jid']
    for i in df2.index:
        tr=[]
        tr.append(df2['id'][i])
        tr.append(df2['filename'][i].split('.')[0])
	tr_list.append(tr)


    with open(root_dir+'/'+'test_ids.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(header) 
	write.writerows(tr_list) 


    tr_list = []

    for i in train.index:
        tr=[]
	tr.append(train['id'][i])
        tr.append(train['filename'][i].split('.')[0])
        tr_list.append(tr)
        
    with open(root_dir+'/'+'train_ids.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(header) 

        write.writerows(tr_list) 


    tr_list = []
    for i in test.index:
        tr=[]
	tr.append(test['id'][i])
        tr.append(test['filename'][i].split('.')[0])
	tr_list.append(tr)


    with open(root_dir+'/'+'val_ids.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(header) 

        write.writerows(tr_list)     
'''



def create_split_mp_poly(root_dir, target_file, target_file2):
    df = pd.read_csv(root_dir+'/'+target_file)
    df2 = pd.read_csv(root_dir+'/'+target_file2)

    tr_list = []
    train, test = train_test_split(df, test_size=0.1)
    # test_2, val = train_test_split(test,test_size=0.5)
    header = ['','jid']
    for i in df2.index:
        tr=[]
        tr.append(df2['id'][i])
        tr.append(df2['filename'][i].split('.')[0])
        #tr.append(val['filename'][i])

        
        tr_list.append(tr)


    with open(root_dir+'/'+'test_ids.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(header) 

        write.writerows(tr_list) 


    tr_list = []
    #header = ['id','compound']
    for i in train.index:
        tr=[]
        tr.append(train['id'][i])
        tr.append(train['filename'][i].split('.')[0])
        #tr.append(train['filename'][i])

        
        tr_list.append(tr)


    with open(root_dir+'/'+'train_ids.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(header) 

        write.writerows(tr_list) 


    tr_list = []

    for i in test.index:
        tr=[]
        tr.append(test['id'][i])
        tr.append(test['filename'][i].split('.')[0])
        #tr.append(test_2['filename'][i])

        
        tr_list.append(tr)


    with open(root_dir+'/'+'val_ids.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(header) 

        write.writerows(tr_list) 
def create_split(root_dir, target_file):
    df = pd.read_csv(root_dir+'/'+target_file)
    tr_list = []
    train, test = train_test_split(df, test_size=0.1)
    test_2, val = train_test_split(test,test_size=0.5)
    header = ['','jid']
    for i in val.index:
        tr=[]
        tr.append(val['id'][i])
        tr.append(val['filename'][i].split('.')[0])
        #tr.append(val['filename'][i])

        
        tr_list.append(tr)


    with open(root_dir+'/'+'val_ids.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(header) 

        write.writerows(tr_list) 


    tr_list = []
    #header = ['id','compound']
    for i in train.index:
        tr=[]
        tr.append(train['id'][i])
        tr.append(train['filename'][i].split('.')[0])
        #tr.append(train['filename'][i])

        
        tr_list.append(tr)


    with open(root_dir+'/'+'train_ids.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(header) 

        write.writerows(tr_list) 


    tr_list = []

    for i in test_2.index:
        tr=[]
        tr.append(test_2['id'][i])
        tr.append(test_2['filename'][i].split('.')[0])
        #tr.append(test_2['filename'][i])

        
        tr_list.append(tr)

#header = ['','jid']
    with open(root_dir+'/'+'test_ids.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerow(header) 

        write.writerows(tr_list) 


    
def get_loaders(
    config,
    root_dir="",
    sub_dir=None,
    dataset_name="",
    target_file="",
    target_file2="",
    suffix=None,
    fold=5,
    trim_graph_dir = None,
):
    t1 = time.time()
    with open(os.path.join(root_dir, dataset_name), 'rb') as handle:
        dataset = pk.load(handle)
    create_split(root_dir,target_file)
    #create_split_mp_poly(root_dir,target_file,target_file2)

    datasets = {"train": [], "val": [], "test": []}
    idx = {data["jid"]: i for i, data in enumerate(dataset)}
    for z in datasets.keys():
        if (suffix is not None) and (z != "test"):
            y = z + "_" + suffix
        else:
            y = z
        
        if sub_dir is None:
            ids = pd.read_csv(os.path.join(config.output_dir, y+"_ids_"+str(fold)+".csv"))
            #ids = pd.read_csv(os.path.join(root_dir, y+"_ids"+".csv"))
        else:
            ids = pd.read_csv(os.path.join(root_dir, sub_dir, y+"_ids_"+str(fold)+".csv"))
        ids = ids["jid"].tolist()

        for id in ids:
            datasets[z].append(dataset[idx[id]])
    
    print(f"Number of data, tot: {len(dataset)}, train: {len(datasets['train'])}, val: {len(datasets['val'])}, test: {len(datasets['test'])}")
    print('-----'*10)

    extra_feat_names = get_extra_feat_names(config.model)
    print(f"START: {extra_feat_names}")

    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        datasets=datasets,
        target=config.target,
        # n_train=config.n_train,
        # n_val=config.n_val,
        # n_test=config.n_test,
        # train_ratio=config.train_ratio,
        # val_ratio=config.val_ratio,
        # test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=config.keep_data_order,
        output_dir=config.output_dir,
        extra_feat_names=extra_feat_names,
        trim_graph_dir = trim_graph_dir,
    )

    t2 = time.time()
    print(f"Loaders prep: {(t2 - t1):.5f} s")    
    return train_loader, val_loader, test_loader, prepare_batch
