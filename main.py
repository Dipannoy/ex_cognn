"""Module to train for a folder with formatted dataset."""
import csv
import fcntl
import os
import sys
import time
import pandas as pd
import jarvis
from jarvis.core.atoms import Atoms
from data import get_train_val_loaders, get_loaders
#from alignn.train import train_dgl
#from alignn.config import TrainingConfig
#from jarvis.db.jsonutils import loadjson
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import argparse
import pickle
from sklearn.model_selection import KFold

from runner import Runner
import numpy as np

# def get_loaders(
#     config,
#     root_dir="",
#     dataset_name="",
#     suffix=None,
# ):
#     t1 = time.time()
#     with open(os.path.join(root_dir, dataset_name), 'rb') as handle:
#         dataset = pickle.load(handle)

#     datasets = {"train": [], "val": [], "test": []}
#     idx = {data["jid"]: i for i, data in enumerate(dataset)}
#     for z in datasets.keys():
#         if (suffix is not None) and (z != "test"):
#             y = z + "_" + suffix
#         else:
#             y = z
#         ids = pd.read_csv(os.path.join(root_dir, y+"_ids.csv"))
#         ids = ids["jid"].tolist()
#         # print(ids)

#         for id in ids:
#             datasets[z].append(dataset[idx[id]])
    
#     print(f"Number of data, tot: {len(dataset)}, train: {len(datasets['train'])}, val: {len(datasets['val'])}, test: {len(datasets['test'])}")
#     print('-----'*10)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_5_fold_CV(root_dir, target_file, output_dir):
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(output_dir)
    df = pd.read_csv(root_dir+'/'+target_file)
    df[''] = df['id']
    df['jid'] = df['filename'].str.split('.').str[0]
    #df.drop(columns=['id', 'filename','label'], inplace=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X = df

# Split the dataset
    folds = list(kf.split(X))

    for fold, (train_val_idx, test_idx) in enumerate(folds):
    # Split into training/validation and test
        train_val_X, test_X = X.iloc[train_val_idx], X.iloc[test_idx]
        # train_val_y, test_y = y.iloc[train_val_idx], y.iloc[test_idx]

        # Further split training/validation into train and validation sets
        train_size = int(0.95 * len(train_val_X))  # 80% train, 20% validation
        train_X, val_X = train_val_X.iloc[:train_size], train_val_X.iloc[train_size:]
        train_X.to_csv(output_dir+"/train_ids_"+str(fold)+".csv", index=False)
        val_X.to_csv(output_dir+"/val_ids_"+str(fold)+".csv", index=False)
        test_X.to_csv(output_dir+"/test_ids_"+str(fold)+".csv", index=False)
def main(root_dir, alignn_layer, output_dir, cutoff, max_neighbors, edge_threshold, batch_size, exp_name, epochs, neighbor_strategy, target, gcn_layer, target_file, 
        target_file2, action_net_glayer, action_net_lglayer, sub_dir, config_name, dataset_name,repeat,create_split,check_threshold,in_act_graph_layer,action_net,random_weight,logit_1,logit_2,trim_graph_dir,
        sparsity_weight=0.0, global_results_csv='outputs/global_results.csv', **kwargs):
    """Loader preparation"""
    if sub_dir is None:
        config_file = os.path.join(root_dir, config_name)
    else:
        config_file = os.path.join(root_dir, sub_dir, config_name)
    config = OmegaConf.load(config_file)
    config.model.alignn_layers = int(alignn_layer)
    config.model.gcn_layers = int(gcn_layer)
    config.model.edge_threshold = float(edge_threshold)
    config.model.check_threshold = check_threshold
    config.model.action_args.num_act_graph_layers = int(action_net_glayer)
    config.model.action_args.num_act_linegraph_layers = int(action_net_lglayer)
    config.model.action_args.num_in_act_graph_layers = int(in_act_graph_layer)
    config.model.random_weight = int(random_weight)
    config.model.logit_1 = float(logit_1)
    config.model.logit_2 = float(logit_2)
    config.model.action_args.action_net = action_net
    config.target = target
    config.batch_size = int(batch_size)
    config.epochs = int(epochs)
    config.optim.scheduler.epochs = int(epochs)
    config.exp_name = exp_name
    config.output_dir = os.path.join('outputs', output_dir)
    os.makedirs(config.output_dir, exist_ok=True)
    config.sparsity_weight = float(sparsity_weight)
    config.cutoff = int(cutoff)  
    config.max_neighbors = int(max_neighbors)
    config.neighbor_strategy = neighbor_strategy

    print('-----'*10)
    print(config)
    print('-----'*10)
    train_tot = 0
    val_tot = 0
    test_tot = 0
    tr_arr = []
    val_arr = []
    test_arr = []
    create_5_fold_CV(root_dir,target_file,config.output_dir)

    for l in range(0,repeat):


        
        loaders = get_loaders(config=config, 
                            root_dir=root_dir, 
                            sub_dir=sub_dir, 
                            dataset_name=dataset_name,
                            target_file=target_file,
                            fold=l,
                            trim_graph_dir=trim_graph_dir)
        '''
        if create_split==True:
            print('Split On')
            loaders = get_loaders(config=config, 
                                root_dir=root_dir, 
                                sub_dir=sub_dir, 
                                dataset_name=dataset_name,
                                target_file=target_file,
                                target_file2=target_file2)
        else:
            print('Split Off')
            loaders = get_loaders_nosplit(config=config, 
                    root_dir=root_dir, 
                    sub_dir=sub_dir, 
                    dataset_name=dataset_name,
                    target_file=target_file)
        '''
        train_loader, val_loader, _, _ = loaders
        runner = Runner(config=config, **kwargs)
        
        tic = time.time()
        print("Start training")
        runner.train(train_loader, val_loader,l)
        print(f"Finish training: {(time.time() - tic):.5f} s")

        tic = time.time()
        print("Test predictions")
        test_list = ['train', 'val', 'test']
        report_data = runner.test(loaders[:-1], test_list, l,root_dir)
        train_tot = train_tot+report_data['train']['mae_metric']
        val_tot = val_tot+report_data['val']['mae_metric']
        test_tot = test_tot+report_data['test']['mae_metric']
        tr_arr.append(report_data['train']['mae_metric'].item())
        val_arr.append(report_data['val']['mae_metric'].item())
        test_arr.append(report_data['test']['mae_metric'].item())

        for k in test_list:
            print(f"{k.upper()} metrics, mae: {report_data[k]['mae_metric']}, mse: {report_data[k]['mse_metric']}")
            print("------"*10)
        print(f"Finish predictions: {(time.time() - tic):.5f} s")

    print('train_avg_mae: ', np.mean(tr_arr))
    print('val_avg_mae: ', np.mean(val_arr))
    print('test_avg_mae: ',np.mean(test_arr))

    print('train_std_mae: ', np.std(tr_arr))
    print('val_std_mae: ', np.std(val_arr))
    print('test_std_mae: ',np.std(test_arr))

    tr_list = []
    tr_list.append(['train',np.mean(tr_arr),np.std(tr_arr)])
    tr_list.append(['validation',np.mean(val_arr),np.std(val_arr)])
    tr_list.append(['test',np.mean(test_arr),np.std(test_arr)])
    df = pd.DataFrame(tr_list, columns=['sample', 'mean', 'std']) # Add column names if needed
    df.to_csv(config.output_dir+'/error.csv', index=False)

    # Append a single row to the global results CSV (file-locked for concurrent jobs)
    dataset_tag = os.path.basename(root_dir.rstrip('/'))
    global_row = [
        dataset_tag,
        action_net,
        float(edge_threshold),
        float(sparsity_weight),
        float(np.mean(tr_arr)),  float(np.std(tr_arr)),
        float(np.mean(val_arr)), float(np.std(val_arr)),
        float(np.mean(test_arr)),float(np.std(test_arr)),
        output_dir,
    ]
    global_header = [
        'dataset', 'action_net', 'threshold', 'sparsity_weight',
        'train_mean', 'train_std',
        'val_mean',   'val_std',
        'test_mean',  'test_std',
        'output_dir',
    ]
    os.makedirs(os.path.dirname(global_results_csv) or '.', exist_ok=True)
    with open(global_results_csv, 'a', newline='') as gf:
        fcntl.flock(gf, fcntl.LOCK_EX)
        write_header = gf.tell() == 0
        writer = csv.writer(gf)
        if write_header:
            writer.writerow(global_header)
        writer.writerow(global_row)
        fcntl.flock(gf, fcntl.LOCK_UN)
    print(f'Global results appended to: {global_results_csv}')


    

    '''
    loaders = get_loaders(config=config, 
                          root_dir=root_dir, 
                          sub_dir=sub_dir, 
                          dataset_name=dataset_name)
    train_loader, val_loader, _, _ = loaders
    runner = Runner(config=config, **kwargs)
    
    tic = time.time()
    print("Start training")
    runner.train(train_loader, val_loader)
    print(f"Finish training: {(time.time() - tic):.5f} s")

    tic = time.time()
    print("Test predictions")
    test_list = ['train', 'val', 'test']
    report_data = runner.test(loaders[:-1], test_list, root_dir)
    for k in test_list:
        print(f"{k.upper()} metrics, mae: {report_data[k]['mae_metric']}, mse: {report_data[k]['mse_metric']}")
        print("------"*10)
    print(f"Finish predictions: {(time.time() - tic):.5f} s")
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network"
    )
    
    parser.add_argument(
        "--root_dir",
        default="./datasets/oxides",
        help="Folder with the dataset file. Folder also with id_props.csv and the config file if sub_dir is None.",
    )
    parser.add_argument(
        "--target_file",
        default="./datasets/oxides",
        help="Folder with the dataset file. Folder also with id_props.csv and the config file if sub_dir is None.",
    )
    parser.add_argument(
        "--target_file2",
        default="./datasets/oxides",
        help="Folder with the dataset file. Folder also with id_props.csv and the config file if sub_dir is None.",
    )



    parser.add_argument(
        "--sub_dir",
        default=None,
        help="Sub-folder with id_props.csv and the config file.",
    )
    
    parser.add_argument(
        "--config_name",
        default="config_example.json",
        help="Name of the config file.",
    )
    
    parser.add_argument(
        "--dataset_name",
        default="oxides_dft.pickle",
        help="Name of the dataset file.",
    )
    
    parser.add_argument(
        "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
    )
    
    parser.add_argument(
        "--keep_data_order",
        default=False,
        help="Whether to randomly shuffle samples, True/False",
    )
    
    parser.add_argument(
        "--classification_threshold",
        default=None,
        help="Floating point threshold for converting into 0/1 class"
        + ", use only for classification tasks",
    )
    
    parser.add_argument(
        "--batch_size", default=None, help="Batch size, generally 64"
    )

    parser.add_argument(
        "--cutoff", default=8, help="Batch size, generally 64"
    )
    parser.add_argument(
        "--max_neighbors", default=12, help="max neighbors, generally 12"
    )
    parser.add_argument(
        "--epochs", default=None, help="Number of epochs, generally 300"
    )
    parser.add_argument(
        "--edge_threshold", default=0.5, help="Number of epochs, generally 300"
    )

    parser.add_argument(
        "--target", default="Ef", help="Folder to save outputs",
    )

    
    parser.add_argument(
        "--output_dir", default="./", help="Folder to save outputs",
    )
    parser.add_argument(
        "--output_dir2", default="./", help="Folder to save outputs",
    )
    parser.add_argument(
        "--ang_layer", default=None,  help="action_net_glayer",
    )
    parser.add_argument(
        "--anlg_layer", default=None,  help="action_net_lglayer",
    )
    parser.add_argument(
        "--neighbor_strategy", default='k-nearest', help='neighbor strategy',

    )

    parser.add_argument(
        "--aln_layer", default=None,  help="alignn_layer",
    )
    parser.add_argument(
        "--gn_layer", default=None,  help="gcn_layer",
    )
    parser.add_argument(
        "--repeat", default=5, type=int,  help="repeat",
    )
    parser.add_argument(
        "--create_split", default='true', type=str2bool,  help="create_split",
    )

    parser.add_argument(
        "--check_threshold", default='true', type=str2bool, help="check_threshold",
    )
    parser.add_argument(
        "--in_act_graph_layer", default=1, type=int, help="in_act_graph_layer",
    )
    parser.add_argument(
        "--action_net", default='ActionNet', help="action network",
    )

    parser.add_argument(
        "--random_weight", default=0,  help="random_weight",
    )
    parser.add_argument(
        "--logit_1", default=0.5,  help="first logit",
    )
    parser.add_argument(
        "--logit_2", default=0.5,  help="second logit",
    )
    parser.add_argument(
        "--trim_graph_dir", default=None,  help="broadcasting graphs",
    )
    parser.add_argument(
        "--sparsity_weight", default=0.0, type=float,
        help="Sparsity regularization weight: penalizes mean edge weight to encourage action net to drop non-informative bonds.",
    )
    parser.add_argument(
        "--global_results_csv", default='outputs/global_results.csv',
        help="Path to the shared CSV file where all experiments append their mean/std test error.",
    )
      
    
    args = parser.parse_args(sys.argv[1:])
    
    main(
        root_dir=args.root_dir,
        sub_dir=args.sub_dir,
        dataset_name=args.dataset_name,
        target_file = args.target_file,
        target_file2 = args.target_file2,
        alignn_layer = args.aln_layer,
        gcn_layer = args.gn_layer,
        cutoff = args.cutoff,
        max_neighbors = args.max_neighbors,
        target = args.target,
        edge_threshold = args.edge_threshold,
        action_net_glayer = args.ang_layer,
        action_net_lglayer = args.anlg_layer,
        #file_name=args.file_name,
        config_name=args.config_name,
        keep_data_order=args.keep_data_order,
        classification_threshold=args.classification_threshold,
        batch_size=(args.batch_size),
        epochs=(args.epochs),
        neighbor_strategy = args.neighbor_strategy,
        file_format=(args.file_format),
        exp_name=args.output_dir2,
        output_dir=args.output_dir,
        repeat = args.repeat,
        create_split = args.create_split,
        check_threshold = args.check_threshold,
        in_act_graph_layer = args.in_act_graph_layer,
        action_net = args.action_net,
        random_weight = args.random_weight,
        logit_1 = args.logit_1,
        logit_2 = args.logit_2,
        trim_graph_dir = args.trim_graph_dir,
        sparsity_weight = args.sparsity_weight,
        global_results_csv = args.global_results_csv,
    )
