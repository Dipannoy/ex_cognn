"""Module to train for a folder with formatted dataset."""
import csv
import os
import sys
import time
import pandas as pd
from jarvis.core.atoms import Atoms
from data import get_train_val_loaders, get_loaders
#from alignn.train import train_dgl
#from alignn.config import TrainingConfig
#from jarvis.db.jsonutils import loadjson
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import argparse
import pickle
import numpy as np

from runner import Runner


def main(root_dir, sub_dir, config_name, dataset_name, n_folds, **kwargs):
    """Loader preparation"""
    t0 = time.time()
    if sub_dir is None:
        config_file = os.path.join(root_dir, config_name)
    else:
        config_file = os.path.join(root_dir, sub_dir, config_name)
    config = OmegaConf.load(config_file)

    print('-----'*10)
    print(config)
    print('-----'*10)
    
    test_list = ['train', 'val', 'test']
    avg_metrics = {k: {m: np.zeros((n_folds,)) for m in ['mae_metric', 'mse_metric']} for k in test_list} 

    for n in range(n_folds):
        print(f"{n+1} of {n_folds} FOLDS begins")
        loaders = get_loaders(config=config, root_dir=root_dir, 
                              dataset_name=dataset_name, suffix=n)
        train_loader, val_loader, _, _ = loaders
        runner = Runner(config=config, **kwargs)
        
        tic = time.time()
        print(f"Start {n+1} of {n_folds} folds training")
        #runner.train(train_loader, val_loader, suffix=n)
        print(f"Finish {n+1} of {n_folds} folds training: {(time.time() - tic):.5f} s")
    
        tic = time.time()
        print(f"Test {n+1} of {n_folds} folds predictions")
        report_data = runner.test(loaders[:-1], test_list, config.output_dir, suffix=n)
        for k in test_list:
            print(f"{k.upper()} {n+1} fold metrics, mae: {report_data[k]['mae_metric']}, mse: {report_data[k]['mse_metric']}")
            print("------"*10)
            avg_metrics[k]['mae_metric'][n] = report_data[k]['mae_metric']
            avg_metrics[k]['mse_metric'][n] = report_data[k]['mse_metric']
        print(f"Finish {n+1} of {n_folds} folds predictions: {(time.time() - tic):.5f} s")

    for k in test_list:
        print(f"{k.upper()} avg metrics, mae: {avg_metrics[k]['mae_metric'].mean()}, mse: {avg_metrics[k]['mse_metric'].mean()}")
    print("------"*10)
    print(f"DONE: {(time.time() - t0):.5f} s")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network"
    )
    
    parser.add_argument(
        "--root_dir",
        default="./datasets/oxides",
        help="Folder with id_props.csv, structure files",
    )

    parser.add_argument(
        "--sub_dir",
        default=None,
        help="Sub-folder with id_props.csv and the config file.",
    )
    
    parser.add_argument(
        "--config_name",
        default="config_example.json",
        help="Name of the config file",
    )
    
    parser.add_argument(
        "--dataset_name",
        default="oxides_dft.pickle",
        help="Name of the dataset file",
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
        "--epochs", default=None, help="Number of epochs, generally 300"
    )
    
    parser.add_argument(
        "--output_dir", default="./", help="Folder to save outputs",
    )

    parser.add_argument(
        "--n_folds", default=1, type=int, help="Number of folds",
    )
    
    args = parser.parse_args(sys.argv[1:])
    
    main(
        root_dir=args.root_dir,
        sub_dir=args.sub_dir,
        dataset_name=args.dataset_name,
        #file_name=args.file_name,
        config_name=args.config_name,
        keep_data_order=args.keep_data_order,
        classification_threshold=args.classification_threshold,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        file_format=args.file_format,
        n_folds=args.n_folds,
    )
