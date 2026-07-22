import csv
import fcntl
import glob
import json
import os
import sys
import time
import pandas as pd
import jarvis
from jarvis.core.atoms import Atoms
# from data import get_train_val_loaders, get_loaders
# #from alignn.train import train_dgl
# #from alignn.config import TrainingConfig
# #from jarvis.db.jsonutils import loadjson
# from omegaconf import OmegaConf
# from omegaconf.dictconfig import DictConfig
# import argparse
# import pickle
from sklearn.model_selection import KFold

# from runner import Runner
# import numpy as np
# import torch


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


create_5_fold_CV(root_dir='./mp_full_gap', target_file='mp_gap_full.csv', output_dir='./mp_full_gap')