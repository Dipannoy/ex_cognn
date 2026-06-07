import numpy as np
import time
import copy
import json
import os
import glob
from functools import partial
import pickle
import dgl
import torch
from torch import nn
import matplotlib as mpl
import matplotlib.pyplot as plt

from models.alignn import CoALIGNN
from collections import defaultdict

#from models.alignn2 import CoALIGNN
# from models.attn_alignn import AttnALIGNN
# from models.attn_alignn_gpslap import AttnALIGNNGPSLap

class Runner:
    def __init__(self, config, verbose=True, **kwargs):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        if (config.model.name == "alignn") or (config.model.name == "coalignn"):
            self.model = CoALIGNN(config.model)
        # elif config.model.name == "attnalignn":
        #     self.model = AttnALIGNN(config.model)
        # elif config.model.name == "attnalignngpslap":
        #     self.model = AttnALIGNNGPSLap(config.model)
        self.model = self.model.to(self.device)
        
        self.optimizer = self.setup_optimizer(self.model.parameters(), config.optim.optimizer)
        self.scheduler = self.setup_scheduler(config.optim.scheduler)

        self.loss_fn = self.criterion(config.criterion)

        self.max_bond_forcing = config.model.max_bond_forcing
        self.sparsity_weight = getattr(config, 'sparsity_weight', 0.0)
        self.num_epochs = config.epochs
        self.step_count = 0
        self.gradient_means = defaultdict(list)
        self.learning_rates = []

        if verbose:
            print(self.model)
    
    def _load_checkpoint(self, mode='train', suffix=None):
        suffix = suffix if suffix is not None else self.config.suffix
        ckpts = list(glob.glob(f'{self.config.output_dir}/*={suffix}.ckpt'))  
        if len(ckpts) > 0:
            ckpt_epochs = [ckpt.split('/')[-1].split('.')[0].split('=')[1] for ckpt in ckpts]
            ckpt_epochs = np.array([int(epoch) if epoch != 'best' else -1 for epoch in ckpt_epochs])
            if mode == 'train':
                ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
            elif mode == 'predict':
                ckpt = str(ckpts[ckpt_epochs.argsort()[0]])
            
            print(f">>>>> Load model from checkpoint:\n{ckpt}\n")
            if torch.cuda.is_available():
                ckpt = torch.load(ckpt)
            else:
                ckpt = torch.load(ckpt, map_location=torch.device('cpu'))
            
            print(f">>>>> Update model from checkpoint:") 
            self.model.load_state_dict(ckpt['model_state_dict'])

            if mode == 'train':
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except:
                    print("ERROR: Loading optimizer")
                try:
                    self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except:
                    print("ERROR: Loading scheduler")                
                   
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt['best_val_loss']
            best_val_epoch = ckpt['best_val_epoch']
            ckpt_path = f"{self.config.output_dir}/epoch={ckpt['epoch']}={suffix}.ckpt"
            best_ckpt_path = f"{self.config.output_dir}/epoch=best={suffix}.ckpt"
            print(f"revived epoch: {ckpt['epoch']}")
            print(f"best_val_loss: {best_val_loss}")
            print(f"best_val_epoch: {best_val_epoch}\n")
            if mode == 'train':
                print(">>>>> CONTINUE TRAINING")
            else:
                print(f">>>>> {mode.upper()}")
            
        else:
            print(f">>>>> NEW TRAINING")
            start_epoch = 0
            best_val_loss, best_val_epoch = 1e12, -1
            ckpt_path, best_ckpt_path = None, None               
        return start_epoch, (best_val_loss, best_val_epoch), (ckpt_path, best_ckpt_path)

    def _save_checkpoint(self, model_ckpt, ckpt_path, epoch, suffix=None):
        suffix = suffix if suffix is not None else self.config.suffix
        new_ckpt_path = f"{self.config.output_dir}/epoch={epoch}={suffix}.ckpt"
        if ckpt_path and os.path.exists(ckpt_path):
            os.remove(ckpt_path)        
        torch.save(model_ckpt, new_ckpt_path)
        #print(f"saved checkpoint at epoch [{epoch}]: ", new_ckpt_path)
        return new_ckpt_path

    def _get_checkpoint_dict(self, epoch, best_val_loss, best_val_epoch):
        model_ckpt = {
                'model_state_dict': copy.deepcopy(self.model.state_dict()), 
                'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()), 
                'scheduler_state_dict': copy.deepcopy(self.scheduler.state_dict()),
                'epoch': epoch, 
                'best_val_loss': best_val_loss,
                'best_val_epoch': best_val_epoch,
                #'train_reg_loss': log_dict['train_reg_loss'], 
                #'val_reg_loss': log_dict['val_reg_loss'] 
            }
        return model_ckpt

    def _logging(self, epoch, train_loss, val_loss, train_survived_edges, val_survived_edges):
        log_dict = {'epoch': epoch, 
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_survived_edges': {k: v if isinstance(v, list) else v.tolist() for k, v in train_survived_edges.items()},
                    'val_survived_edges': {k: v if isinstance(v, list) else v.tolist() for k, v in val_survived_edges.items()},
                   }  
        
        with open(self.config.output_dir + "/training_metrics.json", 'a') as f:
            results = {k: v for k, v in log_dict.items()}
            json.dump(results, f)
            f.write('\n')

    def _early_stopping(self, epoch, best_val_epoch):
        if epoch - best_val_epoch > self.config.early_stopping_patience:
            return True
        else:
            return False

    def criterion(self, task_loss):
        if task_loss == 'mae':
            loss_fn = nn.L1Loss()
        elif task_loss == 'mse':
            loss_fn = nn.MSELoss()
        elif task_loss == 'bce':
            loss_fn = nn.BCELoss()
        return loss_fn

    def setup_optimizer(self, params, config):
        """Set up optimizer for param groups."""
        if config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay,
            )
        return optimizer

    def setup_scheduler(self, config):
        if config.scheduler == "none":
            # always return multiplier of 1 (i.e. do nothing)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda epoch: 1.0
            )
        elif config.scheduler == "onecycle":
            # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
            scheduler = partial(torch.optim.lr_scheduler.OneCycleLR, 
                optimizer=self.optimizer,
                max_lr=config.learning_rate,
                epochs=config.epochs,
                #total_steps=117000,
                #steps_per_epoch=steps_per_epoch,
                # pct_start=pct_start,
                pct_start=0.3,
            )
        elif config.scheduler == "step":
            # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size = 50, gamma=0.1
            )
	
            

        return scheduler

    def draw_plot(self, train_err, valid_err, suffix):
        al = self.config.model.alignn_layers
        gl = self.config.model.gcn_layers
        target = self.config.target
        exp_name = self.config.exp_name
        actgl = self.config.model.action_args.num_act_graph_layers
        actlgl = self.config.model.action_args.num_act_linegraph_layers
        plot_name = './'+self.config.output_dir+'/'+ exp_name+'_'+target+'_'+str(al) + str(gl)+str(actgl) + str(actlgl) +'_'+str(suffix)+ '.png'
 
        plt.plot(train_err, label='train error')
        plt.plot(valid_err, label='validation error')

        plt.title('Plot History: Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train error', 'validation error'], loc='upper right')
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close()
        
    def plot_logit_distributions(self,in_logit, out_logit,key,count):
        plt.figure(figsize=(8, 6))
        al = self.config.model.alignn_layers
        gl = self.config.model.gcn_layers
        target = self.config.target
        exp_name = self.config.exp_name
        actgl = self.config.model.action_args.num_act_graph_layers
        actlgl = self.config.model.action_args.num_act_linegraph_layers
        plot_name = exp_name+'_'+target+'_'+str(al) + str(gl)+str(actgl) + str(actlgl)+'_'+key+'_'+str(count) + '.png'

        # Flatten both arrays to combine both columns
        plt.hist(in_logit.cpu().flatten(), bins=30, alpha=0.5, label='in_logit', density=True)
        plt.hist(out_logit.cpu().flatten(), bins=30, alpha=0.5, label='out_logit', density=True)

        plt.xlabel("Logit Value")
        plt.ylabel("Density")
        plt.title("Distribution of in_logit vs out_logit")
        plt.legend()
        plt.grid(True)
        save_dir = self.config.output_dir+'/'+plot_name
        plt.savefig(save_dir, bbox_inches='tight')

    def draw_lr_plot(self, suffix):
        al = self.config.model.alignn_layers
        gl = self.config.model.gcn_layers
        target = self.config.target
        exp_name = self.config.exp_name
        actgl = self.config.model.action_args.num_act_graph_layers
        actlgl = self.config.model.action_args.num_act_linegraph_layers
        plot_name = './'+self.config.output_dir+'/'+ 'learning_curves_'+exp_name+'_'+target+'_'+str(al) + str(gl)+str(actgl) + str(actlgl) +'_'+str(suffix)+ '.png'

        plt.plot(self.learning_rates, label='learning rates')
        #plt.plot(valid_err, label='validation error')

        plt.title('Learning Rate History')
        plt.ylabel('Learning Rates')
        plt.xlabel('Steps')
        plt.legend(['Learning Rates'], loc='upper right')
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close()
    '''
    def draw_grad_descent(self):
        plt.figure(figsize=(18, 6))
        for layer, grad_means in self.gradient_means.items():
            if 'in_act_graph_net' in layer:
                plt.plot(grad_means, label=layer)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), title="Layers")


        plt.xlabel("Training Steps")
        plt.ylabel("Gradient Mean")
        plt.title("Gradient Mean per Layer Over Training Steps")
        # plt.legend()
        plt.show()
        plt.savefig('./'+self.config.output_dir+'/in_act_grad_descent.png', bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(18, 6))
        for layer, grad_means in self.gradient_means.items():
            if 'out_act_graph_net' in layer:
                plt.plot(grad_means, label=layer)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), title="Layers")


        plt.xlabel("Training Steps")
        plt.ylabel("Gradient Mean")
        plt.title("Gradient Mean per Layer Over Training Steps")
        # plt.legend()
        plt.show()
        plt.savefig('./'+self.config.output_dir+'/out_act_grad_descent.png', bbox_inches='tight')
        plt.close()
    '''
    def draw_grad_descent(self,suffix):
        for layer, grad_means in self.gradient_means.items():
            plt.figure(figsize=(12, 6))
            plt.plot(grad_means, label=layer)
            plt.xlabel("Training Steps")
            plt.ylabel("Gradient Mean")
            plt.title("Gradient Mean per Layer Over Training Steps")
            plt.legend()
            plt.show()
            plt.savefig('./'+self.config.output_dir+'/'+layer+'_'+str(suffix)+'.png', bbox_inches='tight')
            plt.close()
     

        '''        
        for layer, grad_means in self.gradient_means.items():
            if 'out_act_graph_net' in layer:
                plt.figure(figsize=(12, 6))
                plt.plot(grad_means, label=layer)
                plt.xlabel("Training Steps")
                plt.ylabel("Gradient Mean")
                plt.title("Gradient Mean per Layer Over Training Steps")
                plt.legend()
                plt.show()
                plt.savefig('./'+self.config.output_dir+'/'+layer+'.png', bbox_inches='tight')

                plt.close()
        '''
    def save_gradient_mean(self, named_parameters):
        for name, param in named_parameters:
            if param.grad is not None:
                #self.gradient_means[name].append(param.grad.mean().item())
                self.gradient_means[name].append(torch.sqrt(torch.mean(param.grad ** 2)).item())

    def cal_dict(self, new, old={'g': [], 'lg': [], 'slg': [], 'gcn': []}, mul=1.):
        for k in ['g', 'lg', 'slg', 'gcn']:
            if len(new[k]) > 0:
                if not isinstance(new[k], np.ndarray):
                    new[k] = np.array(new[k])
                if len(old[k]) > 0:
                    old[k] += new[k] * mul
                else:
                    old[k] = new[k] * mul
        return old
        
    # def train_step(self, epoch, dataloader):
    #     avgloss = 0
    #     avg_se = {'g': np.array([0.]*self.config.model.alignn_layers), 
    #             'lg': np.array([0.]*self.config.model.alignn_layers), 
    #             'slg': np.array([]),
    #             'gcn': np.array([0.]*self.config.model.gcn_layers),}
    #     tot_ng = 0
    #     keep_bond_forcing = True if epoch < self.max_bond_forcing else False
    #     print('Epoch and loader from step')
    #     print(epoch)
    #     print(len(dataloader))
    #     self.model.train()

    #     # auxiliary loss function (cross entropy)
    #     aux_loss_fn = torch.nn.CrossEntropyLoss()
    #     aux_weight = getattr(self, 'aux_loss_weight', 1.0)

    #     for i, data in enumerate(dataloader):
    #         g, lg, y = data
    #         g, lg, y = g.to(self.device), lg.to(self.device), y.to(self.device)
    #         ng = g.batch_size
    #         tot_ng += ng

    #         context = torch.enable_grad() if epoch % 5 == 0 else torch.no_grad()
    #         out, survived_edges, _, in_logit, out_logit = self.model((g, lg), keep_bond_forcing, context, False)

    #         # main loss
    #         loss = self.loss_fn(out, y)

    #         # aux CE losses
    #         if in_logit is not None:
    #             target_in = torch.ones(in_logit.size(0), dtype=torch.long, device=self.device)
    #             loss_in = aux_loss_fn(in_logit, target_in)
    #         else:
    #             loss_in = torch.tensor(0., device=self.device)

    #         if out_logit is not None:
    #             target_out = torch.ones(out_logit.size(0), dtype=torch.long, device=self.device)
    #             loss_out = aux_loss_fn(out_logit, target_out)
    #         else:
    #             loss_out = torch.tensor(0., device=self.device)

    #         total_loss = loss + aux_weight * (loss_in + loss_out)

    #         avgloss += total_loss.detach().item() * ng
    #         avg_se = self.cal_dict(survived_edges, avg_se, mul=ng)

    #         self.optimizer.zero_grad()
    #         total_loss.backward()
    #         self.optimizer.step()
    #         self.scheduler.step()
    #         self.step_count += 1

    #         if i % 100 == 0:
    #             print(f"[{epoch}|{i}]: total={total_loss:.5f}, main={loss:.5f}, "
    #                 f"in_CE={loss_in:.5f}, out_CE={loss_out:.5f}, survived={survived_edges}")

    #     avgloss /= tot_ng
    #     avg_se = {k: v / tot_ng for k, v in avg_se.items()}
    #     print('*******************************Step Count************************************')
    #     print(self.step_count)
    #     return avgloss, avg_se

    def train_step(self, epoch, dataloader):
        avgloss = 0
        avg_se = {'g': np.array([0.]*self.config.model.alignn_layers), 
                  'lg': np.array([0.]*self.config.model.alignn_layers), 
                  'slg': np.array([]),
                  'gcn': np.array([0.]*self.config.model.gcn_layers),}
        num_iters = len(dataloader)
        tot_ng = 0
        keep_bond_forcing = True if epoch < self.max_bond_forcing else False
        #print('Epoch and loader from step')
        #print(epoch)
        #print(len(dataloader))
        self.model.train()
        for i, data in enumerate(dataloader):
            g, lg, y = data
            g = g.to(self.device)
            lg = lg.to(self.device)
            y = y.to(self.device)
            ng = g.batch_size
            tot_ng += ng

            context = torch.enable_grad() if epoch % 5 == 0 else torch.no_grad()
            
            out, survived_edges, w_bonds, _, _ = self.model((g, lg), keep_bond_forcing, context, False)
            task_loss = self.loss_fn(out, y)
            sparsity_loss = self.sparsity_weight * w_bonds.mean() if self.sparsity_weight > 0 and w_bonds is not None else 0.
            loss = task_loss + sparsity_loss
            avgloss += task_loss.detach().item() * ng
            avg_se = self.cal_dict(survived_edges, avg_se, mul=ng)

            self.optimizer.zero_grad()
            loss.backward()
            
            self.save_gradient_mean(self.model.named_parameters())
            lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(lr)
            self.optimizer.step()
            self.scheduler.step()
            self.step_count = self.step_count + 1

            #if i%100 == 0:
                #print(f"[{epoch}|{i}]: {loss:.5f}, {survived_edges}")

        avgloss /= tot_ng
        # avg_se = self.cal_dict(avg_se, mul=1/tot_ng)
        avg_se = {k: v/tot_ng for k, v in avg_se.items()}
        print('*******************************Step Count************************************')
        print(self.step_count)
        return avgloss, avg_se

    @torch.no_grad()
    def eval_step(self, epoch, dataloader, mode):
        avgloss = 0
        avg_se = {'g': np.array([0.]*self.config.model.alignn_layers), 
                  'lg': np.array([0.]*self.config.model.alignn_layers), 
                  'slg': np.array([]),
                  'gcn': np.array([0.]*self.config.model.gcn_layers),}
        num_iters = len(dataloader)
        tot_ng = 0
        keep_bond_forcing = False
        
        self.model.eval()
        for i, data in enumerate(dataloader):
            g, lg, y = data
            g = g.to(self.device)
            lg = lg.to(self.device)
            y = y.to(self.device)
            ng = g.batch_size
            tot_ng += ng

            context = torch.no_grad()
            out, survived_edges,_,_,_ = self.model((g, lg), keep_bond_forcing, context, True)
            loss = self.loss_fn(out, y)
            avgloss += loss.detach().item() * ng
            avg_se = self.cal_dict(survived_edges, avg_se, mul=ng)

            if i%100 == 0:
                print(f"[{mode}|{epoch}|{i}]: {loss:.5f}, {survived_edges}")

        avgloss /= tot_ng
        # avg_se = self.cal_dict(avg_se, mul=1/tot_ng)
        avg_se = {k: v/tot_ng for k, v in avg_se.items()}
        return avgloss, avg_se

    def train(self, train_loader, val_loader, l, suffix=None):
        tic = time.perf_counter()
        # torch.save(stats, f'{self.config.output_dir}/stats_{self.config.suffix}.pt') # save mean and std
        suffix = self.config.suffix + "_" + str(suffix) if suffix is not None else self.config.suffix
        print(f'saved directory: {self.config.output_dir}, saved name: {suffix}')
        start_epoch, bests, paths = self._load_checkpoint('train')
        best_val_loss, best_val_epoch = bests
        ckpt_path, best_ckpt_path = paths
        train_err = []
        val_err = []
        suffix = l
        '''
        for name, param in self.model.named_parameters():
            if 'out_act_graph_net' in name:
                param.requires_grad = False  # Freeze out act layers
        '''

        if self.config.optim.scheduler.scheduler == "onecycle":
            self.scheduler = self.scheduler(steps_per_epoch=len(train_loader))
        #print('-------------------Init Loader-------------------------')
        #print(len(train_loader))

        for epoch in range(start_epoch, self.num_epochs):
            tic_epoch = time.perf_counter()
            
            train_loss, train_survived_edges = self.train_step(epoch, train_loader)
            val_loss, val_survived_edges = self.eval_step(epoch, val_loader, 'val')
            train_err.append(train_loss)
            val_err.append(val_loss)
            # train_survived_edges = self.cal_dict(train_survived_edges, mul=100)
            # val_survived_edges = self.cal_dict(val_survived_edges, mul=100)
            train_survived_edges = {k: v*100 for k, v in train_survived_edges.items()}
            val_survived_edges = {k: v*100 for k, v in val_survived_edges.items()}
            
            self._logging(epoch, train_loss, val_loss, train_survived_edges, val_survived_edges)  
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch
                model_ckpt = self._get_checkpoint_dict(epoch, best_val_loss, best_val_epoch)
                best_ckpt_path = self._save_checkpoint(model_ckpt, best_ckpt_path, 'best', suffix=suffix)            
            model_ckpt = self._get_checkpoint_dict(epoch, best_val_loss, best_val_epoch)
            ckpt_path = self._save_checkpoint(model_ckpt, ckpt_path, epoch, suffix=suffix) 

            toc_epoch = time.perf_counter()
            self.epoch_summary(epoch, train_loss, val_loss, 
                               train_survived_edges,
                               val_survived_edges,
                               best_val_loss, best_val_epoch)
            print(f'{self.device} run time: {(toc_epoch - tic_epoch):.4f} s')
            print("----------\n")

            # if self._early_stopping(epoch, best_val_epoch):
            #     print(f'Training has been early stopped at epoch {epoch}.')
            #     break

        toc = time.perf_counter()
        #self.draw_grad_descent(suffix)
        gradient_means_dict = dict(self.gradient_means)
        with open(f"{self.config.output_dir}/gradient_rms_{suffix}.json", 'w') as json_file:
            json.dump(gradient_means_dict, json_file, indent=4)
        print(f'total {self.device} run time: {toc - tic} s')
        self.draw_plot(train_err, val_err, suffix)
        #self.draw_lr_plot(suffix)
    def test(self, loaders, report_keys,l, root_dir=None, suffix=None):
        if (not isinstance(loaders, list)) and (not isinstance(loaders, tuple)):
            loaders = [loaders]
            report_keys = [report_keys]

        report_data = {k: {} for k in report_keys}
        suffix=l
        for loader, k in zip(loaders, report_keys):            
            pred_dict, survived_edges, metrics, inference_time = self.predict(loader, k, root_dir, suffix=suffix)
            mae_metric, mse_metric = metrics

            # survived_edges = self.cal_dict(survived_edges, mul=100)
            survived_edges = {k: v*100 for k, v in survived_edges.items()}

            report_data[k]['predictions'] = pred_dict
            report_data[k]['mae_metric'] = mae_metric
            report_data[k]['mse_metric'] = mse_metric
            report_data[k]['survived_edges'] = survived_edges
            report_data[k]['inference_time'] = inference_time

        return report_data

    @torch.no_grad()
    def predict(self, loader, key, root_dir=None, suffix=None):
        from tqdm import tqdm
        import pickle

        # Load model
        suffix = self.config.suffix + "_" + str(suffix) if suffix is not None else self.config.suffix
        self._load_checkpoint('predict', suffix=suffix)
        self.model.eval()

        # Etc
        # mean, std = torch.load(f'{self.config.ckpt_dir}/stats_{self.config.suffix}.pt')

        avg_se = {'g': np.array([0.]*self.config.model.alignn_layers), 
                  'lg': np.array([0.]*self.config.model.alignn_layers), 
                  'slg': np.array([]),
                  'gcn': np.array([0.]*self.config.model.gcn_layers),}
        mae_metric, mse_metric = 0, 0
        tot_ng = 0
        pred_dict = {}
        prob_dict = {}
        total_inference_time = 0

        print("Start prediction")

        for n, data in tqdm(enumerate(loader), total=len(loader), desc=f"{key.upper()}"):
            if len(data) == 2:
                g, lg = data
                y = None
            elif len(data) == 3:
                g, lg, y = data
                y = y.to(self.device)
            g = g.to(self.device)
            lg = lg.to(self.device)
            ng = g.batch_size
            
            unbatched_graphs = dgl.unbatch(g)
            '''
            # Iterate through the unbatched graphs
            for i, ug in enumerate(unbatched_graphs):
                print(f"Graph {i+1}:")
                print(ug.edges())
            '''
            graph_output_dir = key+'_'+suffix+'_graphs'
            torch.cuda.synchronize(self.device)
            t0 = time.time()
            pred, survived_edges,edge_status,in_logit,out_logit  = self.model((g, lg), keep_bond_forcing=False,is_hard=False,is_eval=True)
            torch.cuda.synchronize(self.device)
            t1 = time.time()
            total_inference_time += t1 - t0
            
            if in_logit is not None and out_logit is not None:
                self.plot_logit_distributions(in_logit, out_logit,key,n)

            curr_position = 0
            #sm_edge_status = [0 if x<0.5 else x for x in edge_status]
            sm_edge_status = edge_status
            for i, ug in enumerate(unbatched_graphs):
                    #print('-----------------------------------------------------')
                    #print(f"Graph {i+1}:")

                    # u1,v1 = ug.edges()
                ug_bcast = copy.deepcopy(ug)
                rem_index = []
                k=0
           
                for j in range(curr_position,curr_position+len(ug_bcast.edges()[0])):
                    if sm_edge_status[j] == 0:
                        rem_index.append(k)
                    k=k+1
                prob_dist = sm_edge_status[curr_position:curr_position+len(ug_bcast.edges()[0])]
                bond_lengths = torch.norm(ug.edata['r'], dim=1).cpu().tolist() if 'r' in ug.edata else []
                prob_dict[g.gdata["jid"][i]] = {'key': key, 'prop_dist': prob_dist.tolist(), 'bond_lengths': bond_lengths}
                
                curr_position= curr_position+len(ug_bcast.edges()[0])
                rem_tensor= torch.tensor(rem_index,device=self.device)
                rem_tensor = rem_tensor.to(torch.int64)
                ug_bcast.remove_edges(rem_tensor)
                # --- NEW DEGREE TRACKING ---
                # Get the number of edges connected to each node before and after pruning
                orig_degrees = ug.in_degrees().cpu().tolist()
                kept_degrees = ug_bcast.in_degrees().cpu().tolist()
                
                # Store them in the JSON dictionary
                prob_dict[g.gdata["jid"][i]]['orig_degrees'] = orig_degrees
                prob_dict[g.gdata["jid"][i]]['kept_degrees'] = kept_degrees
                #graph_output_dir = self.config.output_dir+'_'+key+'_'+suffix+'_graphs'
                #os.makedirs(graph_output_dir, exist_ok=True)

                prev_graph_name = './'+self.config.output_dir+'/'+graph_output_dir +'/'+g.gdata["jid"][i]+'_org.bin'
                bcast_graph_name ='./'+self.config.output_dir+'/'+graph_output_dir +'/'+g.gdata["jid"][i]+'_bcast.bin'
                dgl.save_graphs(prev_graph_name, [ug])
                dgl.save_graphs(bcast_graph_name,[ug_bcast])

            
            '''
            if (key == 'test'):
               
                for i, ug in enumerate(unbatched_graphs):
                    #print('-----------------------------------------------------')
                    #print(f"Graph {i+1}:")
                   
                    # u1,v1 = ug.edges()
                    ug_bcast = copy.deepcopy(ug)
                    rem_index = []
                    k=0
                    for j in range(curr_position,curr_position+len(ug_bcast.edges())):
                        if edge_status[j] == 0:
                            rem_index.append(k)
                        k=k+1
                    rem_tensor= torch.tensor(rem_index,device=self.device)
                    rem_tensor = rem_tensor.to(torch.int64)
                    ug_bcast.remove_edges(rem_tensor)
                    graph_output_dir = self.config.output_dir+'_graphs'
                    os.makedirs(graph_output_dir, exist_ok=True)

                    prev_graph_name = './graph_expln/'+ graph_output_dir +'/'+g.gdata["jid"][0]+'_org.bin'
                    bcast_graph_name = './graph_expln/'+ graph_output_dir +'/'+g.gdata["jid"][0]+'_bcast.bin'
                    dgl.save_graphs(prev_graph_name, [ug])
                    dgl.save_graphs(bcast_graph_name,[ug_bcast])
            '''
            pred = pred.view(-1) if len(pred.shape) == 0 else pred
            avg_se = self.cal_dict(survived_edges, avg_se, mul=ng)
            
            if y is not None:
                mae_metric += self.criterion('mae')(pred, y) * ng
                mse_metric += self.criterion('mse')(pred, y) * ng
                
            pred = pred.detach().cpu()             
            pred_dict.update({jid: out 
                              for jid, out in zip(g.gdata["jid"], pred)})
            tot_ng += ng

        # avg_se = self.cal_dict(avg_se, mul=1/tot_ng)
        avg_se = {k: v/tot_ng for k, v in avg_se.items()}
        mae_metric /= tot_ng
        mse_metric /= tot_ng
        
        with open(f"{self.config.output_dir}/{key}_{suffix}_predictions.pkl", "wb") as handle:
            pickle.dump(pred_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{self.config.output_dir}/{key}_{suffix}_prob_dict.json", 'w') as json_file:
            json.dump(prob_dict, json_file, indent=4)

        self.plot_explainability(key, suffix)
        print("Finish prediction")
        return pred_dict, survived_edges, (mae_metric, mse_metric), total_inference_time

    def plot_explainability(self, key, suffix):
        """Generate edge-weight vs bond-length explainability plots for one split/fold."""
        json_path = f"{self.config.output_dir}/{key}_{suffix}_prob_dict.json"
        if not os.path.exists(json_path):
            return
        with open(json_path) as f:
            prob_data = json.load(f)

        all_weights, all_lengths = [], []
        for jid, data in prob_data.items():
            weights = data.get('prop_dist', [])
            lengths = data.get('bond_lengths', [])
            if len(weights) == len(lengths) and len(weights) > 0:
                all_weights.extend(weights)
                all_lengths.extend(lengths)

        if not all_weights:
            print(f"No bond-length data for explainability plot [{key}_{suffix}]")
            return

        all_weights = np.array(all_weights, dtype=float)
        all_lengths = np.array(all_lengths, dtype=float)
        threshold = self.config.model.edge_threshold

        kept_mask = all_weights >= threshold
        kept_lengths = all_lengths[kept_mask]
        dropped_lengths = all_lengths[~kept_mask]

        # --- font / style settings for manuscript-quality output ---
        import matplotlib as mpl
        mpl.rcParams.update({
            'font.family':      'serif',
            'font.size':        14,
            'axes.labelsize':   14,
            'axes.labelweight': 'bold',
            'axes.titlesize':   13,
            'axes.titleweight': 'bold',
            'xtick.labelsize':  12,
            'ytick.labelsize':  12,
            'legend.fontsize':  12,
            'axes.linewidth':   1.8,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'xtick.major.size':  5,
            'ytick.major.size':  5,
        })

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter: edge weight vs bond length
        axes[0].scatter(all_lengths, all_weights, alpha=0.2, s=2, c='steelblue')
        axes[0].axhline(y=threshold, color='r', linestyle='--', linewidth=1.5,
                        label=f'threshold={threshold}')
        axes[0].set_xlabel('Bond Length (\u00c5)')
        axes[0].set_ylabel('Edge Weight')
        axes[0].set_title(f'{key} [{suffix}] \u2014 Edge Weight vs Bond Length')
        axes[0].legend()

        # Histogram: bond length distribution kept vs dropped
        axes[1].hist(kept_lengths, bins=50, alpha=0.6, density=True,
                     label=f'Kept ({kept_mask.sum()})', color='steelblue')
        axes[1].hist(dropped_lengths, bins=50, alpha=0.6, density=True,
                     label=f'Dropped ({(~kept_mask).sum()})', color='tomato')
        axes[1].set_xlabel('Bond Length (\u00c5)')
        axes[1].set_ylabel('Density')
        axes[1].set_title(f'{key} [{suffix}] \u2014 Bond Length: Kept vs Dropped')
        axes[1].legend()

        plt.tight_layout()
        plot_path = f"{self.config.output_dir}/explainability_{key}_{suffix}.png"
        plt.savefig(plot_path, bbox_inches='tight', dpi=200)
        plt.close()
        mpl.rcParams.update(mpl.rcParamsDefault)  # restore defaults
        print(f"Explainability plot saved: {plot_path}")

        summary = {
            'key': key,
            'suffix': suffix,
            'total_edges': int(len(all_weights)),
            'kept_edges': int(kept_mask.sum()),
            'dropped_edges': int((~kept_mask).sum()),
            'kept_fraction': float(kept_mask.mean()),
            'mean_weight': float(all_weights.mean()),
            'mean_bond_length_kept': float(kept_lengths.mean()) if len(kept_lengths) > 0 else None,
            'mean_bond_length_dropped': float(dropped_lengths.mean()) if len(dropped_lengths) > 0 else None,
        }
        summary_path = f"{self.config.output_dir}/explainability_{key}_{suffix}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Explainability summary saved: {summary_path}")

    def epoch_summary(self, epoch, 
                      train_loss, val_loss,
                      train_survived_edges, val_survived_edges, 
                      best_val_loss, best_val_epoch):        
        print("----------")
        print(f">>>>> Summary epoch {epoch}:")
        print(f"[{epoch}] -> train loss: {train_loss:.5f}, val loss: {val_loss:.5f}")
        print("\nTrain survived edges (%):")
        print(f"graph: {train_survived_edges['g']}")
        print(f"line graph: {train_survived_edges['lg']}")
        print(f"sub line graph: {train_survived_edges['slg']}")
        print(f"gcn: {train_survived_edges['gcn']}")
        print("\nVal survived edges (%):")
        print(f"graph: {val_survived_edges['g']}")
        print(f"line graph: {val_survived_edges['lg']}")
        print(f"sub line graph: {val_survived_edges['slg']}")
        print(f"gcn: {val_survived_edges['gcn']}")
        print("----------")
        print(f"Min val    [{best_val_epoch}]: {best_val_loss:.5f}")
        
