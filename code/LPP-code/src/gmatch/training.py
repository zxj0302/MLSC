
import numpy as np
import torch
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import TypedDict
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict
from collections import OrderedDict, deque


from gmatch.utils import color_str

def get_model_device(model):
    return next(model.parameters()).device

def get_device(gpu_index):
    if gpu_index >= 0:
        try:
            assert torch.cuda.is_available(), 'cuda not available'
            assert gpu_index >= 0 and gpu_index < torch.cuda.device_count(), 'gpu index out of range'
            return torch.device('cuda:{}'.format(gpu_index))
        except AssertionError:
            return torch.device('cpu')
    else:
        return torch.device('cpu')

def get_optimizer(model, optim, lr, l2):
    if optim == 'adam':
        return torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2)
    elif optim == 'sgd':
        return torch.optim.SGD( filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2)
    else:
        raise NotImplementedError

def get_scheduler(optimizer, step_size, gamma, last_epoch):
    from torch.optim.lr_scheduler import StepLR    
    scheduler = StepLR(optimizer, step_size, gamma, last_epoch)
    return scheduler


def save_fig(filename, fig):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def ax_plot(ax, data, label):
    ax.plot(data, label=label)

def set_ax(ax, metric='metric'):
    ax.set_title(metric)
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    ax.legend(loc='upper right')
    pass

def makedirs(dir):
    if not isinstance(dir, Path):
        dir = Path(dir)
    if not dir.exists():
        os.makedirs(dir)
    return dir

class Recorder(object):
    """
    """
    def __init__(self, minmax: Dict, checkpoint_dir: str, dataset: str, time_str: str, max_ckpts: int=5, args=None):
        """ 
        1. save metric results as csv files, at checkpoint_dir/dataset/time_str/record.csv
        2. save model state dict as .state_dict files, at checkpoint_dir/dataset/time_str/state.state_dict

        Args:
            minmax: for each metric, indicating the lower the better or the higher the better. 0: lower, 1: better.
                    e.g. {'loss': 0, 'acc': 1}
        

        Notes:
        self.full_metrics: {
            'train': [ {'loss': 0.1, 'mse': 0.2}, {...} ],
            'val': [ {'loss': 0.1, 'mse': 0.2}, {...} ],
            'test': [ {'loss': 0.1, 'mse': 0.2}, {...} ]

        }

        """
        self.minmax = minmax
        self.full_metrics = OrderedDict( {'train': [], 'val': [], 'test': []} )
        self.dataset = dataset
        self.time_str = time_str
        self.model_state = []
        self.checkpoint_dir = Path(checkpoint_dir) if not isinstance(checkpoint_dir, Path) else checkpoint_dir
        self.save_dir = self.checkpoint_dir/self.dataset/self.time_str
        self.saved_ckpts = deque() # only save a few recent checpoint dirs
        self.max_ckpts = max_ckpts
        self.dict_list = {}
        self.args = args
        self.args_saved = False

    def __getitem__(self, key):
        """ allow recorder['key'].append(value) """
        if self.dict_list.get(key, None) is None:
            self.dict_list[key] = []
        return self.dict_list[key]
    
    def save_args_to_json(self):
        import json
        args = vars(self.args)
        fn = self.save_dir/'args.json'
        
        for k, v in args.items():
            if isinstance(v, Path):
                args[k] = str(v)
        try:
            with open(fn, 'w') as f:
                json.dump(args, f, indent=4)
        except:
            print(color_str('Warning: dump args in Recorder() failed.', color='yellow'))

    def exsure_dir_exists(self, dir):
        dir = Path(dir)
        if not dir.exists():
            os.makedirs(dir)

    def draw_plots(self):
        """
        There will be `len(metric_keys)` plots.
        If `combine=True`, all plots are on one figure/pdf.
        Otherwise, each plot will be on one figure/pdf.
        """
        self.exsure_dir_exists(self.save_dir)
        from gmatch.utils import colors, markers, line_styles, fmt_iterator
        fmt_iter = fmt_iterator(colors, markers, line_styles)

        params = {
            'combine': False, # plot different metric on one graph or not
            'width': 6.4, # single plot width
            'height': 5.0, # single plot height
            'format': '.pdf', # .pdf, .png
            'colors': [], # color sequence
            'markers': [], # marker sequence
            'linestyles': [], # line style sequence
            'linewidth': None
        }
        num_metrics = len(self.full_metrics['train'])
        metric_keys = self.full_metrics['train'][0].keys()
        
        if params['combine'] is True:
            width = params['width']*1.1 * num_metrics
            height = params['height']
            figname = self.save_dir/('combined_metrics' + params['format'])
            fig, axes = plt.subplots(1, num_metrics, figsize=(width, height))
            for ax, key in zip(axes, metric_keys):
                for dname in ['train', 'val', 'test']:
                    if len(self.full_metrics[dname]) == 0: continue
                    data = list(map(lambda x:x[key], self.full_metrics[dname]))
                    ax.plot(range(len(data)), data, next(fmt_iter), label=dname)
                set_ax(ax, metric=key)
            save_fig(figname, fig)
        else:
            width = params['width']
            height = params['height']
            for key in metric_keys:
                figname = self.save_dir/(str(key) + params['format'])    
                fig, ax = plt.subplots(1, 1, figsize=(width, height))
                for dname in ['train', 'val', 'test']:
                    if len(self.full_metrics[dname]) == 0: continue
                    data = list(map(lambda x:x[key], self.full_metrics[dname]))
                    ax.plot(range(len(data)), data, next(fmt_iter), label=dname)
                set_ax(ax, metric=key)
                save_fig(figname, fig)

    def append_metrics(self, metrics_results, name):
        assert name in ['train', 'val', 'test']
        self.full_metrics[name].append(metrics_results)
    
    def get_best_metric(self, name):
        if len(self.full_metrics[name]) == 0:
            return None, None
        df = pd.DataFrame( self.full_metrics[name] )
        best_metric = {}
        best_epoch = {}
        for key in df.keys():
            if key not in set(self.minmax.keys()):
                print( color_str(f"Warning: metric {key} does not designate minmax, default 0.", 'yellow') )
            data = np.array(df[key])
            best_metric[key] = np.max(data) if self.minmax.get(key, 0) else np.min(data)
            best_epoch[key] = np.argmax(data) if self.minmax.get(key, 0) else np.argmin(data)
        return best_metric, best_epoch
    
    def get_latest_metric(self, name):
        latest_metric = self.full_metrics[name][-1]
        return latest_metric
    
    def append_model_state(self, state_dict):
        self.model_state.append(state_dict)

    def save_record(self):
        self.exsure_dir_exists(self.save_dir)
        if not self.args_saved:
            self.save_args_to_json()
            self.args_saved = True
        

        filename = self.save_dir/'record.csv'
        concates = []
        for key, metric_list in self.full_metrics.items():
            if len(metric_list) > 0:
                df = pd.DataFrame(metric_list)
                concates.append( df.rename(columns=lambda x: key+'_'+x) ) # 加前缀, 'train', ['val'], 'test'
        if len(concates) > 0:
            df = pd.concat(concates, axis=1 ) # combine train, val, test
            df.to_csv(filename, float_format='%.10f', index=True, index_label='epoch' )

        # save other data in dict_list, if exist
        if len(self.dict_list) > 0:
            for key, item in self.dict_list.items():
                filename = self.save_dir/"{}.csv".format(key)
                df = pd.DataFrame(item)
                df.to_csv(filename)

    @staticmethod
    def load_record(checkpoint_dir, dataset, time_str):
        """ load a record.csv file """
        filename = Path(checkpoint_dir)/dataset/time_str/'record.csv'
        assert filename.exists(), 'No such a file: {}'.format(filename)
        df = pd.read_csv(filename)
        return df
    
    def save_model(self, model, i, interval=2):
        """
        Args:
            latest: save the latest state_dict,
                    Otherwise, 对于每一个metric指标，保存test set上这个指标的最好的epoch的model
        """
        self.ensure_dir_exists(self.save_dir/'state_dicts')
        if i % interval == 0:
            dir = self.save_dir/'state_dicts'/f'epoch{i}.state_dict'
            torch.save( model.state_dict(), dir)
            self.saved_ckpts.append(dir)
            # if len(self.saved_ckpts) > self.max_ckpts:
            #     oldest = self.saved_ckpts.popleft()
            #     os.remove(oldest)

    @staticmethod
    def load_model(state_dict_fn, map_location=None):
        """ readin one state_dict """
        if map_location is None:
            state_dict = torch.load(state_dict_fn)
        else:
            state_dict = torch.load(state_dict_fn, map_location=map_location)
        return state_dict


class DatasetsDict(TypedDict):
    train: Dataset
    val: Dataset
    test: Dataset

class DataGenerator(object):
    def __init__(self, datasets: DatasetsDict, batch_size: int, collate_fn=None, device=None, batch_samplers=None):
        self.datasets = datasets
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.device = device
        self.batch_samplers = {} if batch_samplers is None else batch_samplers
        if self.collate_fn is None:
            from torch.utils.data._utils.collate import default_collate
            self.collate_fn = default_collate
        
        self.dataloaders = []
        for name in ['train', 'val', 'test']:
            if self.batch_samplers.get('train') is not None:
                self.dataloaders.append( DataLoader(datasets[name], batch_sampler=self.batch_samplers[name], pin_memory=True, collate_fn=self.collate_fn) )
            else:
                self.dataloaders.append( DataLoader(datasets[name], batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=self.collate_fn) )
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.dataloaders

        self.curr_dataloader = self.train_dataloader
        self.curr_dataloader_iterator = iter(self.train_dataloader)

    def __len__(self):
        return len(self.curr_dataloader)
    
    def next_batch(self):
        try:
            batch = next(self.curr_dataloader_iterator)
        except StopIteration:
            self.curr_dataloader_iterator = iter(self.curr_dataloader) 
            batch = next(self.curr_dataloader_iterator)
        
        for i, item in enumerate(batch): # Move data to the target device
            batch[i] = item.to(self.device)
        return batch
        
    def train(self):
        """Set to train mode(dataset)"""
        self.curr_dataloader = self.train_dataloader
        self.curr_dataloader_iterator = iter(self.train_dataloader)

    def val(self):
        """Set to val mode(dataset)"""
        self.curr_dataloader = self.val_dataloader
        self.curr_dataloader_iterator = iter(self.val_dataloader)

    def test(self):
        """Set to test mode(dataset)"""
        self.curr_dataloader = self.test_dataloader
        self.curr_dataloader_iterator = iter(self.test_dataloader)


class ModelWraper(object):
    def __init__(self, model):
        self.model = model
        pass

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
    
    def set_device(self, device):
        self.model = self.model.to(device)
    
    def compute(self, inputs, labels):
        raise NotImplementedError
    


def process_results(results_list):
    # detach torch values to scalars
    for i, dic in enumerate(results_list):
        scalar_dic = {}
        for k, v in dic.items():
            scalar_dic[k] = v.detach().cpu().item()
        results_list[i] = scalar_dic
    
    # average each metric
    results_dic = {}
    for i, dic in enumerate(results_list):
        for k, v in dic.items():
            results_dic[k] = results_dic.get(k, 0) + dic[k]
    
    for k, v in results_dic.items():
        results_dic[k] = results_dic[k] / len(results_list)
    
    return results_dic

def str_dict(dict):
    dict_str = ''
    items = list(dict.items())
    for i, (key, value) in enumerate(items):
        if isinstance(value, int) or isinstance(value, np.int64):
            dict_str = dict_str + str(key) + ':' + f' {value:d}'
        else:
            dict_str = dict_str + str(key) + ':' + f' {value:.12f}'
        if i != len(items) - 1:
            dict_str = dict_str + ', '
    string = '{' + f"{dict_str}" + '}'
    return string

class Trainer(object):
    def __init__(self, model_wraper: ModelWraper,
                    data_loader: DataGenerator, 
                    optimizer: Optimizer, 
                    scheduler: _LRScheduler,
                    recorder: Recorder,
                    logger,
                    epochs, device, time_str, checkpoint_dir, batch_filter=None, **kwargs):
        self.model_wraper = model_wraper
        self.data_loader = data_loader
        self.epochs = epochs
        self.device = device

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.time_str = time_str
        self.checkpoint_dir = makedirs(checkpoint_dir)
        self.batch_filter = batch_filter
        
        self.logger = logger # add logger 
        self.recorder = recorder # add recorder
        self.summarizer = None # add summarizer
        self.kwargs = kwargs
        self.val = self.kwargs.get('val', False) # whether to use the val set

        self.train_num_per_epoch = len(self.data_loader.train_dataloader) * self.data_loader.batch_size
        self.test_num_per_epoch = len(self.data_loader.test_dataloader) * self.data_loader.batch_size

        self.model_wraper.set_device(self.device)

    def step_compute(self, batch):
        for i, item in enumerate(batch): # Move data to the target device
            batch[i] = item.to(self.device)
        
        if self.batch_filter is not None and self.batch_filter(batch):
            return None
        
        if self.mode == 'train':
            try:
                step_results = self.model_wraper.compute(batch)
                self.optimizer.zero_grad()
                step_results['loss'].backward()
                self.optimizer.step()
            except Exception as e:
                if 'CUDA out of memory' in e.args[0]:
                    pass
                else: raise
        else:
            with torch.no_grad():
                step_results = self.model_wraper.compute(batch)

        return step_results


    def process_multiple_batchs(self, desc):
        num_batches = len(self.data_loader)
        allstep_results = []
        for batch in tqdm(self.data_loader.curr_dataloader, total=num_batches, desc=desc):
            step_results = self.step_compute(batch)
            if step_results is not None:
                allstep_results.append(step_results)
        results_dic = process_results(allstep_results)
        self.logger.info( color_str(str_dict(results_dic), 'green') )
        return results_dic


    def train_epoch(self):
        """ train one epoch """
        self.model_wraper.train()
        self.data_loader.train()
        self.mode = 'train'
        tqdm_desc = color_str(f'[train][epoch {self.cur_epoch+1}/{self.epochs}]', 'green')
        results_dic = self.process_multiple_batchs(tqdm_desc)
        self.scheduler.step()
        return results_dic

    def val_epoch(self):
        self.model_wraper.eval()
        self.data_loader.val()
        self.mode = 'val'
        tqdm_desc = color_str( f'[val  ][epoch {self.cur_epoch+1}/{self.epochs}]', 'green')
        results_dic = self.process_multiple_batchs(tqdm_desc)
        return results_dic

    def test_epoch(self):
        self.model_wraper.eval()
        self.data_loader.test()
        self.mode = 'test'
        tqdm_desc = color_str( f'[test ][epoch {self.cur_epoch+1}/{self.epochs}]', 'green' )
        results_dic = self.process_multiple_batchs(tqdm_desc)
        return results_dic

    def train(self):
        """ Train model """
        train_results = []
        val_results = []
        test_results = []
        train_times = []
        test_times = []
        for i in range(self.epochs):
            self.cur_epoch = i
            # train
            start_time = time.time()
            tr_results = self.train_epoch()
            train_times.append(time.time() - start_time)
            train_results.append(tr_results)
            # val
            if self.val:
                va_results = self.val_epoch()
                val_results.append(va_results)
            # test
            start_time = time.time()
            te_results = self.test_epoch()
            test_times.append(time.time() - start_time)
            test_results.append(te_results)
            # record results
            if len(tr_results) > 0:
                self.recorder.append_metrics(tr_results, 'train') 
            if self.val and len(va_results) > 0: self.recorder.append_metrics(va_results, 'val')
            if len(te_results) > 0: self.recorder.append_metrics(te_results, 'test')
            # (update) records, plots
            self.recorder.save_record()
            self.recorder.draw_plots()
            # self.recorder.save_model(self.model_wraper.model, i)
            print('\n')
        self.logger.info(color_str("Best results:", 'cyan'))
        for dname in ['train', 'val', 'test']:
            best_metric, best_epoch = self.recorder.get_best_metric(dname)
            if best_metric is not None:
                self.logger.info( color_str(f"[{dname}]", 'green') )
                self.logger.info( color_str(f"value:{str_dict(best_metric)}", 'cyan') )
                self.logger.info( color_str(f"epoch:{str_dict(best_epoch)}", 'cyan') )
        
        total_results = {'train_results': train_results,
                        'val_results': val_results,
                        'test_results': test_results}
        train_time_per_sample = np.mean(train_times)/self.train_num_per_epoch*1000
        test_time_per_sample = np.mean(test_times)/self.test_num_per_epoch*1000
        print(f'Train time: {train_time_per_sample:.6f}ms, test time: {test_time_per_sample:.6f}ms (per sample)\n')
        return total_results
    def eval_ckpt(self, ckpt_file):
        state_dict = Recorder.load_model(ckpt_file, map_location=self.device)
        self.model_wraper.model.load_state_dict(state_dict)
        self.test_epoch()




    
