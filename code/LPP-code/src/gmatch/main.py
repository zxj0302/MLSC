import time
import os
import sys
import argparse
from pathlib import Path

from gmatch.utils import interrupt_wraper

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)) ).parent.parent # NNmatching/
TMP_DIR = ROOT_DIR/'data'/'tmp'


def parse_args(argstring=None):
    parser = argparse.ArgumentParser('LPP.')
    parser.add_argument('--root_dir', type=Path, default=ROOT_DIR, help='Root directory')
    # dataset
    parser.add_argument('--dataset', type=str, default='g8_graphs', help='Dataset name')
    parser.add_argument('--pattern', type=str, default='p_htw3_5_7_10', help='Pattern name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size' )
    parser.add_argument('--r_sampler', default=False, action='store_true', help='Rooted graph sampler')

    # model
    parser.add_argument('--model', type=str, default='cnn', help='Model for the f' )
    parser.add_argument('--in_fea', type=int, default=64, help='Input data dim' )
    parser.add_argument('--mid_fea', type=int, default=512, help=' data dim' )
    parser.add_argument('--out_fea', type=int, default=1, help='Output dim' )

    # training
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs' )
    parser.add_argument('--device', type=str, default='cuda:0', help='Default device')

    # optimizer
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer' )
    parser.add_argument('--lr', type=float, default=2*1e-4, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='L2 value' )

    # scheduler
    parser.add_argument('--sh_ss', type=int, default=200, help='Step size' )
    parser.add_argument('--sh_ga', type=float, default=0.5, help='Gamma' )
    parser.add_argument('--sh_le', type=int, default=-1, help='Last epoch' )

    # recorder
    parser.add_argument('--ckpt_dir', type=Path, default=ROOT_DIR/'checkpoints', help='Ckpt dir')
    parser.add_argument('--max_ckpts', type=int, default=10, help='Max ckpts to save')
    parser.add_argument('--time_str', type=str, default=time.strftime('%Y_%m_%d_%H_%M_%S'), help='Ececution time str')
    parser.add_argument('--desc', type=str, default='desc string', help='Description string')

    try:
        if argstring is not None:
            args = parser.parse_args(argstring)
        else:
            args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args


@interrupt_wraper
def main():
    from gmatch.model import LPPMLP, LPPModelWraper, LPPCNN, LPPTransformer
    from gmatch.dataset import LPPDataset, collate_fn, WholeGraphSampler, batch_filter
    from gmatch.preprocessing import PrepareDataset
    from gmatch.training import DataGenerator, Trainer, Recorder, get_optimizer, get_scheduler
    from gmatch.utils import get_default_logger

    args = parse_args()
    try:
        test_dataset = LPPDataset(args.dataset, args.pattern, 'train')
    except FileNotFoundError:
        command = f"python -m gmatch.preprocessing -d {args.dataset} -p {args.pattern} --confirm"
        os.system(command)
    
    # datasets
    datasets = {
        'train': LPPDataset(args.dataset, args.pattern, 'train'),
        'val': LPPDataset(args.dataset, args.pattern, 'val'),
        'test': LPPDataset(args.dataset, args.pattern, 'test'),
    }
    args.max_nodes = datasets['train'].max_nodes
    if not args.r_sampler: # WholeGraphSampler
        sampler_index_fn = args.root_dir/'data'/f'{args.dataset}_{args.pattern}'/PrepareDataset.filenames['w_sampler_index']
        batch_samplers = {
            'train': WholeGraphSampler(sampler_index_fn, 'train', args.batch_size),
            'val': WholeGraphSampler(sampler_index_fn, 'val', args.batch_size),
            'test': WholeGraphSampler(sampler_index_fn, 'test', args.batch_size),
        }
    else: batch_samplers = None
    data_loader = DataGenerator(datasets, batch_size=args.batch_size, collate_fn=collate_fn, device=args.device, batch_samplers=batch_samplers)

    # model
    if args.model == 'mlp':
        model = LPPMLP(args.max_nodes*args.max_nodes, args.out_fea, args.mid_fea)
    elif args.model == 'cnn':
        model = LPPCNN(args.max_nodes)
    elif args.model == 'tf':
        model = LPPTransformer(args.max_nodes)
    model_wraper = LPPModelWraper(model)

    # configs
    optimizer = get_optimizer(model, args.optim, args.lr, args.l2)
    scheduler = get_scheduler(optimizer, args.sh_ss, args.sh_ga, args.sh_le)
    minmax = {'loss': 0, 'rmse': 0, 'mae': 0, 'label_mean': 0, 'w_mse': 0, 'w_rmse': 0, 'w_label_mean': 1}
    recorder = Recorder(minmax, args.ckpt_dir, f"{args.dataset}_{args.pattern}", args.time_str, args.max_ckpts, args=args)
    logger = get_default_logger(ROOT_DIR/'logs'/f"{args.dataset}_{args.pattern}"/f"{args.time_str}", args, sys.argv)

    # trainer
    trainer = Trainer(model_wraper, data_loader, optimizer, scheduler, recorder, logger, \
        args.epochs, 
        args.device, 
        args.time_str, 
        args.ckpt_dir,
        val=False,
        batch_filter=batch_filter)

    # train
    trainer.train()


if __name__ == '__main__':
    main()