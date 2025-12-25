from argparse import Namespace
import os
import pdb
import warnings
import numpy as np
from tqdm import tqdm
from xmuda.data.build import build_dataloader
from xmuda.data.collate import get_collate_scn
from xmuda.common.utils.torch_util import worker_init_fn
from torch.utils.data.dataloader import DataLoader

def get_semkitti_points():
    cfg_path = ''
    save_path = ''

    print('pkl_path:',cfg_path)

    args = Namespace(
    ckpt2d='', 
    ckpt3d='', 
    config_file=cfg_path, 
    opts=[], 
    )

    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.DATASET_TARGET.SemanticKITTISCN.return_ori_points = True
    cfg.freeze()
    print('Loaded configuration file {:s}'.format(args.config_file))
    train_dataloader_trg = build_dataloader(cfg, mode='train',domain='target')
    train_dataset_trg = train_dataloader_trg.dataset

    collate_fn_val = get_collate_scn(is_train = True)
    val_loader = DataLoader(
        train_dataset_trg,
        batch_size=1,
        drop_last=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn_val
    )

    semkitti_points = []

    for idx, data_batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        # pdb.set_trace()
        # points = data_batch['x'][0][:,:3].numpy()
        points = data_batch['ori_points'][0]
        semkitti_points.append(points)

    save_dir = os.path.dirname(save_path)
    if not os.path.isdir(save_dir):
        warnings.warn('Make a new directory: {}'.format(save_dir))
        os.makedirs(save_dir,exist_ok=True)

    np.save(save_path,np.array(semkitti_points,dtype=object))
    print(f'save points to {save_path} !')
    
def main():
    get_semkitti_points()

if __name__ == '__main__':
    main()