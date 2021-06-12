"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import yaml
from pytorch_lightning import Trainer, seed_everything

sys.path.append('/home/jc3/YYL/JS_fastMRI/SR_fastMRI-master/')

from ixi_module_t2net import UnetModule  # experimental.unet.unet_module


def main(args):
    """Main training routine."""
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    seed_everything(args.seed)
    model = UnetModule(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(args)

    # ------------------------
    # 3 START TRAINING OR TEST
    # ------------------------
    if args.mode == "train":
        trainer.fit(model)
    elif args.mode == "test":
        assert args.resume_from_checkpoint is not None
        trainer.test(model)
    else:
        # raise ValueError(f'unrecognized mode {args.mode} ')
        print('unkown mode')


def build_args():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    path_config = "ixi_config.yaml"

    with open(path_config) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        ixi_args = SimpleNamespace(**data)
        ixi_args.mask_path = ('./masks_mei/1D-Cartesian_6X_256.mat')

    data_path = data['data_dir']
    logdir = data['output_dir'] + "/dense_edsr/ixi/edsr_transformer"  #

    parent_parser = ArgumentParser(add_help=False)

    parser = UnetModule.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)

    num_gpus = 1
    backend = "ddp"
    batch_size = 8 if backend == "ddp" else num_gpus

    # module config
    config = dict(
        n_channels_in=1,
        n_channels_out=1,
        n_resgroups=5,  # 10
        n_resblocks=8,  # 20
        n_feats=64,  # 64
        lr=0.00005,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        data_path=data_path,
        exp_dir=logdir,
        exp_name="unet_demo",
        test_split="test",
        batch_size=batch_size,
        ixi_args=ixi_args,
    )
    parser.set_defaults(**config)

    # trainer config
    parser.set_defaults(
        gpus=num_gpus,
        max_epochs=50,
        default_root_dir=logdir,
        replace_sampler_ddp=(backend != "ddp"),
        distributed_backend=backend,
        seed=42,
        deterministic=True,
    )

    parser.add_argument("--mode", default="train", type=str)
    args = parser.parse_args()

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == "__main__":
    run_cli()
