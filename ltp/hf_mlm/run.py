from dataclasses import dataclass, field
from datetime import datetime as dt
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import transformers as tr
import wandb

from .data import HfMlmDataModule, HfMlmDataModuleConfig
from .model import HfMlmModel, HfMlmModelConfig
from ..config import LTP_RUN_CACHE


@dataclass
class HfMlmTrainerConfig:
    experiment_name: str = field(
        default=None,
        metadata={'help': 'Identifier for experiment.'}
    )
    timestamp_id: bool = field(
        default=True,
        metadata={'help': 'Add timestamp to experiment id.'}
    )
    log_dir: str = field(
        default=LTP_RUN_CACHE,
        metadata={'help': 'Parent directory where model checkpoints, logs, '
                          'and other artefacts will be stored. Note that '
                          'these are saved in a subdirectory named according '
                          'to the experiment identifier.'}
    )
    num_training_steps: int = field(
        default=int(1e5),
        metadata={'help': 'Maximum number of training steps for experiment.'}
    )
    save_every_n_steps: int = field(
        default=None,
        metadata={'help': 'Save a model checkpoint every n steps.'}
    )
    save_last: bool = field(
        default=True,
        metadata={'help': 'Save a model checkpoint at the end of training.'}
    )
    log_every_n_steps: int = field(
        default=100,
        metadata={'help': 'Log metrics during training every n steps.'}
    )
    eval_every_n_steps: int = field(
        default=1000,
        metadata={'help': 'Run validation step during training every n steps.'}
    )
    limit_val_batches: int = field(
        default=0,
        metadata={'help': 'Limit validation batches. Set to 0.0 to skip.'}
    )
    precision: int = field(
        default=32,
        metadata={'help': 'Precision flag for pytorch-lightning Trainer.'}
    )
    total_bsz: int = field(
        default=None,
        metadata={'help': 'Total batch size (use with `per_device_bsz` to '
                          'automatically determine gradient accumulation '
                          'steps consistent with number of nodes/devices). '
                          'If None, is set to `per_device_bsz` times the '
                          'total number of devices across nodes.'}
    )
    accelerator: str = field(
        default='gpu',
        metadata={'help': 'Accelerator flag for pytorch-lightning Trainer.'}
    )
    devices: str = field(
        default="1",
        metadata={'help': 'Number of accelerator devices or list of device ids.'
                          'Must be a str that will evaluate to a python '
                          'expression. E.g. "1" or "[0,1]".'}
    )
    num_nodes: int = field(
        default=1,
        metadata={'help': 'Number of nodes.'}
    )
    strategy: str = field(
        default=None,
        metadata={'help': 'Set to `ddp` for multiple devices or nodes.'}
    )
    fault_tolerant: bool = field(
        default=True,
        metadata={'help': 'Whether to run pytorch-lightning in fault-tolerant mode. \n'
                          'See: https://pytorch-lightning.readthedocs.io/en/1.6.3/advanced/fault_tolerant_training.html?highlight=fault%20tolerant \n'
                          'and: https://github.com/PyTorchLightning/pytorch-lightning/blob/1.6.3/pl_examples/fault_tolerant/automatic.py'}
    )
    slurm_auto_requeue: bool = field(
        default=True,
        metadata={'help': 'Whether to run pytorch-lightning with auto-requeue. \n'
                          'See: https://pytorch-lightning.readthedocs.io/en/1.6.3/clouds/cluster.html?highlight=slurm#wall-time-auto-resubmit'}
    )
    wandb_project: str = field(
        default=None,
        metadata={'help': 'Weights and bias project to log to.'}
    )
    wandb_entity: str = field(
        default="clac_nsl",
        metadata={'help': 'Weights and bias entity to log to.'}
    )

    experiment_id: str = field(init=False)
    experiment_dir: Path = field(init=False)

    def __post_init__(self):
        self.experiment_id = self.init_experiment_id()
        self.experiment_dir = self.init_experiment_dir()
        self.devices = eval(self.devices)
        if isinstance(self.devices, list):
            self.num_devices = len(self.devices)
        elif isinstance(self.devices, int):
            self.num_devices = self.devices
        else:
            raise ValueError("devices must be an int or a list of ints")

    def init_experiment_id(self):
        ts = dt.utcnow().isoformat(timespec='seconds')
        if self.experiment_name and self.timestamp_id:
            return f"{self.experiment_name}_{ts}"
        if self.experiment_name:
            return self.experiment_name
        if self.timestamp_id:
            return ts
        else:
            raise ValueError("If `experiment_name` is not specified, "
                             "`timestamp_id` must be True.")

    def init_experiment_dir(self):
        p = Path(self.log_dir) / self.experiment_id
        return p


def main():
    # parse arguments
    dc: HfMlmDataModuleConfig
    mc: HfMlmModelConfig
    tc: HfMlmTrainerConfig
    configs = (HfMlmDataModuleConfig, HfMlmModelConfig, HfMlmTrainerConfig)
    parser = tr.HfArgumentParser(configs)
    dc, mc, tc = parser.parse_args_into_dataclasses()

    # setup fault tolerant / auto-requeue
    if tc.fault_tolerant:
        os.environ["PL_FAULT_TOLERANT_TRAINING"] = "automatic"
    if tc.slurm_auto_requeue:
        plugins = [pl.plugins.environments.SLURMEnvironment(auto_requeue=True)]
    else:
        plugins = [pl.plugins.environments.SLURMEnvironment(auto_requeue=False)]


    # setup gradient accumulation
    if tc.total_bsz is None:
        tc.total_bsz = dc.per_device_bsz * tc.num_devices * tc.num_nodes
    accumulate_grad_batches = (tc.total_bsz // dc.per_device_bsz
                               // tc.num_devices // tc.num_nodes)
    assert (tc.total_bsz == dc.per_device_bsz * tc.num_devices
            * accumulate_grad_batches * tc.num_nodes)


    # callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=tc.save_every_n_steps,
        save_last=tc.save_last,
        verbose=True,
        save_top_k=-1,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='step',
    )
    trainer_callbacks = [checkpoint_callback, lr_monitor]

    # logger
    if tc.wandb_project:
        logger = WandbLogger(
            project=tc.wandb_project,
            entity=tc.wandb_entity,
            name=tc.experiment_id,
            id=tc.experiment_id,
        )
    else:
        # will use tensorboard logger by default
        logger = True

    trainer = pl.Trainer(
        max_steps=tc.num_training_steps,
        max_epochs=None,
        log_every_n_steps=tc.log_every_n_steps,
        val_check_interval=tc.eval_every_n_steps,
        callbacks=trainer_callbacks,
        precision=tc.precision,
        strategy=tc.strategy,
        accelerator=tc.accelerator,
        devices=tc.devices,
        num_nodes=tc.num_nodes,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger,
        default_root_dir=str(tc.experiment_dir),
        gradient_clip_val=mc.gradient_clip_val,
        limit_val_batches=tc.limit_val_batches,
        plugins=plugins,
    )
    model = HfMlmModel(mc)
    trainer.fit(model)
    wandb.finish()