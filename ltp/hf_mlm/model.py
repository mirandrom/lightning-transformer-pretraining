from dataclasses import dataclass, field

import pytorch_lightning as pl
from torch.optim import AdamW
import transformers as tr

from typing import *


@dataclass
class HfMlmModelConfig:
    model_name: str = field(
        default='bert-base-uncased',
        metadata={'help': 'Huggingface model name; e.g. `bert-base-uncased`.'}
    )
    pretrained: bool = field(
        default=False,
        metadata={'help': 'Whether to load pretrained model weights.'}
    )
    seed: int = field(
        default=1337,
        metadata={'help': 'Seed to be used by `pl.seed_everything`.'}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={'help': 'Learning rate (peak if used with scheduler).'}
    )
    lr_scheduler_name: str = field(
        default='constant',
        metadata={'help': 'Name of learning rate scheduler from huggingface '
                          'transformers library. Valid options are '
                          f'{[x.value for x in tr.optimization.SchedulerType]}'
                  }
    )
    num_warmup_steps: int = field(
        default=None,
        metadata={'help': 'Number of warmup steps for `lr_scheduler_name`.'}
    )
    num_warmup_total_steps: int = field(
        default=None,
        metadata={'help': 'Number of total steps for `lr_scheduler_name`.'}
    )
    gradient_clip_val: float = field(
        default=0,
        metadata={'help': 'Gradient clipping (by L2 norm); '
                          'for use by pytorch-ligtning `Trainer`. '}
    )
    adam_eps: float = field(
        default=1e-8,
        metadata={'help': 'Hyperparameter for AdamW optimizer.'}
    )
    adam_wd: float = field(
        default=0.0,
        metadata={'help': 'Weight decay for AdamW optimizer.'}
    )
    adam_beta_1: float = field(
        default=0.9,
        metadata={'help': 'Hyperparameter for AdamW optimizer.'}
    )
    adam_beta_2: float = field(
        default=0.999,
        metadata={'help': 'Hyperparameter for AdamW optimizer.'}
    )


class HfMlmModel(pl.LightningModule):
    def __init__(self, config: HfMlmModelConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.__dict__)
        pl.seed_everything(self.config.seed)
        self.model = self.init_model()

    def init_model(self):
        if self.config.pretrained:
            return tr.AutoModelForMaskedLM.from_pretrained(self.config.model_name)
        else:
            hf_config = tr.AutoConfig.from_pretrained(self.config.model_name)
            return tr.AutoModelForMaskedLM.from_config(hf_config)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.adam_wd,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.config.learning_rate,
                          betas=(self.config.adam_beta_1,
                                 self.config.adam_beta_2),
                          eps=self.config.adam_eps
                          )
        lr_scheduler = tr.get_scheduler(
            name=self.config.lr_scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.config.num_warmup_total_steps,
        )
        lr_dict = dict(
            scheduler=lr_scheduler,
            interval="step",
        )
        return dict(optimizer=optimizer, lr_scheduler=lr_dict)

    def forward(self, batch):
        out = self.model(batch['input_ids'],
                         labels=batch['labels'],
                         attention_mask=batch['attention_mask'],
                         output_hidden_states=True)
        return out

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = out.loss
        self.log('train_loss', loss.item())
        # manually log global_step to prevent issues with resuming from ckpt
        # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/13163
        self.log('global_step', self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = out.loss
        self.log('eval_loss', loss.item())
        # manually log global_step to prevent issues with resuming from ckpt
        # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/13163
        self.log('global_step', self.global_step)
        return loss
