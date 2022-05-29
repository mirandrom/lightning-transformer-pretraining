from dataclasses import dataclass, field
import os
from pathlib import Path

import datasets as ds
import pytorch_lightning as pl
import transformers as tr
import torch
from torch.utils.data import DataLoader

from .. import config as ltp_conf
from ..utils.logging import get_logger
from ..utils.fingerprint import fingerprint_dict

from typing import *
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

CACHE_DIR = os.path.join(ltp_conf.DEFAULT_LTP_DATA_CACHE, "hf_mlm")
logger = get_logger(__name__)

@dataclass
class HfMlmDataModuleConfig:
    dataset_name: str = field(
        default='wikitext',
        metadata={'help': 'Huggingface dataset name.'}
    )
    dataset_config_name: str = field(
        default='wikitext-2-raw-v1',
        metadata={'help': 'Huggingface dataset config name.'}
    )
    valid_split: float = field(
        default=0.05,
        metadata={'help': 'Fraction of dataset to reserve for validation.'}
    )
    text_col: str = field(
        default='text',
        metadata={'help': 'Name of text column in Huggingface dataset.'}
    )
    hf_tokenizer: str = field(
        default='bert-base-uncased',
        metadata={'help': 'Name of pretrained Huggingface tokenizer.'}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={'help': 'Masking probability for masked language modelling.'}
    )
    max_seq_len: int = field(
        default=128,
        metadata={'help': 'Maximum sequence length in tokens.'}
    )
    per_device_bsz: int = field(
        default=256,
        metadata={'help': 'Batch size (per device).'}
    )
    num_preprocess_workers: int = field(
        default=4,
        metadata={'help': 'Number of workers for dataset preprocessing.'}
    )
    num_dataloader_workers: int = field(
        default=0,
        metadata={'help': 'Number of workers for dataloader.'}
    )
    data_seed: int = field(
        default=1234,
        metadata={'help': 'Seed for dataset splitting and masking.'}
    )
    cache_dir: str = field(
        default=CACHE_DIR,
        metadata={'help': f"Directory for caching preprocessed dataset.\n"
                          f"Defaults to {CACHE_DIR}"}
    )
    overwrite: bool = field(
        default=False,
        metadata={'help': "Rerun preprocessing and overwrite cache."}
    )
    shuffle: bool = field(
        default=False,
        metadata={'help': "Shuffle dataset in dataloader."}
    )
    shuffle_seed: int = field(
        default=0xBE575E3D,
        metadata={'help': "Seed for shuffling dataset."}
    )

    def __post_init__(self):
        self.fingerprint = self._init_fingerprint()

    def _init_fingerprint(self):
        KEYS_TO_HASH = [
            'dataset_name',
            'dataset_config_name',
            'valid_split',
            'text_col',
            'hf_tokenizer',
            'max_seq_len'
        ]
        state = self.__dict__
        state = {k: state[k] for k in KEYS_TO_HASH}
        return fingerprint_dict(state)


class HfMlmDataModule(pl.LightningDataModule):
    def __init__(self, config: HfMlmDataModuleConfig):
        super().__init__()
        self.config = config
        self.cache_file_path = self._get_cache_file_path()

    def _get_cache_file_path(self):
        fp = self.config.fingerprint
        return os.path.join(self.config.cache_dir, fp)

    def prepare_data(self) -> None:
        if Path(self.cache_file_path).exists() and not self.config.overwrite:
            logger.info(f"Preprocessed dataset already cached in "
                        f"{self.cache_file_path}, skipping `prepare_data`.")
            return
        d = self._load_raw_datasets()
        d = self._preprocess_raw_datasets(d)
        d = self._split_raw_datasets(d)
        logger.info(f"Caching processed dataset to {self.cache_file_path}")
        d.save_to_disk(self.cache_file_path)

    def _load_raw_datasets(self):
        c = self.config
        if isinstance(c.dataset_name, list):
            raw_datasets_train = []
            raw_datasets_valid = []
            raw_datasets_test = []
            for name, config in zip(c.dataset_name, c.dataset_config_name):
                raw_dataset = ds.load_dataset(name, config)
                for split, rds in raw_dataset.items():
                    cols_to_remove = [c for c in rds.column_names if c != c.text_col]
                    raw_dataset[split] = rds.remove_columns(cols_to_remove)
                if 'train' in raw_dataset:
                    raw_datasets_train.append(raw_dataset['train'])
                if 'validation' in raw_dataset:
                    raw_datasets_valid.append(raw_dataset['validation'])
                if 'test' in raw_dataset:
                    raw_datasets_test.append(raw_dataset['test'])
            raw_datasets = ds.DatasetDict()
            if raw_datasets_train:
                raw_datasets['train'] = ds.concatenate_datasets(raw_datasets_train)
            if raw_datasets_valid:
                raw_datasets['validation'] = ds.concatenate_datasets(raw_datasets_valid)
            if raw_datasets_test:
                raw_datasets['test'] = ds.concatenate_datasets(raw_datasets_test)
        else:
            raw_datasets = ds.load_dataset(c.dataset_name, c.dataset_config_name)
        return raw_datasets

    def _preprocess_raw_datasets(self, d: ds.DatasetDict):
        c = self.config
        tokenizer = tr.AutoTokenizer.from_pretrained(c.hf_tokenizer)

        def batch_preprocess(examples):
            # filter out empty lines
            examples[c.text_col] = [
                line for line in examples[c.text_col]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[c.text_col],
                padding=False,
                truncation=True,
                max_length=c.max_seq_len,
                return_special_tokens_mask=True,
            )
        tokenized_datasets = d.map(
            batch_preprocess,
            batched=True,
            num_proc=c.num_preprocess_workers,
            remove_columns=d['train'].column_names,
            desc="Running tokenizer on dataset line_by_line",
        )

        return tokenized_datasets

    def _split_raw_datasets(self, d: ds.DatasetDict) -> ds.DatasetDict:
        if not self.config.valid_split:
            logger.info("No validation split specified, skipping validation split")
        elif "validation" in d:
            logger.info("Validation set already in raw datasets, skipping validation split")
        elif "train" in d:
            split_dataset = d["train"].train_test_split(
                test_size=self.config.valid_split,
                seed=self.config.data_seed,
            )
            d["train"] = split_dataset["train"]
            d["validation"] = split_dataset["test"]
        else:
            logger.info("Train set not in raw datasets, skipping validation split")
        return d

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ds.load_from_disk(self.cache_file_path)
        self.tokenizer = tr.AutoTokenizer.from_pretrained(self.config.hf_tokenizer)
        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    def collate_fn(self, examples: List[Dict[str, Any]]):
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            padding='max_length',
            max_length=self.config.max_seq_len
        )
        batch = batch.data # convert from batch encoding to dict for compatibility
        batch['labels'] = batch['input_ids'].clone()
        probability_matrix = torch.full(batch['labels'].shape, self.config.mlm_probability)
        special_tokens_mask = batch["special_tokens_mask"].bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        batch['labels'][~masked_indices] = -100
        batch['input_ids'][masked_indices] = self.mask_id
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        c = self.config
        rng = torch.Generator().manual_seed(c.shuffle_seed) if c.shuffle else None
        return DataLoader(
            self.dataset['train'],
            batch_size=c.per_device_bsz,
            collate_fn=self.collate_fn,
            num_workers=c.num_dataloader_workers,
            shuffle=c.shuffle,
            generator=rng,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        c = self.config
        rng = torch.Generator().manual_seed(c.shuffle_seed)
        return DataLoader(
            self.dataset['validation'],
            batch_size=c.per_device_bsz,
            collate_fn=self.collate_fn,
            num_workers=c.num_dataloader_workers,
            shuffle=True,
            generator=rng,
        )
