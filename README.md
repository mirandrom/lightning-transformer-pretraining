# lightning-transformer-pretraining
Pretraining transformer language models with pytorch-lightning.

This repository is a work in progress, with the main goal of providing skeleton structures over which to build simple yet scalable experiment workflows for language model pretraining using pytorch-lightning.



## Setup
### Installation
Requires Python 3.10
```bash
git clone https://github.com/mirandrom/lightning-transformer-pretraining.git
cd lightning-transformer-pretraining
pip install -e .
```

### Cache Directories
Optionally, set the following environment variables:
- `LTP_DATA_CACHE`: where to save preprocessed datasets as pyarrow tables;
- `LTP_RUN_CACHE`: where to save lightning logs and other run artifacts;

Alternatively, you can set `LTP_CACHE` 
and datasets will be saved in a `data` subdirectory, 
while runs will be saved in a `runs` subdirectory.
## Examples
### Masked Language Model Pretraining with Huggingface
To see all options, run:
```bash
python ltp/hf_mlm/run.py --help

options:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        Huggingface dataset name(s); e.g. `wikitext`. (default: wikitext)
  --dataset_config_name DATASET_CONFIG_NAME
                        Huggingface dataset config name(s), aligned with `dataset_name` if list; e.g. `wikitext-2-raw-v1` for `wikitext` (default: wikitext-2-raw-v1)
  --valid_split VALID_SPLIT
                        Huggingface dataset config name(s), aligned with `dataset_name` if list; e.g. `20200501.en` for `wikipedia` (default: 0.05)
  --text_col TEXT_COL   Name of text column in Huggingface dataset. (default: text)
  --hf_tokenizer HF_TOKENIZER
                        Name of pretrained Huggingface tokenizer. (default: bert-base-uncased)
  --mlm_probability MLM_PROBABILITY
                        Masking probability for masked language modelling. (default: 0.15)
  --max_seq_len MAX_SEQ_LEN
                        Maximum sequence length in tokens. (default: 128)
  --per_device_bsz PER_DEVICE_BSZ
                        Batch size (per device). (default: 256)
  --num_preprocess_workers NUM_PREPROCESS_WORKERS
                        Number of workers for dataset preprocessing. (default: 4)
  --num_dataloader_workers NUM_DATALOADER_WORKERS
                        Number of workers for dataloader. (default: 0)
  --data_seed DATA_SEED
                        Seed for dataset splitting and masking. (default: 1234)
  --cache_dir CACHE_DIR
                        Directory for caching preprocessed dataset. Defaults to /home/mila/m/mirceara/.cache/ltp/data/hf_mlm (default: /home/mila/m/mirceara/.cache/ltp/data/hf_mlm)
  --overwrite [OVERWRITE]
                        Rerun preprocessing and overwrite cache. (default: False)
  --model_name MODEL_NAME
                        Huggingface model name; e.g. `bert-base-uncased`. (default: bert-base-uncased)
  --pretrained [PRETRAINED]
                        Whether to load pretrained model weights. (default: False)
  --model_seed MODEL_SEED
                        Seed to be used by `pl.seed_everything`. (default: 1337)
  --learning_rate LEARNING_RATE
                        Learning rate (peak if used with scheduler). (default: 5e-05)
  --lr_scheduler_name LR_SCHEDULER_NAME
                        Name of learning rate scheduler from huggingface transformers library. Valid options are ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'] (default: constant)
  --num_warmup_steps NUM_WARMUP_STEPS
                        Number of warmup steps for `lr_scheduler_name`. (default: None)
  --num_warmup_total_steps NUM_WARMUP_TOTAL_STEPS
                        Number of total steps for `lr_scheduler_name`. (default: None)
  --gradient_clip_val GRADIENT_CLIP_VAL
                        Gradient clipping (by L2 norm); for use by pytorch-lightning `Trainer`. (default: 0)
  --adam_eps ADAM_EPS   Hyperparameter for AdamW optimizer. (default: 1e-08)
  --adam_wd ADAM_WD     Weight decay for AdamW optimizer. (default: 0.0)
  --adam_beta_1 ADAM_BETA_1
                        Hyperparameter for AdamW optimizer. (default: 0.9)
  --adam_beta_2 ADAM_BETA_2
                        Hyperparameter for AdamW optimizer. (default: 0.999)
  --experiment_name EXPERIMENT_NAME
                        Identifier for experiment. (default: None)
  --timestamp_id [TIMESTAMP_ID]
                        Add timestamp to experiment id. (default: True)
  --no_timestamp_id     Add timestamp to experiment id. (default: False)
  --log_dir LOG_DIR     Parent directory where model checkpoints, logs, and other artefacts will be stored. Note that these are saved in a subdirectory named according to the experiment identifier. (default:
                        /home/mila/m/mirceara/.cache/ltp/runs)
  --num_training_steps NUM_TRAINING_STEPS
                        Maximum number of training steps for experiment. (default: 100000)
  --save_every_n_steps SAVE_EVERY_N_STEPS
                        Save a model checkpoint every n steps. (default: None)
  --save_last [SAVE_LAST]
                        Save a model checkpoint at the end of training. (default: True)
  --no_save_last        Save a model checkpoint at the end of training. (default: False)
  --log_every_n_steps LOG_EVERY_N_STEPS
                        Log metrics during training every n steps. (default: 100)
  --eval_every_n_steps EVAL_EVERY_N_STEPS
                        Run validation step during training every n steps. (default: 1000)
  --limit_val_batches LIMIT_VAL_BATCHES
                        Limit validation batches. Set to 0.0 to skip. (default: 0)
  --skip_eval [SKIP_EVAL]
                        Skip eval loop when training. (default: False)
  --precision PRECISION
                        Precision flag for pytorch-lightning Trainer. (default: 32)
  --total_bsz TOTAL_BSZ
                        Total batch size (use with `per_device_bsz` to automatically determine gradient accumulation steps consistent with number of nodes/devices). If None, is set to `per_device_bsz` times the total number of devices
                        across nodes. (default: None)
  --accelerator ACCELERATOR
                        Accelerator flag for pytorch-lightning Trainer. (default: gpu)
  --devices DEVICES     Number of accelerator devices or list of device ids.Must be a str that will evaluate to a python expression. E.g. "1" or "[0,1]". (default: 1)
  --num_nodes NUM_NODES
                        Number of nodes. (default: 1)
  --strategy STRATEGY   Set to `ddp` for multiple devices or nodes. (default: None)
  --fault_tolerant [FAULT_TOLERANT]
                        Whether to run pytorch-lightning in fault-tolerant mode. See: https://pytorch-lightning.readthedocs.io/en/1.6.3/advanced/fault_tolerant_training.html?highlight=fault%20tolerant and:
                        https://github.com/PyTorchLightning/pytorch-lightning/blob/1.6.3/pl_examples/fault_tolerant/automatic.py (default: True)
  --no_fault_tolerant   Whether to run pytorch-lightning in fault-tolerant mode. See: https://pytorch-lightning.readthedocs.io/en/1.6.3/advanced/fault_tolerant_training.html?highlight=fault%20tolerant and:
                        https://github.com/PyTorchLightning/pytorch-lightning/blob/1.6.3/pl_examples/fault_tolerant/automatic.py (default: False)
  --slurm_auto_requeue [SLURM_AUTO_REQUEUE]
                        Whether to run pytorch-lightning with auto-requeue. See: https://pytorch-lightning.readthedocs.io/en/1.6.3/clouds/cluster.html?highlight=slurm#wall-time-auto-resubmit (default: True)
  --no_slurm_auto_requeue
                        Whether to run pytorch-lightning with auto-requeue. See: https://pytorch-lightning.readthedocs.io/en/1.6.3/clouds/cluster.html?highlight=slurm#wall-time-auto-resubmit (default: False)
  --wandb_project WANDB_PROJECT
                        Weights and bias project to log to. (default: None)
  --wandb_entity WANDB_ENTITY
                        Weights and bias entity to log to. (default: None)
```

