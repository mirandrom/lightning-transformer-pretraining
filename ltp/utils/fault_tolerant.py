import os
from time import sleep

import pytorch_lightning as pl

from .logging import get_logger

from typing import *

logger = get_logger(__name__)
logger.setLevel('INFO')


class TestPlFaultTolerant(pl.Callback):
    """ Test fault tolerant on any module/trainer """
    def __init__(self, fail_on_step: int):
        self.fail_on_step = fail_on_step

    def on_train_batch_start(self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if pl_module.global_step == self.fail_on_step:
            logger.warning(
                f"READY TO BE KILLED WITH SIGTERM SIGNAL. " 
                f"Run `kill -SIGTERM {os.getpid()}` in another terminal."
            )
            while not trainer._terminate_gracefully:
                sleep(0.1)


