import os
import time
from contextlib import redirect_stdout, redirect_stderr, contextmanager, nullcontext
from typing import Set, Callable

# PyTorch core imports
import torch
import torch.nn as nn
import torch.distributed.checkpoint as dcp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast

# PyTorch distributed training imports
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

# Third-party libraries
import torchinfo
import wandb

# Utility libraries
import functools

def is_master_process():
    return int(os.environ["LOCAL_RANK"]) == 0

# decorator that makes a function runs only by master process
def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_master_process():
            return func(*args, **kwargs)
        else:
            return None
    return wrapper

class DDPTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler_fn: callable, 
                 **ddp_kwargs):
        # setup_ddp should be run before this
        self.model = model
        self.optimizer = optimizer
        self.start_lr = optimizer.param_groups[0]['lr']

        if scheduler_fn is None:
            scheduler_fn = lambda step: 1.0
        self.scheduler_fn = scheduler_fn

        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model.cuda()
        self.model = DDP(model, device_ids=[self.gpu_id], **ddp_kwargs)

        self._reconfigure_optimizer()
        self.scaler = GradScaler()

        self.global_step = 0    # total step of optimizer. accumulated across epochs
        self.dataset_step = 0   # step of dataset. reset after each epoch
        self.epoch = 0

    def _reconfigure_optimizer(self):
        '''
        rebuild a same optimizer for wrapped model, since FSDP will change the param shape
        '''
        new_param_map = {id(p): p for p in self.model.parameters()}
        new_param_groups = []
        for group in self.optimizer.param_groups:
            new_params = [new_param_map[id(p)] for p in group['params'] if id(p) in new_param_map]
            if not new_params:
                print("warning: optimizer group has no parameters in fsdp model")
                continue
            new_group = {key: value for key, value in group.items() if key != 'params'}
            new_group['params'] = new_params
            new_param_groups.append(new_group)
        new_optimizer = torch.optim.AdamW(new_param_groups)
        self.optimizer = new_optimizer

    @staticmethod
    def setup_ddp():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend="nccl")

    @staticmethod
    def cleanup_ddp():
        destroy_process_group()

    def stat_model(self, input_data = None, input_size = None):
        self.model.eval()
        if input_data is not None:
            input_data = input_data.cuda()
        torchinfo.summary(self.model.module, input_data=input_data, input_size=input_size)

    def update_step(self):
        self.global_step += 1

    def update_lr(self):
        lr = self.scheduler_fn(self.global_step) * self.start_lr
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def save_checkpoint(self, path: str):
        """
        Saves model and optimizer state to a checkpoint file. 
        `self.global_step` should be maintained if this function is called.

        Args:
            path (str): path to the checkpoint file

        Note: This function is only called by the master process.
        """
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):    # suppress warnings from torch
            model_dict, optimizer_dict = get_state_dict(self.model, self.optimizer)
        ckpt = {
            'model_state_dict': model_dict,
            'optimizer_state_dict': optimizer_dict,
            'epoch': self.epoch,
            'dataset_step': self.dataset_step,
            'optimizer_step': self.global_step,
        }
        writer = dcp.FileSystemWriter(path, overwrite=True)
        dcp.save(ckpt, storage_writer=writer)

    def load_checkpoint(self, path: str):
        print(f"Loading checkpoint from {path}")

        # Suppress warnings from torch
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            model_dict, optimizer_dict = get_state_dict(self.model, self.optimizer)

        # Initialize placeholders
        ckpt = {
            'model_state_dict': model_dict,
            'optimizer_state_dict': optimizer_dict,
            'epoch': 0,
            'dataset_step': 0,
            'optimizer_step': 0,
        }

        # Load checkpoint using dcp
        reader = dcp.FileSystemReader(path)
        dcp.load(ckpt, storage_reader=reader)

        # Restore model and optimizer state
        set_state_dict(self.model, self.optimizer,
                    model_state_dict=ckpt['model_state_dict'],
                    optim_state_dict=ckpt['optimizer_state_dict'])

        # Update class attributes from loaded checkpoint
        self.epoch = ckpt['epoch']
        self.dataset_step = ckpt['dataset_step']
        self.global_step = ckpt['optimizer_step']  # Ensure global_step is updated correctly
        
    def prepare_train_dataset(self, train_dataset: Dataset, batch_size: int):
        """
        Prepares the training dataset by creating a data loader with a distributed sampler.
        This is the default setting for DDP. Can be overridden as you like.
        Args:
            train_dataset (Dataset): The training dataset to be loaded. Actually can be anything that is iterable.
            batch_size (int): The number of samples per batch to load.
            **loader_kwargs: Additional keyword arguments to pass to the DataLoader.

        This method sets up a DataLoader with a DistributedSampler for the training dataset 
        to ensure that data is evenly distributed across multiple processes in distributed training.
        """
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        sampler = DistributedSampler(train_dataset)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    def train(self, max_epoches: int, test_interval_steps: int, log_interval_steps: int, gradient_accumulation_steps: int, 
              amp_enable: bool, ckpt_path: str, grad_clip: float = 0.0, resume_path: str = None, resume_dataset: bool = False):
        
        # resume training
        first_resume_epoch = False
        if resume_path is not None:
            self.load_checkpoint(resume_path)
            first_resume_epoch = True
        
        amp_ctx = nullcontext()
        if amp_enable:
            scaler = self.scaler
            amp_ctx = autocast(device_type='cuda', dtype=torch.float16)
        
        for i in range(self.epoch, max_epoches):

            # train one epoch
            enumerator = enumerate(self.train_dataloader)
            local_step = 0          # optimizer step inside an epoch. used only for progress bar, not for optimizer
            self.epoch = i
            num_batches = int(len(self.train_dataloader))
            num_steps = num_batches // gradient_accumulation_steps

            if first_resume_epoch:
                first_resume_epoch = False  # keep the first epoch unchanged
                if resume_dataset:
                    for j in range(self.dataset_step):
                        next(enumerator)
                num_steps -= self.dataset_step // gradient_accumulation_steps
            else:
                self.dataset_step = 0   # step of dataset. reset after each epoch

            while True:
                try:
                    # forward and backward
                    t0 = time.time()
                    for micro_step in range(gradient_accumulation_steps):
                        self.model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                        sample_id, data = next(enumerator)
                        self.dataset_step += 1
                        with amp_ctx:
                            loss = self.forward_step(data)
                            accum_loss = loss / gradient_accumulation_steps
                        if amp_enable:
                            scaler.scale(accum_loss).backward()
                        else:
                            accum_loss.backward()

                    # gradient clipping
                    if grad_clip != 0.0:
                        if amp_enable:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                    # set the proper learning rate. 
                    # first step learning rate maybe different from initial learning rate of 
                    # optimizer, which can take large effect on performance.
                    self.update_lr()

                    # param and optimizer updating
                    if amp_enable:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.update_step()

                    # flush the gradients as soon as we can, no need for this memory anymore
                    self.optimizer.zero_grad(set_to_none=True)

                    # logging
                    elapsed = time.time() - t0
                    local_step += 1
                    if is_master_process() and self.global_step % log_interval_steps == 0:
                        wandb.log({"loss": loss.item(), 'lr': self.optimizer.param_groups[0]['lr']}, step=self.global_step)
                        print(
                            "Epoch: {}, Iteration: {}/{}, Iteration time: {:.3f} ms, Loss: {:.5f}, Learning rate: {:.3e}".format(
                                i, local_step, num_steps, elapsed * 1000, loss.item(), self.optimizer.param_groups[0]['lr']
                            )
                        )

                    # evaluation. implemented by user
                    if self.global_step % test_interval_steps == 0:
                        self.test_and_save(ckpt_path)

                except StopIteration:
                    break

    def prepare_test_dataset(self):
        raise NotImplementedError

    def forward_step(self, data) -> torch.Tensor:
        """
        This function should be overrided by subclasses. It is a part of the training loop.
        It should forward the data and return the loss.
        
        Args:
            data: a batch of data, which is retrieved from the `train_dataset` in a batched form 
        
        Returns:
            loss: the loss of this batch
        """
        raise NotImplementedError

    def test_and_save(self):
        raise NotImplementedError
    
class FSDPTrainer(DDPTrainer):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler_fn: Callable, 
                wrap_policy: Callable, cpu_offload: bool = False, **fsdp_kwargs):
        # setup_ddp should be run before this
        self.model = model
        self.optimizer = optimizer
        self.start_lr = optimizer.param_groups[0]['lr']

        if scheduler_fn is None:
            scheduler_fn = lambda step: 1.0
        self.scheduler_fn = scheduler_fn

        # distributed training setup
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model.cuda()

        if cpu_offload:
            self.model = FSDP(
                self.model,
                auto_wrap_policy=wrap_policy,
                cpu_offload=CPUOffload(offload_params=True),
                use_orig_params=True,       # FSDP will use flat parameter by default, which will corrupt the optimizer
            )
        else:
            self.model = FSDP(
                self.model,
                auto_wrap_policy=wrap_policy,
                use_orig_params=True,       # FSDP will use flat parameter by default, which will corrupt the optimizer
            )

        self._reconfigure_optimizer()
        self.scaler = ShardedGradScaler()

        self.global_step = 0    # total step of optimizer. accumulated across epochs
        self.dataset_step = 0   # step of dataset. reset after each epoch
        self.epoch = 0
