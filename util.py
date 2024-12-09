import torch
import torch.nn as nn
import copy
from functools import partial
import math
import numpy as np
import os
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
from spikingjelly.clock_driven import functional
from torch.utils.data import Dataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from trainer import DDPTrainer, master_only, is_master_process

def transform_model(model: nn.Module, mapping: dict, inplace: bool = False) -> nn.Module:
    '''
    Transform a model by replacing certain modules according to a given mapping.
    The target module class should have a method `from_float(cls, module)` that converts
    source module to target module

    Args:
        model: the model to be transformed
        mapping: a dictionary that maps from source module type to target module type
        inplace: whether to modify the original model or make a copy

    Returns:
        the transformed model

    Example:
        ```
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
        mapping = {
            nn.Conv2d: QuantConv2d
        }
        quant_model = transform_model(model, mapping, inplace=False)
        ```
    '''
    if not inplace:
        model = copy.deepcopy(model)
    return _recursive_replace(model, mapping)
    

def _recursive_replace(model: nn.Module, mapping: dict) -> nn.Module:
    for name, module in model.named_children():
        for module_class in mapping.keys():
            if isinstance(module, module_class):
                new_module = mapping[module_class].from_float(module)
                setattr(model, name, new_module)
            else:
                _recursive_replace(module, mapping)
    return model

def module_mapping_call(model: nn.Module, function: callable, silent = False) -> None:
    """
    Applies a given function to all modules within the model.

    Args:
        model: The neural network model containing modules.
        function: A callable function to be applied to each module.
        silent: if `False`, a warning will be printed when the function fails to be applied to a module.

    Note:
        If the function raises an exception when applied to a module,
        a warning will be printed and the module will be skipped.
        If the warning is not expected, use `module_mapping_silent_call`
    """
    for name, module in model.named_modules():
        try:
            function(module)
        except Exception as e:
            if not silent:
                print(f'Warning: function mapping failed on module {name}. Skipped. Exception: {e}')

module_mapping_silent_call = partial(module_mapping_call, silent=True)

def linear_lr_fn(start_lr, end_lr, num_steps):
    """
    Creates a learning rate scheduler that linearly decreases the learning rate
    from start_lr to end_lr over num_steps steps.

    Args:
        start_lr (float): The learning rate at step 0.
        end_lr (float): The learning rate at the final step.
        num_steps (int): The number of steps.

    Returns:
        A function that can be used as a lambda scheduler in Pytorch.
    """
    return lambda step: max(end_lr / start_lr, 1.0 - step / num_steps + (end_lr / start_lr) * (step / num_steps))

def warmup_cosine_scheduler_fn(warmup_step, anneal_step, max_lr, min_lr):
    """
    构造一个 Warmup + Cosine Annealing 调度函数。
    
    Args:
        warmup_step (int): 预热阶段的步数。
        anneal_step (int): 余弦退火阶段的步数。
        max_lr (float): 最大学习率。
        min_lr (float): 最小学习率。
    
    Returns:
        callable: 一个函数，可直接用于 PyTorch 的 LambdaLR。
    """
    def lr_lambda(current_step):
        if current_step < warmup_step:
            # Warmup phase: 线性增长
            return current_step / warmup_step
        elif current_step < warmup_step + anneal_step:
            # Cosine Annealing phase
            progress = (current_step - warmup_step) / anneal_step
            return min_lr / max_lr + (1 - min_lr / max_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            # 训练步数超过计划，保持最小学习率
            return min_lr / max_lr
    
    return lr_lambda

def get_batch(data_dir, seqlen, batch_size):
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - seqlen, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seqlen]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seqlen]).astype(np.int64)) for i in ix])
    return x, y

class OpenWebText2(Dataset):
    def __init__(self, datapath, seqlen, bs_size, num_samples):
        self.datapath = datapath
        self.seqlen = seqlen
        self.bs_size = bs_size
        self.num_samples = int(num_samples)  # 控制总共可以生成多少批次

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # index 在这里是无用的，因为我们随机采样
        x, y = get_batch(self.datapath, self.seqlen, self.bs_size)
        return x, y
    
class OWT2:
    def __init__(self, datapath, seqlen, bs_size, num_samples):
        self.datapath = datapath
        self.seqlen = seqlen
        self.bs_size = bs_size
        self.num_samples = int(num_samples)  # 控制总共可以生成多少批次

    def get_batch(self):
        x, y = get_batch(self.datapath, self.seqlen, self.bs_size)
        return x, y

def posemb_sincos(seqlen, dim, temperature: int = 10000, dtype=torch.float32):
    """
    Compute sinusoidal positional embeddings as used in GPT.

    Args:
        seqlen (int): Sequence length.
        dim (int): Dimensionality of the positional encoding.
        temperature (int): Scaling factor for positional encodings. Default is 10000.
        dtype (torch.dtype): Data type of the returned tensor. Default is torch.float32.

    Returns:
        torch.Tensor: A [seqlen, dim] positional encoding matrix.
    """
    assert dim % 2 == 0, "Dimension should be even for sinusoidal positional encoding."
    
    # Create position indices [0, 1, ..., seqlen-1]
    position = torch.arange(seqlen, dtype=dtype).unsqueeze(1)  # Shape: [seqlen, 1]
    
    # Create frequency indices [0, 1, ..., dim/2-1]
    div_term = torch.exp(
        -torch.arange(0, dim, 2, dtype=dtype) * (math.log(temperature) / dim)
    )  # Shape: [dim/2]
    
    # Compute sin and cos embeddings
    pos_embed = torch.zeros((seqlen, dim), dtype=dtype)
    pos_embed[:, 0::2] = torch.sin(position * div_term)  # Sin for even indices
    pos_embed[:, 1::2] = torch.cos(position * div_term)  # Cos for odd indices
    
    return pos_embed

def auto_regressive_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute auto-regressive cross entropy loss.

    The input logits and labels are expected to be of shape (batch_size, sequence_length, vocab_size).
    The loss is computed as the cross entropy between the input logits and labels, with the tokens shifted so that
    tokens < n predict n.

    Args:
        logits (torch.Tensor): The input logits.
        labels (torch.Tensor): The input labels. Should not be shifted, it will be shifted automatically.

    Returns:
        torch.Tensor: The loss tensor.
    """
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    shift_logits = shift_logits.flatten(0, 1)
    shift_labels = shift_labels.flatten(0, 1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = F.cross_entropy(shift_logits, shift_labels)
    return loss

class WikiTextValidator:
    """
    A validator for the WikiText dataset.

    This class is responsible for loading the WikiText test set, preprocessing the data,
    and validating a given model on the test set.

    Attributes:
        tokenizer (Tokenizer): The tokenizer to use to preprocess the wikitext test set.
        max_length (int): The context length.
        stride (int): The window stride for sliding window ppl calculation.
        encodings (Dict[str, torch.Tensor]): The preprocessed wikitext test set.
    """
    def __init__(self, tokenizer, max_length: int, stride: int):
        test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
        self.encodings = encodings
        self.max_length = max_length
        self.stride = stride

    def validate_model_on_wikitext(self, model: nn.Module, is_snn: bool) -> float:
        """
        Validate the model on wikitext test set.

        Args:
            model (nn.Module): The model to validate. The model should return the logits in shape [batch_size, seq_len, vocab_size].
            is_snn (bool): Whether the model is a spiking neural network.

        Returns:
            float: The perplexity of the model on the test set.
        """
        model.eval()
        seq_len = self.encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        pbar = tqdm(range(0, seq_len, self.stride))
        for begin_loc in pbar:
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100  # no influence if trg_len == max_length

            with torch.no_grad():
                if is_snn:
                    functional.reset_net(model)
                outputs = model(input_ids)
                loss = auto_regressive_cross_entropy(outputs, target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = loss

            nlls.append(neg_log_likelihood)
            pbar.set_postfix({"ppl": torch.exp(torch.stack(nlls).mean()).item()})
            # print(f'{torch.exp(torch.stack(nlls).mean())}')

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()
    
def get_param_groups(model, weight_decay: float = 0.0):
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    return optim_groups
