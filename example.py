import torch
import yaml
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import wandb
import torchinfo
from util import warmup_cosine_scheduler_fn, get_batch, WikiTextValidator, OpenWebText2, OWT2, auto_regressive_cross_entropy, get_param_groups
from transformers import AutoTokenizer
from trainer import DDPTrainer, master_only, is_master_process, FSDPTrainer
from annmodel import CausalLM
import torch.nn.functional as F
from model import SpikeLM, Block
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
import functools

class LMTrainer(DDPTrainer):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler_fn: callable, 
                is_snn: bool,
                 **ddp_kwargs):
        super().__init__(model, optimizer, scheduler_fn, **ddp_kwargs)
        self.is_snn = is_snn
        self.best_ppl = float("inf")

    @master_only
    def prepare_test_dataset(self, tokenizer, max_length: int, stride: int):
        self.test_dataset = WikiTextValidator(tokenizer, max_length, stride)

    def forward_step(self, data) -> torch.Tensor:
        x, y = data
        logits = self.model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1).to(logits.device))
        return loss

    @master_only
    def test_and_save(self, ckpt_path):
        self.model.eval()
        ppl = self.test_dataset.validate_model_on_wikitext(self.model, self.is_snn)
        if is_master_process():
            wandb.log({"test_ppl": ppl})
        self.model.train()
        if ppl < self.best_ppl:
            self.best_ppl = ppl
            self.save_checkpoint(ckpt_path)

# 从 YAML 文件加载配置
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 加载配置
config_path = "config.yaml"
config = load_config(config_path)

if is_master_process():
    wandb.init(
        project="spikelm",
        name=config["run_name"],
        config=config
    )
    wandb.run.log_code(".")

# 加载数据集
train_dataset = OpenWebText2(
    config["datapath"], 
    config["seqlen"], 
    config["bs_size"], 
    config["num_tokens"] // config["seqlen"]
)

# 创建模型
v = SpikeLM(
    drop_rate=0.,
    drop_path_rate=0.,
    img_size_h=32, img_size_w=32,
    patch_size=config["patch_size"], 
    embed_dims=config["dim"], 
    num_heads=config["heads"], 
    mlp_ratios=config["mlp_ratio"],
    in_channels=3, 
    num_classes=-1, 
    qkv_bias=False,
    depths=config["depth"], 
    sr_ratios=1,
    T=config["T"]
)

# v = CausalLM(
#     embed_dim=config["dim"],
#     depth=config["depth"],
#     heads=config["heads"],
#     mlp_dim=config["dim"] * config["mlp_ratio"],
#     seqlen=config["seqlen"]
# )
if is_master_process():
    print("Compiling model...")
# v = torch.compile(v)
torch.set_float32_matmul_precision('high')

# 配置优化器和学习率调度器
param_groups = get_param_groups(v, weight_decay=1e-1)
optimizer = torch.optim.AdamW(param_groups, lr=config["start_lr"], betas=(0.9, 0.95), fused=True)
lr_lambda = warmup_cosine_scheduler_fn(
    config["warmup_steps"], 
    config["anneal_steps"], 
    config["start_lr"], 
    config["end_lr"]
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 设置 DDP 环境
DDPTrainer.setup_ddp()
policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
# policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e8)
trainer = LMTrainer(v, optimizer, lr_lambda, is_snn=config["is_snn"], 
                    find_unused_parameters=False)
# trainer.stat_model(input_data=torch.randint(0, 255, (1, 10)))
trainer.prepare_train_dataset(train_dataset, None)
trainer.prepare_test_dataset(tokenizer, max_length=config["seqlen"], stride=config["stride"])

# 开始训练
if is_master_process():
    print("Starting training...")
trainer.train(
    max_epoches=config["max_epoches"], 
    test_interval_steps=config["test_step"], 
    log_interval_steps=10,
    gradient_accumulation_steps=config["accumulate_steps"],
    amp_enable=True,
    grad_clip=config["grad_clip"],
    ckpt_path=f'./spikelm_{config["run_name"]}'
)
trainer.cleanup_ddp()
