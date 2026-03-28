from __future__ import annotations

import math
import os
import random
import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# Match original default dtype behavior
torch.set_default_tensor_type(torch.DoubleTensor)


class CfgNode(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def merge_from_dict(self, dct: dict) -> None:
        for key, val in dct.items():
            self[key] = val


def same_seeds(seed: int = 2023) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device() -> torch.device:
    env_device = os.getenv("SCPLANT_DEVICE", "").strip()
    if env_device:
        return torch.device(env_device)
    if torch.cuda.is_available():
        # Use only cuda:5 when available, otherwise fall back to CPU
        if torch.cuda.device_count() > 5:
            return torch.device("cuda:5")
        return torch.device("cpu")
    return torch.device("cpu")


def configure_cpu_threads(max_threads: int = 16) -> None:
    torch.set_num_threads(min(max_threads, torch.get_num_threads()))
    try:
        torch.set_num_interop_threads(min(max_threads, torch.get_num_interop_threads()))
    except Exception:
        pass


def enforce_thread_limits(max_threads: int = 16) -> None:
    max_threads = int(max_threads)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(max_threads)
    torch.set_num_threads(max_threads)
    try:
        torch.set_num_interop_threads(max_threads)
    except Exception:
        pass


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _get_act_name(act: Optional[str]) -> str:
    if not act:
        return "relu"
    return str(act).strip().lower()


def _get_act_fn(act: Optional[str]):
    name = _get_act_name(act)
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {act}")


def _get_act_layer(act: Optional[str]) -> nn.Module:
    name = _get_act_name(act)
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {act}")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.act = _get_act_fn(getattr(config, "act", "relu"))

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = self.act(q)
        k = self.act(k)
        v = self.act(v)

        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
        if attn_mask is not None:
            key_mask = attn_mask[:, None, None, :].to(dtype=torch.bool, device=att.device)
            att = att.masked_fill(~key_mask, float("-inf"))

        att = self.act(att)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.c_proj(y))

        if attn_mask is not None:
            y = y * attn_mask[:, :, None]
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        act_layer = _get_act_layer(getattr(config, "act", "relu"))
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 2 * config.n_embd),
                c_proj=nn.Linear(2 * config.n_embd, config.n_embd),
                act=act_layer,
                dropout=nn.Dropout(config.resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x


class scDataSet(Dataset):
    def __init__(self, data: np.ndarray, label: np.ndarray):
        self.data = torch.from_numpy(np.asarray(data))
        self.label = torch.from_numpy(np.asarray(label))
        self.length = self.data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class GPT(nn.Module):
    @staticmethod
    def get_default_config():
        c = CfgNode()
        c.model_type = "gpt"
        c.n_layer = None
        c.n_head = None
        c.n_embd = None
        c.vocab_size = None
        c.block_size = None
        c.embd_pdrop = 0.1
        c.resid_pdrop = 0.1
        c.attn_pdrop = 0.1
        c.entreg = 0.1
        c.p = 2
        c.h = 2
        c.loss1 = 50
        c.mod2_dim = 134
        c.act = "relu"
        return c

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.conf = config
        type_given = config.model_type is not None
        params_given = all(
            [config.n_layer is not None, config.n_head is not None, config.n_embd is not None]
        )
        assert type_given ^ params_given
        if type_given:
            config.merge_from_dict(
                {
                    "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),
                    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
                    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
                    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
                    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
                    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
                    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
                    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
                    "gpt-nano": dict(n_layer=1, n_head=config.h, n_embd=config.n_embd),
                }[config.model_type]
            )

        self.pro = nn.Linear(config.vocab_size, config.n_embd)
        self.transformer = nn.ModuleDict(
            dict(
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.mod2_dim)
        self.act = _get_act_fn(getattr(config, "act", "relu"))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=x.dtype, device=x.device)
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x * mask_f[:, :, None]).sum(dim=1) / denom

    def encode(
        self,
        idx: torch.Tensor,
        pos_idx: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if idx.dtype != torch.double:
            idx = idx.double()
        b, t, _ = idx.size()
        device = idx.device
        if pos_idx is None:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        else:
            pos = pos_idx
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(idx + pos_emb)
        for block in self.transformer.h:
            x = self.act(x)
            x = block(x, attn_mask=attn_mask)
        x = self.act(x)
        x = self.transformer.ln_f(x)
        x = self.act(x)
        if attn_mask is None:
            x = torch.mean(x, dim=1)
        else:
            x = self._masked_mean(x, attn_mask)
        emb = x
        logits = self.lm_head(x)
        mod_logits = self.act(logits)
        return emb, mod_logits

    def cross_mod(self, mod1, mod2=None):
        if isinstance(mod1, torch.Tensor):
            idx = mod1
            if idx.dtype != torch.double:
                idx = idx.double()
        else:
            idx = torch.tensor(mod1, dtype=torch.double)
        device = idx.device
        b, t, _ = idx.size()
        num_cls = int(self.conf.block_size / 2)
        cls1 = torch.zeros(num_cls).long().to(device)
        cls2 = torch.ones(num_cls).long().to(device)
        cls = torch.cat((cls1, cls2), dim=0).to(device)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        pos_emb = self.transformer.wpe(pos)
        cls_emb = self.transformer.wpe(cls)
        _ = cls_emb

        x = self.transformer.drop(idx + pos_emb)
        for block in self.transformer.h:
            x = self.act(x)
            x = block(x)
        x = self.act(x)
        x = self.transformer.ln_f(x)
        x = self.act(x)
        x = torch.mean(x, dim=1)
        emb = x
        logits = self.lm_head(x)
        mod_logits = self.act(logits)

        loss = None
        if mod2 is not None:
            if isinstance(mod2, torch.Tensor):
                targets = mod2
                if targets.dtype != torch.double:
                    targets = targets.double()
            else:
                targets = torch.tensor(mod2, dtype=torch.double, device=device)
            loss1 = F.mse_loss(mod_logits, targets) ** 0.5
            loss = loss1
        return loss, emb, mod_logits

    def forward(self, X, Y):
        loss1, emb_mod, mod1_logits2 = self.cross_mod(X, Y)
        loss = self.conf.loss1 * loss1
        return emb_mod, loss, mod1_logits2

    def masked_rmse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask_f = mask.to(dtype=pred.dtype, device=pred.device)
        # diff = (pred - target) * mask_f
        # denom = mask_f.sum().clamp(min=1.0)
        # mse = diff.pow(2).sum() / denom
        target  = target.to(dtype=pred.dtype, device=pred.device)
        loss1 = F.mse_loss(pred, target) ** 0.5

        return loss1 #torch.sqrt(mse + 1e-12)

    def forward_mae(
        self,
        x_vis: torch.Tensor,
        pos_idx: torch.Tensor,
        attn_mask: torch.Tensor,
        targets: torch.Tensor,
        gene_mask: torch.Tensor,
    ):
        emb, pred = self.encode(x_vis, pos_idx=pos_idx, attn_mask=attn_mask)
        loss1 = self.masked_rmse(pred, targets, gene_mask)
        loss = self.conf.loss1 * loss1
        return emb, loss, pred

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas,
        )
        return optimizer


class Trainer:
    @staticmethod
    def get_default_config():
        c = CfgNode()
        c.device = "auto"
        c.num_workers = 8
        c.epoch = 100
        c.batch_size = 64
        c.learning_rate = 3e-4
        c.betas = (0.95, 0.99)
        c.weight_decay = 0.1
        c.grad_norm_clip = 1.0
        c.log_interval = 50
        return c

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        if config.device == "auto":
            self.device = select_device()
        else:
            self.device = torch.device(config.device)
        if self.device.type == "cpu":
            configure_cpu_threads(16)
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def run(self):
        model, config = self.model, self.config
        model = model.to(self.device)
        optim_model = model.module if hasattr(model, "module") else model
        self.optimizer = optim_model.configure_optimizers(config)
        if self._resume_optimizer_state is not None:
            self.optimizer.load_state_dict(self._resume_optimizer_state)
        if self._resume_optimizer_state is not None:
            self.optimizer.load_state_dict(self._resume_optimizer_state)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )

        model.train()
        n_epochs = config.epoch
        total_steps = max(0, n_epochs - self.start_epoch) * len(train_loader)
        log_interval = max(1, int(getattr(config, "log_interval", 50)))
        global_step = 0
        start_time = time.perf_counter()
        for epoch in range(self.start_epoch, n_epochs):
            train_loss = []
            emb_mods = []
            epoch_start = time.perf_counter()
            for batch_idx, batch in enumerate(train_loader, 1):
                X, Y = batch
                X = X.to(self.device)
                Y = Y.to(self.device)
                emb_mod, self.loss, _ = model(X, Y)

                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
                train_loss.append(self.loss.item())

                if epoch == n_epochs - 1:
                    if self.device.type == "cuda":
                        emb_mods.extend(emb_mod.cpu().detach().numpy())
                    else:
                        emb_mods.extend(emb_mod.detach().numpy())

                global_step += 1
                if global_step % log_interval == 0 or batch_idx == len(train_loader):
                    elapsed = time.perf_counter() - start_time
                    avg_step = elapsed / max(1, global_step)
                    eta = avg_step * (total_steps - global_step)
                    print(
                        "[ Train | step {:06d}/{:06d} ] epoch={:03d}/{:03d} "
                        "loss={:.5f} elapsed={} eta={}".format(
                            global_step,
                            total_steps,
                            epoch + 1,
                            n_epochs,
                            self.loss.item(),
                            format_duration(elapsed),
                            format_duration(eta),
                        )
                    )

            train_loss = sum(train_loss) / len(train_loss)
            epoch_time = time.perf_counter() - epoch_start
            total_elapsed = time.perf_counter() - start_time
            avg_epoch = total_elapsed / (epoch + 1)
            eta_epochs = avg_epoch * (n_epochs - epoch - 1)
            print(
                "[ Train | {:03d}/{:03d} ] loss={:.5f} epoch_time={} elapsed={} eta={}".format(
                    epoch + 1,
                    n_epochs,
                    train_loss,
                    format_duration(epoch_time),
                    format_duration(total_elapsed),
                    format_duration(eta_epochs),
                )
            )

        emb_mods = np.asarray(emb_mods)
        return emb_mods

    def load_checkpoint(self, ckpt_path: str) -> int:
        if not ckpt_path:
            return 0
        state = torch.load(ckpt_path, map_location="cpu")
        model_state = state.get("model_state_dict", state)
        load_target = self.model.module if hasattr(self.model, "module") else self.model
        load_target.load_state_dict(model_state, strict=True)
        self._resume_optimizer_state = state.get("optimizer_state_dict")
        self.start_epoch = int(state.get("epoch", 0))
        return self.start_epoch


class TrainerMAE:
    @staticmethod
    def get_default_config():
        c = CfgNode()
        c.device = "auto"
        c.num_workers = 8
        c.epoch = 100
        c.batch_size = 64
        c.learning_rate = 3e-4
        c.betas = (0.95, 0.99)
        c.weight_decay = 0.1
        c.grad_norm_clip = 1.0
        c.log_interval = 50
        c.mask_ratio = 0.75
        c.checkpoint_interval = 0
        c.checkpoint_dir = ""
        c.start_epoch = 0
        c.lr_schedule = "none"
        c.warmup_steps = 0
        return c

    def __init__(
        self,
        config,
        model,
        train_dataset,
        patch_gene_index: torch.Tensor,
        train_sampler=None,
        is_distributed: bool = False,
        rank: int = 0,
    ):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.train_sampler = train_sampler
        self.is_distributed = is_distributed
        self.rank = rank
        self.callbacks = defaultdict(list)
        if config.device == "auto":
            self.device = select_device()
        else:
            self.device = torch.device(config.device)
        if self.device.type == "cpu":
            configure_cpu_threads(16)
        self.model = self.model.to(self.device)
        if patch_gene_index.dtype != torch.long:
            patch_gene_index = patch_gene_index.long()
        self.patch_gene_index = patch_gene_index.to(self.device)
        print("running on device", self.device)

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.checkpoint_interval = int(getattr(config, "checkpoint_interval", 0) or 0)
        self.checkpoint_dir = getattr(config, "checkpoint_dir", "") or ""
        self.start_epoch = int(getattr(config, "start_epoch", 0) or 0)
        self._resume_optimizer_state = None

    def _sample_mask(self, b: int, n_patches: int, device: torch.device, ratio: float) -> torch.Tensor:
        mask = torch.rand((b, n_patches), device=device) < ratio
        # ensure at least one visible and one masked
        all_masked = mask.all(dim=1)
        if all_masked.any():
            idx = torch.nonzero(all_masked, as_tuple=False).squeeze(1)
            for i in idx.tolist():
                j = torch.randint(0, n_patches, (1,), device=device).item()
                mask[i, j] = False
        none_masked = (~mask).all(dim=1)
        if none_masked.any():
            idx = torch.nonzero(none_masked, as_tuple=False).squeeze(1)
            for i in idx.tolist():
                j = torch.randint(0, n_patches, (1,), device=device).item()
                mask[i, j] = True
        return mask

    def _build_visible_inputs(self, x: torch.Tensor, visible_mask: torch.Tensor):
        b, t, c = x.shape
        lengths = visible_mask.sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() else 1
        if max_len < 1:
            max_len = 1
        sort_key = (~visible_mask).to(torch.int64)
        sort_idx = torch.argsort(sort_key, dim=1, stable=True)
        pos_vis = sort_idx[:, :max_len]
        x_vis = torch.gather(x, 1, pos_vis.unsqueeze(-1).expand(-1, -1, c))
        attn_mask = torch.arange(max_len, device=x.device)[None, :] < lengths[:, None]
        return x_vis, pos_vis, attn_mask

    def _build_gene_mask(self, mask: torch.Tensor, n_genes: int) -> torch.Tensor:
        b, _ = mask.shape
        gene_mask = torch.zeros((b, n_genes), dtype=torch.bool, device=mask.device)
        if mask.any():
            masked_idx = torch.nonzero(mask, as_tuple=False)
            rows = masked_idx[:, 0]
            patch_idx = masked_idx[:, 1]
            gene_idx = self.patch_gene_index[patch_idx]
            cols = gene_idx.reshape(-1)
            rows = rows.repeat_interleave(gene_idx.shape[1])
            gene_mask[rows, cols] = True
        return gene_mask

    def run(self):
        model, config = self.model, self.config
        model = model.to(self.device)
        optim_model = model.module if hasattr(model, "module") else model
        self.optimizer = optim_model.configure_optimizers(config)
        if self._resume_optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(self._resume_optimizer_state)
                print("[Info] Loaded optimizer state from checkpoint.")
            except Exception as exc:
                print(f"[Warn] Failed to load optimizer state: {exc}")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )

        model.train()
        n_epochs = config.epoch
        total_steps = max(0, n_epochs - self.start_epoch) * len(train_loader)
        schedule = str(getattr(config, "lr_schedule", "none") or "none").lower()
        warmup_steps = int(getattr(config, "warmup_steps", 0) or 0)
        if schedule != "none" and warmup_steps <= 0 and total_steps > 0:
            warmup_steps = max(1, int(0.01 * total_steps))

        def _compute_lr(step: int) -> float:
            base_lr = float(config.learning_rate)
            if schedule == "none" or total_steps <= 0:
                return base_lr
            step = max(1, int(step))
            if warmup_steps > 0 and step <= warmup_steps:
                return base_lr * step / warmup_steps
            if schedule == "cosine":
                denom = max(1, total_steps - warmup_steps)
                progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
                return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            return base_lr
        log_interval = max(1, int(getattr(config, "log_interval", 50)))
        global_step = 0
        start_time = time.perf_counter()
        for epoch in range(self.start_epoch, n_epochs):
            if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(epoch)
            train_loss = []
            emb_mods = []
            epoch_start = time.perf_counter()
            for batch_idx, batch in enumerate(train_loader, 1):
                X, Y = batch
                X = X.to(self.device)
                Y = Y.to(self.device)
                bsz, n_patches, _ = X.shape

                mask = self._sample_mask(bsz, n_patches, self.device, config.mask_ratio)
                visible_mask = ~mask
                x_vis, pos_vis, attn_mask = self._build_visible_inputs(X, visible_mask)
                gene_mask = self._build_gene_mask(mask, Y.shape[1])

                forward_model = model.module if hasattr(model, "module") else model
                emb_mod, self.loss, _ = forward_model.forward_mae(
                    x_vis=x_vis,
                    pos_idx=pos_vis,
                    attn_mask=attn_mask,
                    targets=Y,
                    gene_mask=gene_mask,
                )

                if schedule != "none":
                    lr = _compute_lr(global_step + 1)
                    for group in self.optimizer.param_groups:
                        group["lr"] = lr
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
                train_loss.append(self.loss.item())

                if epoch == n_epochs - 1:
                    if self.device.type == "cuda":
                        emb_mods.extend(emb_mod.cpu().detach().numpy())
                    else:
                        emb_mods.extend(emb_mod.detach().numpy())

                global_step += 1
                if global_step % log_interval == 0 or batch_idx == len(train_loader):
                    elapsed = time.perf_counter() - start_time
                    avg_step = elapsed / max(1, global_step)
                    eta = avg_step * (total_steps - global_step)
                    print(
                        "[ Train | step {:06d}/{:06d} ] epoch={:03d}/{:03d} "
                        "loss={:.5f} elapsed={} eta={}".format(
                            global_step,
                            total_steps,
                            epoch + 1,
                            n_epochs,
                            self.loss.item(),
                            format_duration(elapsed),
                            format_duration(eta),
                        )
                    )

            train_loss = sum(train_loss) / len(train_loss)
            epoch_time = time.perf_counter() - epoch_start
            total_elapsed = time.perf_counter() - start_time
            avg_epoch = total_elapsed / (epoch + 1)
            eta_epochs = avg_epoch * (n_epochs - epoch - 1)
            print(
                "[ Train | {:03d}/{:03d} ] loss={:.5f} epoch_time={} elapsed={} eta={}".format(
                    epoch + 1,
                    n_epochs,
                    train_loss,
                    format_duration(epoch_time),
                    format_duration(total_elapsed),
                    format_duration(eta_epochs),
                )
            )

            if (
                self.checkpoint_interval > 0
                and (epoch + 1) % self.checkpoint_interval == 0
                and (not self.is_distributed or self.rank == 0)
                and self.checkpoint_dir
            ):
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                ckpt_name = f"ckpt_epoch_{epoch + 1:03d}.pth"
                ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
                state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": state,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                    },
                    ckpt_path,
                )

        emb_mods = np.asarray(emb_mods)
        return emb_mods

    def load_checkpoint(self, ckpt_path: str) -> int:
        if not ckpt_path:
            return 0
        state = torch.load(ckpt_path, map_location="cpu")
        model_state = state.get("model_state_dict", state)
        load_target = self.model.module if hasattr(self.model, "module") else self.model
        load_target.load_state_dict(model_state, strict=True)
        self._resume_optimizer_state = state.get("optimizer_state_dict")
        self.start_epoch = int(state.get("epoch", 0))
        return self.start_epoch
