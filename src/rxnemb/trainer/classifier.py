import datetime
import logging
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

from rxnemb.core.data import MultiRXNDataset, PairDataset, get_idx_split, pair_collate_fn
from rxnemb.core.model import RXNGraphormer
from rxnemb.core.utils import align_config, update_dict_key

from .scheduler import NoamLR, get_linear_scheduler_with_warmup
from .utils import get_lr, get_model_state_dict, grad_norm, param_norm, set_seed, setup_logger


class ClassifierTrainer:
    def __init__(self, config):
        self.config = config
        self.multi_gpu = bool(self.config.others.multi_gpu)
        set_seed(self.config.data.seed)

        self.model = self._build_model()
        self._setup_device()
        self._init_parameters()
        self._setup_save_dirs()
        self._setup_logging()
        self.init_optimizer()
        self.init_scheduler()
        self.loss_func = self._build_loss()
        self._build_datasets()
        self._build_dataloaders()
        self._load_pretrained_if_needed()
        self._finalize_output_dirs()

    def _build_model(self):
        input_param = {
            "emb_dim": self.config.model.emb_dim,
            "gnn_type": self.config.model.gnn_type,
            "gnn_aggr": self.config.model.gnn_aggr,
            "gnum_layer": self.config.model.gnn_num_layer,
            "node_readout": self.config.model.node_readout,
            "num_heads": self.config.model.num_heads,
            "JK": self.config.model.gnn_jk,
            "graph_pooling": self.config.model.graph_pooling,
            "tnum_layer": self.config.model.trans_num_layer,
            "trans_readout": self.config.model.trans_readout,
            "onum_layer": self.config.model.output_num_layer,
            "drop_ratio": self.config.model.drop_ratio,
            "output_size": 2,
            "split_process": True,
            "split_merge_method": self.config.model.split_merge_method,
            "output_act_func": self.config.model.output_act_func,
        }
        return RXNGraphormer("classification", align_config(input_param, "classifier")).get_model()

    def _setup_device(self):
        if self.multi_gpu:
            self.local_rank = int(self.config.others.local_rank)
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl")
            self.model.to(self.local_rank)
            self.device_num = dist.get_world_size()
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.model.graph_pooling == "attentionxl",
            )
        else:
            device_name = self.config.others.device if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device_name)
            self.model.to(self.device)

    def _init_parameters(self):
        for parameter in self.model.parameters():
            if parameter.dim() > 1 and parameter.requires_grad:
                xavier_uniform_(parameter)

    def _setup_save_dirs(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = self.config.data.tag
        suffix = (
            "_ft"
            if getattr(self.config.model, "pretrained_model", "") and getattr(self.config.model, "fine_tune", False)
            else ""
        )
        self.save_dir = Path(self.config.model.save_dir) / f"{timestamp}-classifier-split-{tag}{suffix}"

    def _setup_logging(self):
        if self.multi_gpu:
            if dist.get_rank() != 0:
                return
            setup_logger(self.save_dir / "log")
        else:
            setup_logger(self.save_dir / "log")

        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        logging.info(str(self.config))
        logging.info(f"[INFO] Model parameters: {int(total_params / 1024 / 1024)} M")
        if self.multi_gpu:
            logging.info(f"[INFO] World size: {self.device_num}")

    def _build_loss(self):
        loss_name = self._get_loss_name()
        if loss_name != "ce":
            raise NotImplementedError(f"Loss function {loss_name} is not implemented yet.")

        weight = self._get_loss_weight()
        if weight is None:
            return torch.nn.CrossEntropyLoss(reduction="mean")

        target_device = torch.device("cuda", self.local_rank) if self.multi_gpu else self.device
        return torch.nn.CrossEntropyLoss(weight=torch.tensor(weight, device=target_device), reduction="mean")

    def _get_loss_name(self):
        loss_cfg = self.config.training.loss
        if isinstance(loss_cfg, str):
            return loss_cfg.lower()
        if hasattr(loss_cfg, "name"):
            return str(loss_cfg.name).lower()
        if hasattr(loss_cfg, "type"):
            return str(loss_cfg.type).lower()
        return str(loss_cfg).lower()

    def _get_loss_weight(self):
        loss_cfg = self.config.training.loss
        if isinstance(loss_cfg, str):
            return None
        if not hasattr(loss_cfg, "weight") or loss_cfg.weight is False:
            return None
        return loss_cfg.weight

    def _build_datasets(self):
        self._log_dataset_paths()
        if self.config.data.rct_name_regrex:
            self._build_split_from_single_pattern()
        else:
            self._build_split_from_explicit_patterns()

        self.train_dataset = PairDataset(self.train_rct_dataset, self.train_pdt_dataset)
        self.valid_dataset = PairDataset(self.valid_rct_dataset, self.valid_pdt_dataset)
        self.test_dataset = PairDataset(self.test_rct_dataset, self.test_pdt_dataset)

    def _log_dataset_paths(self):
        data_path = Path(self.config.data.data_path)
        if self.multi_gpu and dist.get_rank() != 0:
            return
        if self.config.data.rct_name_regrex:
            logging.info(
                f"[INFO] Load reactant dataset {data_path / self.config.data.rct_name_regrex}, "
                f"file trunck {self.config.data.file_num_trunck}, data trunck {self.config.data.data_trunck}..."
            )
            logging.info(
                f"[INFO] Load product dataset {data_path / self.config.data.pdt_name_regrex}, "
                f"file trunck {self.config.data.file_num_trunck}, data trunck {self.config.data.data_trunck}..."
            )

    def _build_split_from_single_pattern(self):
        self.rct_dataset = self._build_dataset(self.config.data.rct_name_regrex, name_tag="rct")
        self.pdt_dataset = self._build_dataset(self.config.data.pdt_name_regrex, name_tag="pdt")
        assert len(self.rct_dataset) == len(self.pdt_dataset), "The number of reactant and product data are not equal."

        split_ids_map = get_idx_split(
            len(self.rct_dataset),
            int(self.config.data.train_ratio * len(self.rct_dataset)),
            int(self.config.data.valid_ratio * len(self.rct_dataset)),
            self.config.data.seed,
        )

        self.train_rct_dataset = self.rct_dataset[split_ids_map["train"]]
        self.valid_rct_dataset = self.rct_dataset[split_ids_map["valid"]]
        self.train_pdt_dataset = self.pdt_dataset[split_ids_map["train"]]
        self.valid_pdt_dataset = self.pdt_dataset[split_ids_map["valid"]]
        self.test_rct_dataset = self.rct_dataset[split_ids_map["test"]]
        self.test_pdt_dataset = self.pdt_dataset[split_ids_map["test"]]

    def _build_split_from_explicit_patterns(self):
        self.train_rct_dataset = self._build_dataset(self.config.data.train_rct_name_regrex, name_tag="rct")
        self.train_pdt_dataset = self._build_dataset(self.config.data.train_pdt_name_regrex, name_tag="pdt")
        self.valid_rct_dataset = self._build_dataset(self.config.data.valid_rct_name_regrex, name_tag="rct")
        self.valid_pdt_dataset = self._build_dataset(self.config.data.valid_pdt_name_regrex, name_tag="pdt")
        self.test_rct_dataset = self._build_dataset(self.config.data.test_rct_name_regrex, name_tag="rct")
        self.test_pdt_dataset = self._build_dataset(self.config.data.test_pdt_name_regrex, name_tag="pdt")

    def _build_dataset(self, name_regrex, name_tag):
        return MultiRXNDataset(
            root=self.config.data.data_path,
            name_regrex=name_regrex,
            trunck=self.config.data.data_trunck,
            task="classification",
            file_num_trunck=self.config.data.file_num_trunck,
            name_tag=name_tag,
        )

    def _build_dataloaders(self):
        if self.multi_gpu:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=False)
            test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset, shuffle=False)
            batch_size = self.config.data.batch_size // self.device_num
            self.train_dataloader = DataLoader(
                self.train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
            )
            self.test_dataloader = DataLoader(
                self.test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0
            )
            return

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            collate_fn=pair_collate_fn,
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            collate_fn=pair_collate_fn,
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            collate_fn=pair_collate_fn,
        )

    def _load_pretrained_if_needed(self):
        pretrained_model = getattr(self.config.model, "pretrained_model", "")
        if not pretrained_model:
            return

        checkpoint_path = Path(pretrained_model) / "model" / "valid_checkpoint.pt"
        pretrained_info = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state_dict = update_dict_key(pretrained_info["model_state_dict"])
        if self.multi_gpu:
            self.model.module.load_state_dict(model_state_dict)
            self.model.to(self.local_rank)
        else:
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)

        if getattr(self.config.model, "fine_tune", False):
            self._log_rank0("Fine-tune setup!")
            self.fine_tune()

    def _finalize_output_dirs(self):
        self._log_rank0(f"[INFO] Training results will be saved in {self.save_dir}")
        self.log_dir = self.save_dir / "log"
        self.model_save_dir = self.save_dir / "model"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        is_rank0 = not self.multi_gpu or dist.get_rank() == 0
        self.writer = self._create_writer() if is_rank0 else None
        if is_rank0:
            self.config.to_json(filename=str(self.save_dir / "parameters.json"))

    def _log_rank0(self, message):
        if not self.multi_gpu or dist.get_rank() == 0:
            logging.info(message)

    def _create_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:
            logging.warning("TensorBoard is not installed. Training will continue without TensorBoard logs.")

            class _NullSummaryWriter:
                def add_scalar(self, *args, **kwargs):
                    return None

                def close(self):
                    return None

            return _NullSummaryWriter()
        return SummaryWriter(log_dir=str(self.log_dir))

    def init_optimizer(self):
        optimizer_name = self.config.optimizer.optimizer.lower()
        parameters = filter(lambda parameter: parameter.requires_grad, self.model.parameters())
        if optimizer_name == "adam":
            self.optimizer = Adam(
                parameters,
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
            )
        elif optimizer_name == "adamw":
            self.optimizer = AdamW(
                parameters,
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unsupport optimizer: '{optimizer_name}'")

    def init_scheduler(self):
        scheduler_name = self.config.scheduler.type.lower()
        if scheduler_name == "steplr":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.scheduler.lr_decay_step_size,
                gamma=self.config.scheduler.lr_decay_factor,
            )
        elif scheduler_name == "warmup":
            self.scheduler = get_linear_scheduler_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.scheduler.warmup_step,
                num_training_steps=self.config.training.epoch,
            )
        elif scheduler_name == "noamlr":
            self.scheduler = NoamLR(
                self.optimizer,
                model_size=self.config.model.emb_dim,
                warmup_steps=self.config.scheduler.warmup_step,
            )
        else:
            raise ValueError(f"Unsupport scheduler: '{scheduler_name}'")

    def fine_tune(self):
        if self.config.model.trainable != "decoder":
            raise ValueError("trainable should be in ['decoder']")

        model = self.model.module if self.multi_gpu else self.model
        trainable_params_id = {id(parameter) for parameter in model.decoder.parameters()}
        for parameter in model.parameters():
            if id(parameter) not in trainable_params_id:
                parameter.requires_grad = False

        self.init_optimizer()
        self.init_scheduler()

    def _move_batch_to_device(self, batch_data):
        rct_data, pdt_data = batch_data
        if self.multi_gpu:
            return rct_data.to(self.local_rank), pdt_data.to(self.local_rank)
        return rct_data.to(self.device), pdt_data.to(self.device)

    def train(self):
        self.model.train()
        loss_accum = 0.0
        acc_lst = []
        g_norm = float("nan")

        if self.multi_gpu:
            self.train_dataloader.sampler.set_epoch(self.epoch)

        self.optimizer.zero_grad(set_to_none=True)
        total_steps = len(self.train_dataloader)

        for step, batch_data in enumerate(self.train_dataloader, start=1):
            rct_data, pdt_data = self._move_batch_to_device(batch_data)
            out = self.model([rct_data, pdt_data])
            loss = self.loss_func(out, rct_data.y)
            acc_lst.append((out.argmax(dim=1) == rct_data.y).float().mean().cpu())
            loss.backward()
            loss_accum += loss.detach().cpu().item()

            should_step = step % self.config.training.accum == 0 or step == total_steps
            if should_step:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_norm)
                self.optimizer.step()
                if self.config.scheduler.type.lower() == "noamlr":
                    self.scheduler.step()
                g_norm = grad_norm(self.model)
                self.optimizer.zero_grad(set_to_none=True)

            if step % self.config.training.log_iter_step == 0:
                p_norm = param_norm(self.model)
                lr_cur = get_lr(self.optimizer)
                self._log_rank0(
                    f"Training step {step}, gradient norm: {g_norm:.8f}, parameters norm: {p_norm:.8f}, "
                    f"lr: {lr_cur}, loss: {loss_accum / step:.4f}, acc: {np.mean(acc_lst):.4f}"
                )

        return loss_accum / total_steps, float(np.mean(acc_lst))

    def val(self, dataloader):
        self.model.eval()
        preds = []
        targets = []
        loss_accum = 0.0

        if self.multi_gpu:
            dataloader.sampler.set_epoch(self.epoch)

        with torch.no_grad():
            for step, batch_data in enumerate(dataloader, start=1):
                rct_data, pdt_data = self._move_batch_to_device(batch_data)
                out = self.model([rct_data, pdt_data])
                loss_accum += self.loss_func(out, rct_data.y).detach().cpu().item()
                preds.append(torch.argmax(out, dim=1).detach().cpu())
                targets.append(rct_data.y.detach().cpu())

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        accuracy = (preds == targets).float().mean().item()
        return accuracy, loss_accum / len(dataloader)

    def _save_checkpoint(self, best_value):
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": get_model_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_valid_mae": best_value,
        }
        torch.save(checkpoint, self.model_save_dir / "valid_checkpoint.pt")

    def run(self):
        best_valid = -float("inf")
        best_test = -float("inf")
        lowest_valid_loss = float("inf")

        self.model.zero_grad()
        for self.epoch in range(1, self.config.training.epoch + 1):
            self._log_rank0(f"============= Epoch {self.epoch} =============")
            self._log_rank0("Training...")
            train_loss, train_acc = self.train()

            self._log_rank0("Evaluating...")
            valid_acc, valid_loss = self.val(self.valid_dataloader)
            has_test = hasattr(self.config.data, "test_rct_name_regrex") and bool(self.config.data.test_rct_name_regrex)
            if has_test:
                test_acc, test_loss = self.val(self.test_dataloader)
            else:
                test_acc, test_loss = -1.0, float("inf")

            lr_cur = get_lr(self.optimizer)
            self._log_rank0(
                f"Train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, valid acc: {valid_acc:.4f}, "
                f"loss: {valid_loss:.4f}, test acc: {test_acc:.4f}, loss: {test_loss:.4f}, lr: {lr_cur}"
            )

            if not self.multi_gpu or dist.get_rank() == 0:
                self.writer.add_scalar("train_loss", train_loss, self.epoch)
                self.writer.add_scalar("valid_acc", valid_acc, self.epoch)

                save_mode = getattr(self.config.model, "save_mode", "acc")
                if save_mode == "acc":
                    if valid_acc > best_valid:
                        best_valid = valid_acc
                        best_test = test_acc
                        logging.info("Saving checkpoint...")
                        self._save_checkpoint(best_valid)
                else:
                    if valid_loss < lowest_valid_loss:
                        lowest_valid_loss = valid_loss
                        best_test = test_acc
                        logging.info("Saving checkpoint...")
                        self._save_checkpoint(lowest_valid_loss)

            if self.config.scheduler.type.lower() == "steplr":
                self.scheduler.step()

        self._log_rank0(f"Best validation accuracy so far: {best_valid}")
        self._log_rank0(f"Test accuracy when got best validation result: {best_test}")
        if not self.multi_gpu or dist.get_rank() == 0:
            self.writer.close()


SPLITClassifierTrainer = ClassifierTrainer
