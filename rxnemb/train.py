import datetime
import json
import logging
import os

import numpy as np
import torch
import torch.distributed as dist
from box import Box
from sklearn.metrics import r2_score
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from .data import (
    MultiRXNDataset,
    PairDataset,
    RXNDataset,
    TripleDataset,
    get_idx_split,
    pair_collate_fn,
    triple_collate_fn,
)
from .model import RXNGraphormer
from .scheduler import NoamLR, get_linear_scheduler_with_warmup
from .utils import (
    align_config,
    get_lr,
    grad_norm,
    param_norm,
    setup_logger,
    update_dict_key,
)


class SPLITClassifierTrainer:
    def __init__(self, config):
        self.config = config
        self.multi_gpu = self.config.others.multi_gpu

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

        rxng = RXNGraphormer(
            "classification", align_config(input_param, "classifier"), ""
        )
        self.model = rxng.get_model()

        if self.multi_gpu:
            self.local_rank = self.config.others.local_rank
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl")
            self.model.to(self.local_rank)
            if self.config.model.graph_pooling != "attentionxl":
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                )
            else:
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                )
        else:
            self.device = (
                self.config.others.device if torch.cuda.is_available() else "cpu"
            )
            self.model.to(self.device)

        # Initial parameters
        for p in self.model.parameters():
            if p.dim() > 1 and p.requires_grad:
                xavier_uniform_(p)

        total_params = sum(p.numel() for p in self.model.parameters())
        self.save_dir = f"{self.config.model.save_dir}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-classifier-split-{self.config.data.tag}"
        if self.multi_gpu and dist.get_rank() == 0:
            logger = setup_logger(f"{self.save_dir}/log")
        elif not self.multi_gpu:
            logger = setup_logger(f"{self.save_dir}/log")
        if self.multi_gpu:
            self.device_num = dist.get_world_size()
        if self.multi_gpu and dist.get_rank() == 0:
            # self.device_num = dist.get_world_size()
            logging.info(str(self.config))
            logging.info(f"[INFO] Model parameters: {int(total_params/1024/1024)} M")
            logging.info(f"[INFO] World size: {self.device_num}")

        elif not self.multi_gpu:
            logging.info(str(self.config))
            logging.info(f"[INFO] Model parameters: {int(total_params/1024/1024)} M")
        self.init_optimizer()
        self.init_scheduler()

        if self.config.training.loss.lower() == "ce":
            if (
                not hasattr(self.config.training.loss, "weight")
                or self.config.training.loss.weight is False
            ):
                self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
            else:
                self.loss_func = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor(self.config.training.loss.weight).to(
                        self.device
                    ),
                    reduction="mean",
                )
        else:
            raise NotImplementedError(
                f"Loss function {self.config.training.loss} is not implemented yet."
            )
        if self.multi_gpu and dist.get_rank() == 0:
            logging.info(
                f"[INFO] Load reactant dataset {self.config.data.data_path}/{self.config.data.rct_name_regrex}, file trunck {self.config.data.file_num_trunck}, data trunck {self.config.data.data_trunck}..."
            )
            logging.info(
                f"[INFO] Load product dataset {self.config.data.data_path}/{self.config.data.pdt_name_regrex}, file trunck {self.config.data.file_num_trunck}, data trunck {self.config.data.data_trunck}..."
            )
        elif not self.multi_gpu:
            logging.info(
                f"[INFO] Load reactant dataset {self.config.data.data_path}/{self.config.data.rct_name_regrex}, file trunck {self.config.data.file_num_trunck}, data trunck {self.config.data.data_trunck}..."
            )
            logging.info(
                f"[INFO] Load product dataset {self.config.data.data_path}/{self.config.data.pdt_name_regrex}, file trunck {self.config.data.file_num_trunck}, data trunck {self.config.data.data_trunck}..."
            )

        if self.config.data.rct_name_regrex:
            self.rct_dataset = MultiRXNDataset(
                root=self.config.data.data_path,
                name_regrex=self.config.data.rct_name_regrex,
                trunck=self.config.data.data_trunck,
                task=self.config.data.task,
                file_num_trunck=self.config.data.file_num_trunck,
                name_tag="rct",
            )
            self.pdt_dataset = MultiRXNDataset(
                root=self.config.data.data_path,
                name_regrex=self.config.data.pdt_name_regrex,
                trunck=self.config.data.data_trunck,
                task=self.config.data.task,
                file_num_trunck=self.config.data.file_num_trunck,
                name_tag="pdt",
            )
            assert len(self.rct_dataset) == len(
                self.pdt_dataset
            ), "The number of reactant and product data are not equal."
            # self.dataset = PairDataset(self.rct_dataset,self.pdt_dataset)

            self.split_ids_map = get_idx_split(
                len(self.rct_dataset),
                int(self.config.data.train_ratio * len(self.rct_dataset)),
                int(self.config.data.valid_ratio * len(self.rct_dataset)),
                self.config.data.seed,
            )

            self.train_rct_dataset = self.rct_dataset[self.split_ids_map["train"]]
            self.valid_rct_dataset = self.rct_dataset[self.split_ids_map["valid"]]
            self.train_pdt_dataset = self.pdt_dataset[self.split_ids_map["train"]]
            self.valid_pdt_dataset = self.pdt_dataset[self.split_ids_map["valid"]]

            # for pre-train task, test set is unnecessary, so we use valid set as test set
            self.test_rct_dataset = self.valid_rct_dataset
            self.test_pdt_dataset = self.valid_pdt_dataset
        else:
            self.train_rct_dataset = MultiRXNDataset(
                root=self.config.data.data_path,
                name_regrex=self.config.data.train_rct_name_regrex,
                trunck=self.config.data.data_trunck,
                task=self.config.data.task,
                file_num_trunck=self.config.data.file_num_trunck,
                name_tag="rct",
            )
            self.train_pdt_dataset = MultiRXNDataset(
                root=self.config.data.data_path,
                name_regrex=self.config.data.train_pdt_name_regrex,
                trunck=self.config.data.data_trunck,
                task=self.config.data.task,
                file_num_trunck=self.config.data.file_num_trunck,
                name_tag="pdt",
            )
            self.valid_rct_dataset = MultiRXNDataset(
                root=self.config.data.data_path,
                name_regrex=self.config.data.valid_rct_name_regrex,
                trunck=self.config.data.data_trunck,
                task=self.config.data.task,
                file_num_trunck=self.config.data.file_num_trunck,
                name_tag="rct",
            )
            self.valid_pdt_dataset = MultiRXNDataset(
                root=self.config.data.data_path,
                name_regrex=self.config.data.valid_pdt_name_regrex,
                trunck=self.config.data.data_trunck,
                task=self.config.data.task,
                file_num_trunck=self.config.data.file_num_trunck,
                name_tag="pdt",
            )
            self.test_rct_dataset = MultiRXNDataset(
                root=self.config.data.data_path,
                name_regrex=self.config.data.test_rct_name_regrex,
                trunck=self.config.data.data_trunck,
                task=self.config.data.task,
                file_num_trunck=self.config.data.file_num_trunck,
                name_tag="rct",
            )
            self.test_pdt_dataset = MultiRXNDataset(
                root=self.config.data.data_path,
                name_regrex=self.config.data.test_pdt_name_regrex,
                trunck=self.config.data.data_trunck,
                task=self.config.data.task,
                file_num_trunck=self.config.data.file_num_trunck,
                name_tag="pdt",
            )

        if not self.multi_gpu:
            self.train_dataset = PairDataset(
                self.train_rct_dataset, self.train_pdt_dataset
            )
            self.valid_dataset = PairDataset(
                self.valid_rct_dataset, self.valid_pdt_dataset
            )
            self.test_dataset = PairDataset(
                self.test_rct_dataset, self.test_pdt_dataset
            )

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
        else:
            self.train_dataset = PairDataset(
                self.train_rct_dataset, self.train_pdt_dataset
            )
            self.valid_dataset = PairDataset(
                self.valid_rct_dataset, self.valid_pdt_dataset
            )
            self.test_dataset = PairDataset(
                self.test_rct_dataset, self.test_pdt_dataset
            )
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, shuffle=True
            )
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset, shuffle=False
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_dataset, shuffle=False
            )
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.data.batch_size // self.device_num,
                sampler=train_sampler,
                num_workers=0,
            )
            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=self.config.data.batch_size // self.device_num,
                sampler=valid_sampler,
                num_workers=0,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.config.data.batch_size // self.device_num,
                sampler=test_sampler,
                num_workers=0,
            )

        if self.config.model.pretrained_model:
            pretrained_inf = torch.load(
                f"{self.config.model.pretrained_model}/model/valid_checkpoint.pt"
            )
            model_state_dict = pretrained_inf["model_state_dict"]
            optimizer_state_dict = pretrained_inf["optimizer_state_dict"]
            scheduler_state_dict = pretrained_inf["scheduler_state_dict"]
            self.model.load_state_dict(model_state_dict)
            if not self.multi_gpu:
                self.model.to(self.device)
            else:
                self.model.to(self.local_rank)
            if self.config.model.fine_tune:
                if self.multi_gpu and dist.get_rank() == 0:
                    logging.info("Fine-tune setup!")
                elif not self.multi_gpu:
                    logging.info("Fine-tune setup!")
                self.fine_tune()
                self.save_dir += "_ft"
        if self.multi_gpu and dist.get_rank() == 0:
            logging.info(f"[INFO] Training results will be saved in {self.save_dir}")
        elif not self.multi_gpu:
            logging.info(f"[INFO] Training results will be saved in {self.save_dir}")

        # if not os.path.exists(self.save_dir):
        #    os.makedirs(self.save_dir)

        self.log_dir = f"{self.save_dir}/log"
        self.model_save_dir = f"{self.save_dir}/model"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.config.to_json(filename=f"{self.save_dir}/parameters.json")

    def init_optimizer(self):
        if self.config.optimizer.optimizer.lower() == "adam":
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
            )
        elif self.config.optimizer.optimizer.lower() == "adamw":
            self.optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
            )
        else:
            raise Exception(
                f"Unsupport optimizer: '{self.config.optimizer.optimizer.lower()}'"
            )

    def init_scheduler(self):
        if self.config.scheduler.type.lower() == "steplr":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.scheduler.lr_decay_step_size,
                gamma=self.config.scheduler.lr_decay_factor,
            )
        elif self.config.scheduler.type.lower() == "warmup":
            self.scheduler = get_linear_scheduler_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.scheduler.warmup_step,
                num_training_steps=self.config.training.epoch,
            )
        elif self.config.scheduler.type.lower() == "noamlr":
            self.scheduler = NoamLR(
                self.optimizer,
                model_size=self.config.model.emb_dim,
                warmup_steps=self.config.scheduler.warmup_step,
            )
        else:
            raise Exception(
                f"Unsupport scheduler: '{self.config.scheduler.type.lower()}'"
            )

    def fine_tune(self):

        if self.config.model.trainable == "decoder":
            trainable_params_id = list(map(id, self.model.decoder.parameters()))
            fixed_params = filter(
                lambda p: id(p) not in trainable_params_id, self.model.parameters()
            )
            for p in fixed_params:
                p.requires_grad = False
        else:
            raise Exception("trainable should be in ['decoder']")
        self.init_optimizer()
        self.init_scheduler()

    def train(self):
        self.model.train()
        loss_accum = 0
        acc_lst = []
        if self.multi_gpu:
            self.train_dataloader.sampler.set_epoch(self.epoch)
        for step, batch_data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            rct_data, pdt_data = batch_data
            if not self.multi_gpu:
                rct_data = rct_data.to(self.device)
                pdt_data = pdt_data.to(self.device)
            else:
                rct_data = rct_data.to(self.local_rank)
                pdt_data = pdt_data.to(self.local_rank)

            out = self.model([rct_data, pdt_data])
            loss = self.loss_func(out, rct_data.y)
            acc_lst.append((out.argmax(dim=1) == rct_data.y).float().mean().cpu())
            loss.backward()
            if (step + 1) % self.config.training.accum == 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.clip_norm
                )
                self.optimizer.step()
                if self.config.scheduler.type.lower() == "noamlr":
                    self.scheduler.step()
                g_norm = grad_norm(self.model)
                self.model.zero_grad()

            loss_accum += loss.detach().cpu().item()
            if (step + 1) % self.config.training.log_iter_step == 0:
                p_norm = param_norm(self.model)
                lr_cur = get_lr(self.optimizer)
                if self.multi_gpu and dist.get_rank() == 0:
                    logging.info(
                        f"Training step {step+1}, gradient norm: {g_norm:.8f}, parameters norm: {p_norm:.8f}, lr: {lr_cur}, loss: {loss_accum/(step+1):.4f}, acc: {np.mean(acc_lst):.4f}"
                    )
                elif not self.multi_gpu:
                    logging.info(
                        f"Training step {step+1}, gradient norm: {g_norm:.8f}, parameters norm: {p_norm:.8f}, lr: {lr_cur}, loss: {loss_accum/(step+1):.4f}, acc: {np.mean(acc_lst):.4f}"
                    )
        loss_ave = loss_accum / (step + 1)
        acc_ave = np.mean(acc_lst)
        return loss_ave, acc_ave

    def val(self, dataloader):
        self.model.eval()
        if not self.multi_gpu:
            preds = torch.Tensor([]).to(self.device)
            targets = torch.Tensor([]).to(self.device)
        else:
            preds = torch.Tensor([]).to(self.local_rank)
            targets = torch.Tensor([]).to(self.local_rank)
            dataloader.sampler.set_epoch(self.epoch)
        loss_accum = 0
        with torch.no_grad():
            for step, batch_data in enumerate(dataloader):
                rct_data, pdt_data = batch_data
                if not self.multi_gpu:
                    rct_data = rct_data.to(self.device)
                    pdt_data = pdt_data.to(self.device)
                else:
                    rct_data = rct_data.to(self.local_rank)
                    pdt_data = pdt_data.to(self.local_rank)
                out = self.model([rct_data, pdt_data])
                loss = self.loss_func(out, rct_data.y)
                loss_accum += loss.detach().cpu().item()
                pred = torch.argmax(out, dim=1)

                preds = torch.cat([preds, pred.detach_()], dim=0)
                targets = torch.cat([targets, rct_data.y.unsqueeze(1)], dim=0)
            loss_ave = loss_accum / (step + 1)
        return (preds == targets.view(-1)).float().mean().cpu().item(), loss_ave

    def run(self):
        best_valid = -float("inf")
        best_test = -float("inf")

        lowest_valid_loss = float("inf")
        lowest_test_loss = float("inf")

        self.model.zero_grad()
        for self.epoch in range(1, self.config.training.epoch + 1):
            logging.info(f"============= Epoch {self.epoch} =============")

            logging.info("Training...")

            train_loss, train_acc = self.train()

            logging.info("Evaluating...")
            valid_acc, valid_loss = self.val(self.valid_dataloader)
            if (
                hasattr(self.config.data, "test_rct_name_regrex")
                and self.config.data.test_rct_name_regrex
            ):
                test_acc, test_loss = self.val(self.test_dataloader)
            else:
                test_acc = -1
                test_loss = 99999
            lr_cur = get_lr(self.optimizer)

            logging.info(
                f"Train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, valid acc: {valid_acc:.4f}, loss: {valid_loss:.4f}, test acc: {test_acc:.4f}, loss: {test_loss:.4f}, lr: {lr_cur}"
            )

            self.writer.add_scalar("train_loss", train_loss, self.epoch)
            self.writer.add_scalar("valid_acc", valid_acc, self.epoch)
            if (
                not hasattr(self.config.model, "save_mode")
                or self.config.model.save_mode == "acc"
            ):
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    best_test = test_acc

                    logging.info("Saving checkpoint...")
                    checkpoint = {
                        "epoch": self.epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "best_valid_mae": best_valid,
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(self.model_save_dir, "valid_checkpoint.pt"),
                    )
            else:
                if valid_loss < lowest_valid_loss:
                    lowest_valid_loss = valid_loss
                    best_test = test_acc

                    logging.info("Saving checkpoint...")
                    checkpoint = {
                        "epoch": self.epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "best_valid_mae": lowest_valid_loss,
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(self.model_save_dir, "valid_checkpoint.pt"),
                    )
            if self.config.scheduler.type.lower() == "steplr":
                self.scheduler.step()
            # torch.distributed.barrier()  ## 强制同步
        if self.multi_gpu and dist.get_rank() == 0:
            logging.info(f"Best validation accuracy so far: {best_valid}")
            logging.info(f"Test accuracy when got best validation result: {best_test}")
        elif not self.multi_gpu:
            logging.info(f"Best validation accuracy so far: {best_valid}")
            logging.info(f"Test accuracy when got best validation result: {best_test}")
        self.writer.close()


class SPLITRegressorTrainer:
    def __init__(self, config):
        self.config = config
        self.device = self.config.others.device if torch.cuda.is_available() else "cpu"
        self.use_mid_inf = eval(self.config.model.use_mid_inf)
        if self.config.data.rct_data_file:
            prefix = self.config.data.rct_data_file.split(".")[0][:20]
        else:
            prefix = self.config.data.train_rct_data_file.split(".")[0][:20]
        if self.use_mid_inf:
            prefix = prefix + "_mid"
        prefix = f"{prefix}-{self.config.others.tag}"
        self.save_dir = f"{self.config.model.save_dir}/{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not self.config.model.pretrained_model_path:
            logging.info("Training from scratch")
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
                "output_size": 1,
                "output_norm": eval(self.config.model.output_norm),
                "split_process": True,
                "split_merge_method": self.config.model.split_merge_method,
                "output_act_func": self.config.model.output_act_func,
                "rct_layer_norm": eval(self.config.model.rct_layer_norm),
                "pdt_layer_norm": eval(self.config.model.pdt_layer_norm),
                "use_mid_inf": self.use_mid_inf,
                "mid_iteract_method": self.config.model.mid_iteract_method,
                "mid_layer_norm": eval(self.config.model.mid_layer_norm),
                "mid_layer_num": self.config.model.mid_layer_num,
            }

            rxng = RXNGraphormer(
                "regression", align_config(input_param, "regressor"), ""
            )  # pretrain models are all None
            self.model = rxng.get_model()

        else:
            self.pretrained_model_freeze = self.config.model.pretrained_model_freeze
            self.pretrained_lr_scaled_coef = self.config.model.pretrained_lr_scaled_coef
            self.save_dir += "_ft"
            logging.info("Loading pretrained model")
            pretrained_para_json = (
                f"{self.config.model.pretrained_model_path}/parameters.json"
            )
            with open(pretrained_para_json, "r") as fr:
                pretrained_config_dict = json.load(fr)
            pretrained_config = Box(pretrained_config_dict)
            ckpt_file = (
                f"{self.config.model.pretrained_model_path}/model/valid_checkpoint.pt"
            )
            ckpt_inf = torch.load(ckpt_file, map_location=self.device)

            input_param = {
                "emb_dim": pretrained_config.model.emb_dim,
                "gnn_type": pretrained_config.model.gnn_type,
                "gnn_aggr": pretrained_config.model.gnn_aggr,
                "gnum_layer": pretrained_config.model.gnn_num_layer,
                "node_readout": pretrained_config.model.node_readout,
                "num_heads": pretrained_config.model.num_heads,
                "JK": pretrained_config.model.gnn_jk,
                "graph_pooling": pretrained_config.model.graph_pooling,
                "tnum_layer": pretrained_config.model.trans_num_layer,
                "trans_readout": pretrained_config.model.trans_readout,
                "onum_layer": pretrained_config.model.output_num_layer,
                "drop_ratio": pretrained_config.model.drop_ratio,
                "output_size": 2,
                "split_process": True,
                "split_merge_method": pretrained_config.model.split_merge_method,
                "output_act_func": self.config.model.output_act_func,
            }

            rxng = RXNGraphormer(
                "classification", align_config(input_param, "classifier"), ""
            )
            self.pretrained_model = rxng.get_model()

            pretrained_model_params = ckpt_inf["model_state_dict"]
            if list(pretrained_model_params.keys())[0].startswith("module."):
                pretrained_model_params = update_dict_key(
                    pretrained_model_params, "module."
                )
            self.pretrained_model.load_state_dict(pretrained_model_params)
            rct_encoder = self.pretrained_model.rct_encoder
            pdt_encoder = self.pretrained_model.pdt_encoder
            ## before initializing all model parameters, freeze the parameters in pretrained part
            for param in rct_encoder.parameters():
                param.requires_grad = False
            for param in pdt_encoder.parameters():
                param.requires_grad = False

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
                "output_size": 1,
                "output_norm": eval(self.config.model.output_norm),
                "split_process": True,
                "split_merge_method": self.config.model.split_merge_method,
                "output_act_func": self.config.model.output_act_func,
                "rct_layer_norm": eval(self.config.model.rct_layer_norm),
                "pdt_layer_norm": eval(self.config.model.pdt_layer_norm),
                "use_mid_inf": self.use_mid_inf,
                "mid_iteract_method": self.config.model.mid_iteract_method,
                "mid_layer_norm": eval(self.config.model.mid_layer_norm),
                "mid_layer_num": self.config.model.mid_layer_num,
            }

            rxng = RXNGraphormer(
                "regression",
                align_config(input_param, "regressor"),
                "",
                {
                    "pretrained_encoder": None,
                    "pretrained_rct_encoder": rct_encoder,
                    "pretrained_pdt_encoder": pdt_encoder,
                    "pretrained_mid_encoder": None,
                },
            )
            self.model = rxng.get_model()

        self.model.to(self.device)

        # Initial parameters
        for p in self.model.parameters():
            if p.dim() > 1 and p.requires_grad:
                xavier_uniform_(p)
        if self.config.model.pretrained_model_path and not self.pretrained_model_freeze:
            ## unfreeze pretrained model
            logging.info(f"[INFO] Unfreeze pretrained model")
            for param in self.model.rct_encoder.parameters():
                param.requires_grad = True
            for param in self.model.pdt_encoder.parameters():
                param.requires_grad = True

        total_params = sum(p.numel() for p in self.model.parameters())

        logger = setup_logger(f"{self.save_dir}/log")
        logging.info(str(self.config))
        logging.info(f"[INFO] Model parameters: {int(total_params/1024/1024)} M")
        self.init_optimizer()
        self.init_scheduler()

        if self.config.training.loss.lower() == "l1":
            self.loss_func = torch.nn.L1Loss()
        elif self.config.training.loss.lower() == "l2":
            self.loss_func = torch.nn.MSELoss()

        if self.config.data.rct_data_file:
            logging.info(
                f"[INFO] Load dataset {self.config.data.data_path}/{self.config.data.rct_data_file} trunck {self.config.data.data_trunck}..."
            )
            self.rct_dataset = RXNDataset(
                root=self.config.data.data_path,
                name=self.config.data.rct_data_file,
                trunck=self.config.data.data_trunck,
            )
            self.pdt_dataset = RXNDataset(
                root=self.config.data.data_path,
                name=self.config.data.pdt_data_file,
                trunck=self.config.data.data_trunck,
            )

            assert len(self.rct_dataset) == len(
                self.pdt_dataset
            ), "The number of reactants and products should be the same."
            self.split_ids_map = get_idx_split(
                len(self.rct_dataset),
                int(self.config.data.train_ratio * len(self.rct_dataset)),
                int(self.config.data.valid_ratio * len(self.rct_dataset)),
                self.config.data.seed,
            )
            self.train_rct_dataset = self.rct_dataset[self.split_ids_map["train"]]
            self.valid_rct_dataset = self.rct_dataset[self.split_ids_map["valid"]]
            self.train_pdt_dataset = self.pdt_dataset[self.split_ids_map["train"]]
            self.valid_pdt_dataset = self.pdt_dataset[self.split_ids_map["valid"]]

            if len(self.split_ids_map["test"]) > 10:
                self.test_rct_dataset = self.rct_dataset[self.split_ids_map["test"]]
                self.test_pdt_dataset = self.pdt_dataset[self.split_ids_map["test"]]
            else:
                self.test_rct_dataset = self.valid_rct_dataset
                self.test_pdt_dataset = self.valid_pdt_dataset
            logging.info(
                f"[INFO] Split dataset into train: {len(self.train_rct_dataset)}, valid: {len(self.valid_rct_dataset)}, test: {len(self.test_rct_dataset)}. Random seed: {self.config.data.seed}"
            )
        else:
            logging.info(
                f"[INFO] Load dataset {self.config.data.data_path}/{self.config.data.train_rct_data_file} and {self.config.data.train_pdt_data_file} trunck {self.config.data.data_trunck}..."
            )
            self.train_rct_dataset = RXNDataset(
                root=self.config.data.data_path,
                name=self.config.data.train_rct_data_file,
                trunck=self.config.data.data_trunck,
            )
            self.train_pdt_dataset = RXNDataset(
                root=self.config.data.data_path,
                name=self.config.data.train_pdt_data_file,
                trunck=self.config.data.data_trunck,
            )
            logging.info(
                f"[INFO] Load dataset {self.config.data.data_path}/{self.config.data.val_rct_data_file} and {self.config.data.val_pdt_data_file} trunck {self.config.data.data_trunck}..."
            )
            self.valid_rct_dataset = RXNDataset(
                root=self.config.data.data_path,
                name=self.config.data.val_rct_data_file,
                trunck=self.config.data.data_trunck,
            )
            self.valid_pdt_dataset = RXNDataset(
                root=self.config.data.data_path,
                name=self.config.data.val_pdt_data_file,
                trunck=self.config.data.data_trunck,
            )
            logging.info(
                f"[INFO] Load dataset {self.config.data.data_path}/{self.config.data.test_rct_data_file} and {self.config.data.test_pdt_data_file} trunck {self.config.data.data_trunck}..."
            )
            self.test_rct_dataset = RXNDataset(
                root=self.config.data.data_path,
                name=self.config.data.test_rct_data_file,
                trunck=self.config.data.data_trunck,
            )
            self.test_pdt_dataset = RXNDataset(
                root=self.config.data.data_path,
                name=self.config.data.test_pdt_data_file,
                trunck=self.config.data.data_trunck,
            )
            assert (
                len(self.train_rct_dataset) == len(self.train_pdt_dataset)
                and len(self.valid_rct_dataset) == len(self.valid_pdt_dataset)
                and len(self.test_rct_dataset) == len(self.test_pdt_dataset)
            ), "The length of the datasets are not equal!"

        if self.use_mid_inf:
            if self.config.data.rct_data_file:
                self.mid_dataset = RXNDataset(
                    root=self.config.data.data_path,
                    name=self.config.data.mid_data_file,
                    trunck=self.config.data.data_trunck,
                )
                self.train_mid_dataset = self.mid_dataset[self.split_ids_map["train"]]
                self.valid_mid_dataset = self.mid_dataset[self.split_ids_map["valid"]]

                if len(self.split_ids_map["test"]) > 10:
                    self.test_mid_dataset = self.mid_dataset[self.split_ids_map["test"]]
                else:
                    self.test_mid_dataset = self.valid_mid_dataset

            else:
                logging.info(
                    f"[INFO] Load dataset {self.config.data.data_path}/{self.config.data.train_mid_data_file} trunck {self.config.data.data_trunck}..."
                )
                self.train_mid_dataset = RXNDataset(
                    root=self.config.data.data_path,
                    name=self.config.data.train_mid_data_file,
                    trunck=self.config.data.data_trunck,
                )
                logging.info(
                    f"[INFO] Load dataset {self.config.data.data_path}/{self.config.data.val_mid_data_file} trunck {self.config.data.data_trunck}..."
                )
                self.valid_mid_dataset = RXNDataset(
                    root=self.config.data.data_path,
                    name=self.config.data.val_mid_data_file,
                    trunck=self.config.data.data_trunck,
                )
                logging.info(
                    f"[INFO] Load dataset {self.config.data.data_path}/{self.config.data.test_mid_data_file} trunck {self.config.data.data_trunck}..."
                )
                self.test_mid_dataset = RXNDataset(
                    root=self.config.data.data_path,
                    name=self.config.data.test_mid_data_file,
                    trunck=self.config.data.data_trunck,
                )
            assert (
                len(self.train_mid_dataset)
                == len(self.train_rct_dataset)
                == len(self.train_pdt_dataset)
            ), "Length of train dataset is not equal"
            assert (
                len(self.valid_mid_dataset)
                == len(self.valid_rct_dataset)
                == len(self.valid_pdt_dataset)
            ), "Length of valid dataset is not equal"
            assert (
                len(self.test_mid_dataset)
                == len(self.test_rct_dataset)
                == len(self.test_pdt_dataset)
            ), "Length of test dataset is not equal"
            self.train_dataset = TripleDataset(
                self.train_rct_dataset, self.train_pdt_dataset, self.train_mid_dataset
            )
            self.valid_dataset = TripleDataset(
                self.valid_rct_dataset, self.valid_pdt_dataset, self.valid_mid_dataset
            )
            self.test_dataset = TripleDataset(
                self.test_rct_dataset, self.test_pdt_dataset, self.test_mid_dataset
            )

            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=True,
                collate_fn=triple_collate_fn,
            )
            self.valid_dataloader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                collate_fn=triple_collate_fn,
            )
            self.test_dataloader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                collate_fn=triple_collate_fn,
            )
        else:
            assert len(self.train_rct_dataset) == len(
                self.train_pdt_dataset
            ), "Length of train dataset is not equal"
            assert len(self.valid_rct_dataset) == len(
                self.valid_pdt_dataset
            ), "Length of valid dataset is not equal"
            assert len(self.test_rct_dataset) == len(
                self.test_pdt_dataset
            ), "Length of test dataset is not equal"
            self.train_dataset = PairDataset(
                self.train_rct_dataset, self.train_pdt_dataset
            )
            self.valid_dataset = PairDataset(
                self.valid_rct_dataset, self.valid_pdt_dataset
            )
            self.test_dataset = PairDataset(
                self.test_rct_dataset, self.test_pdt_dataset
            )

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

        logging.info(f"[INFO] Training results will be saved in {self.save_dir}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.log_dir = f"{self.save_dir}/log"
        self.model_save_dir = f"{self.save_dir}/model"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.config.to_json(filename=f"{self.save_dir}/parameters.json")

    def init_optimizer(self):
        if self.config.model.pretrained_model_path and not self.pretrained_model_freeze:
            param_lst = [
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.model.rct_encoder.parameters()
                    ),
                    "lr": self.config.optimizer.learning_rate
                    * self.pretrained_lr_scaled_coef,
                },
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.model.pdt_encoder.parameters()
                    ),
                    "lr": self.config.optimizer.learning_rate
                    * self.pretrained_lr_scaled_coef,
                },
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.model.decoder.parameters()
                    ),
                    "lr": self.config.optimizer.learning_rate,
                },
            ]
            if self.use_mid_inf:
                param_lst.append(
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            self.model.mid_encoder.parameters(),
                        ),
                        "lr": self.config.optimizer.learning_rate,
                    }
                )
                param_lst.append(
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            self.model.mid_iteract.parameters(),
                        ),
                        "lr": self.config.optimizer.learning_rate,
                    }
                )
                param_lst.append(
                    {
                        "params": filter(
                            lambda p: p.requires_grad,
                            self.model.mid_decoder.parameters(),
                        ),
                        "lr": self.config.optimizer.learning_rate,
                    }
                )
        if self.config.optimizer.optimizer.lower() == "adam":
            if (
                not self.config.model.pretrained_model_path
                or self.pretrained_model_freeze
            ):
                self.optimizer = Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.config.optimizer.learning_rate,
                    weight_decay=self.config.optimizer.weight_decay,
                )
            else:
                logging.info(f"[INFO] Use different learning rate for pretrained model")

                self.optimizer = Adam(
                    param_lst,
                    lr=self.config.optimizer.learning_rate,
                    weight_decay=self.config.optimizer.weight_decay,
                )
        elif self.config.optimizer.optimizer.lower() == "adamw":
            if (
                not self.config.model.pretrained_model_path
                or self.pretrained_model_freeze
            ):
                self.optimizer = AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.config.optimizer.learning_rate,
                    weight_decay=self.config.optimizer.weight_decay,
                )
            else:
                logging.info(f"[INFO] Use different learning rate for pretrained model")
                self.optimizer = AdamW(
                    param_lst,
                    lr=self.config.optimizer.learning_rate,
                    weight_decay=self.config.optimizer.weight_decay,
                )
        else:
            raise Exception(
                f"Unsupport optimizer: '{self.config.optimizer.optimizer.lower()}'"
            )

    def init_scheduler(self):
        if self.config.scheduler.type.lower() == "steplr":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.scheduler.lr_decay_step_size,
                gamma=self.config.scheduler.lr_decay_factor,
            )
        elif self.config.scheduler.type.lower() == "warmup":
            self.scheduler = get_linear_scheduler_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.scheduler.warmup_step,
                num_training_steps=self.config.training.epoch,
            )
        elif self.config.scheduler.type.lower() == "noamlr":
            self.scheduler = NoamLR(
                self.optimizer,
                model_size=self.config.model.emb_dim,
                warmup_steps=self.config.scheduler.warmup_step,
            )
        else:
            raise Exception(
                f"Unsupport scheduler: '{self.config.scheduler.type.lower()}'"
            )

    def train(self):
        self.model.train()
        loss_accum = 0
        for step, batch_data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            if not self.use_mid_inf:
                rct_data, pdt_data = batch_data
                rct_data = rct_data.to(self.device)
                pdt_data = pdt_data.to(self.device)
                out = self.model([rct_data, pdt_data])
            else:
                rct_data, pdt_data, mid_data = batch_data
                rct_data = rct_data.to(self.device)
                pdt_data = pdt_data.to(self.device)
                mid_data = mid_data.to(self.device)
                out = self.model([rct_data, pdt_data, mid_data])

            loss = self.loss_func(out, rct_data.y.unsqueeze(1))
            loss.backward()
            if (step + 1) % self.config.training.accum == 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.clip_norm
                )
                self.optimizer.step()
                if self.config.scheduler.type.lower() == "noamlr":
                    self.scheduler.step()
                g_norm = grad_norm(self.model)
                self.model.zero_grad()
            loss_accum += loss.detach().cpu().item()
        loss_ave = loss_accum / (step + 1)
        return loss_ave

    def val(self, dataloader):
        self.model.eval()
        preds = torch.Tensor([]).to(self.device)
        targets = torch.Tensor([]).to(self.device)
        with torch.no_grad():
            for step, batch_data in enumerate(dataloader):
                if not self.use_mid_inf:
                    rct_data, pdt_data = batch_data
                    rct_data = rct_data.to(self.device)
                    pdt_data = pdt_data.to(self.device)
                    out = self.model([rct_data, pdt_data])
                else:
                    rct_data, pdt_data, mid_data = batch_data
                    rct_data = rct_data.to(self.device)
                    pdt_data = pdt_data.to(self.device)
                    mid_data = mid_data.to(self.device)
                    out = self.model([rct_data, pdt_data, mid_data])
                preds = torch.cat([preds, out.detach_()], dim=0)
                targets = torch.cat([targets, rct_data.y.unsqueeze(1)], dim=0)
        r2 = r2_score(targets.cpu(), preds.cpu())
        return torch.mean(torch.abs(targets - preds)).cpu().item(), r2

    def run(self):
        best_valid = float("inf")
        best_test = float("inf")
        best_valid_r2 = -float("inf")
        best_test_r2 = -float("inf")
        self.model.zero_grad()
        for epoch in range(1, self.config.training.epoch + 1):
            logging.info(f"============= Epoch {epoch} =============")

            logging.info("Training...")

            train_loss = self.train()

            logging.info("Evaluating...")
            valid_mae, valid_r2 = self.val(self.valid_dataloader)

            logging.info("Testing...")
            test_mae, test_r2 = self.val(self.test_dataloader)

            lr_cur = get_lr(self.optimizer)

            logging.info(
                f"Train: {train_loss:.8f}, validation mae: {valid_mae:.8f}, r2: {valid_r2}, test mae: {test_mae:.8f}, r2: {test_r2}, lr: {lr_cur}"
            )

            self.writer.add_scalar("train_loss", train_loss, epoch)
            self.writer.add_scalar("valid_mae", valid_mae, epoch)
            self.writer.add_scalar("test_mae", test_mae, epoch)

            if valid_mae < best_valid:
                best_valid = valid_mae
                best_test = test_mae
                best_valid_r2 = valid_r2
                best_test_r2 = test_r2
                logging.info("Saving checkpoint...")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_valid_mae": best_valid,
                    "best_valid_r2": best_valid_r2,
                }
                torch.save(
                    checkpoint, os.path.join(self.model_save_dir, "valid_checkpoint.pt")
                )
            if self.config.scheduler.type.lower() == "steplr":
                self.scheduler.step()

        logging.info(f"Best validation MAE so far: {best_valid}, R2: {best_valid_r2}")
        logging.info(
            f"Test MAE when got best validation result: {best_test}, R2: {best_test_r2}"
        )

        self.writer.close()
