from models.BaseModel import BaseModel
from dataloaders.prediction_molecule_dataset import PygPredictionMoleculeDataset
from misc import eval_func
from models.GraphQwen3Action import GraphQwen3Action

import yaml 
import os
import os.path as osp 
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader


class Qwen3(BaseModel):
    def __init__(self):
        super(InfoAlign, self).__init__()

        with open("configs/Graph_Qwen3_Action.yaml", "r") as f:
            configs = yaml.safe_load(f)
        self.configs = configs
        self.build_dataloader(configs)
        self.build_model(configs)
        self.build_optimizer(configs)
        # self.scheduler = self.build_scheduler(configs)

        self.build_criterion(configs)
        self.build_metrics(configs)

    def train(self):
        eval_metric = self.configs["dataset"]["eval_metric"]
        if eval_metric == "roc_auc":
            best_val_metric = 0.0
        elif eval_metric == 'avg_mae':
            best_val_metric = 10e9
        else:
            raise ValueError(f"Invalid eval metric: {eval_metric}")

        if self.configs["model"]["load_model"]:
            self.load_model(self.configs["model"]["load_model_path"])
            best_val_metric = self.evaluate(mode='val')[eval_metric]
        
        for epoch in range(self.configs['trainer']['max_epochs']):
            self.model.train()
            average_loss = 0.0
            for batch_idx, data in enumerate(self.train_dataloader):
                data = data.to(self.device)
                targets = data.y.to(torch.float32).to(self.device)
                is_labeled = targets == targets
                preds = self.model(data)
                
                batch_loss = self.loss_fn(
                    preds.view(targets.size()).to(torch.float32)[is_labeled],
                    targets[is_labeled],
                ).mean()
                average_loss += batch_loss.item()
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                # print training loss
                print_average_loss = average_loss / (batch_idx + 1)
                #print(f"Train Epoch: {epoch + 1}/{self.configs['trainer']['max_epochs']}. Iter: {batch_idx + 1}/{len(self.train_dataloader)}. Loss: {print_average_loss:.4f}")
            eval_metrics = self.evaluate(mode='val')

            if eval_metric == "roc_auc":
                if eval_metrics[eval_metric] > best_val_metric:
                    best_val_metric = eval_metrics[eval_metric]
                    self.save_model(f"infoalign_model_best.pth")
                    print(f"Best model saved at epoch {epoch + 1} with ROC-AUC: {best_val_metric:.4f}")
            elif eval_metric == "avg_mae":
                if eval_metrics[eval_metric] < best_val_metric:
                    best_val_metric = eval_metrics[eval_metric]
                    self.save_model(f"infoalign_model_best.pth")
                    print(f"Best model saved at epoch {epoch + 1} with AVG-MAE: {best_val_metric:.4f}")
            else:
                raise ValueError(f"Invalid eval metric: {eval_metric}")
        
        # save model
        self.save_model(f"infoalign_model_last.pth")

    def evaluate(self, mode='val', load_model=False):
        if mode == 'val':
            dataloader = self.val_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if load_model:
            self.configs["model"]["load_model"] = True
            self.load_model(self.configs["model"]["load_model_path"])

        self.model.eval()
        gt, predictions = [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                data = data.to(self.device)
                targets = data.y.to(torch.float32).to(self.device)
                preds = self.model(data)

                gt.append(targets.view(preds.shape).detach().cpu())
                predictions.append(preds.detach().cpu())

        gt = torch.cat(gt, dim=0).numpy()
        predictions = torch.cat(predictions, dim=0).numpy()
        metrics = self.metrics_fn(predictions, gt)
        # print(f"Validation Metrics: {metrics['roc_auc']:.4f}")
        return metrics

    def save_model(self, ckpt):
        ckpt_dir = osp.join(self.configs['model']['save_path'], self.configs['dataset']['name'])
        if not osp.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = osp.join(ckpt_dir, ckpt)
        # torch save model and optimizer state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, ckpt_path)
 
    def build_model(self, configs):
        self.model = InfoAlignModule(configs)
        self.model = self.model.to(self.device)
    
    def build_dataloader(self, configs):
        dataset = PygPredictionMoleculeDataset(configs)
        split_idx = dataset.get_idx_split()
        self.train_dataloader = DataLoader(
            dataset[split_idx["train"]], 
            batch_size=configs["trainer"]["batch_size"],
            shuffle=True, 
        )
        self.val_dataloader = DataLoader(
            dataset[split_idx["valid"]], 
            batch_size=configs["trainer"]["batch_size"], 
            shuffle=False, 
        )
        self.test_dataloader = DataLoader(
            dataset[split_idx["test"]], 
            batch_size=configs["trainer"]["batch_size"], 
            shuffle=False, 
        )

    def build_optimizer(self, configs):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=configs["trainer"]["learning_rate"],
            weight_decay=configs["trainer"]["weight_decay"]
        )
    
    def build_scheduler(self, configs):
        self.scheduler = self.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=configs["trainer"]["num_warmup_steps"],
            num_training_steps=configs["trainer"]["num_training_steps"]
        )
    
    def build_criterion(self, configs):
        if configs["dataset"]["criterion"] == "mse":
            self.loss_fn = nn.MSELoss()
        elif configs["dataset"]["criterion"] == "mae":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Invalid criterion: {configs['dataset']['criterion']}")
    
    def build_metrics(self, configs):
        self.metrics_fn = eval_func

    def load_model(self, ckpt):
        if self.configs["model"]["load_model"]:
            ckpt_path = osp.join(self.configs["model"]["save_path"], self.configs["dataset"]["name"], self.configs["model"]["load_model_path"])
            if osp.exists(ckpt_path):
                self.model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
                print(f"Model loaded from {ckpt_path}")
            else:
                raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
        else:
            print("No model checkpoint to load")
    
    