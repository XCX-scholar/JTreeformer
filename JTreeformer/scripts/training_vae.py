import os
import sys
import json
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import VAEConfig
from models.jtreeformer import JTreeformer
from data_processing.dataloader import create_vae_dataloader

class KLCycleScheduler:
    """
    A scheduler for the KL divergence weight (beta) that cycles periodically.
    It ramps up linearly for half the cycle and stays at max_beta for the other half.
    """

    def __init__(self, cycle_len: int, start_beta: float = 0.0, max_beta: float = 1.0):
        self.cycle_len = cycle_len
        self.start_beta = start_beta
        self.max_beta = max_beta
        self.step_count = 0

    def step(self) -> float:
        """Advance one step and return the current beta value."""
        self.step_count += 1
        if self.cycle_len == 0:
            return self.max_beta

        ramp_up_steps = self.cycle_len // 2
        cycle_step = self.step_count % self.cycle_len

        if cycle_step < ramp_up_steps:
            beta = self.start_beta + (self.max_beta - self.start_beta) * (cycle_step / ramp_up_steps)
        else:
            beta = self.max_beta
        return beta

    def state_dict(self) -> dict:
        return {'step_count': self.step_count}

    def load_state_dict(self, state_dict: dict):
        self.step_count = state_dict['step_count']


class Trainer:
    """
    A class to encapsulate the training and validation loop for the JTreeformer model.
    """

    def __init__(self, config: VAEConfig, args):
        self.config = config
        self.args = args
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # --- Data Loading ---
        if args.train:
            self.train_loader = create_vae_dataloader(args.train_path, args.batch_size, shuffle=True)
            self.valid_loader = create_vae_dataloader(args.valid_path, args.batch_size, shuffle=False)
        if args.evaluate:
            self.test_loader = create_vae_dataloader(args.test_path, args.batch_size, shuffle=False)

        # --- Model, Vocab, and Scaler ---
        self.vocab = self._load_vocab(args.vocab_path)
        self.model = JTreeformer(config, self.vocab).to(self.device)

        self.scaler = None
        if args.predict_properties:
            config.predict_properties = True
            self.scaler = self._load_scaler(args.scaler_path)
            self.property_loss_fn = nn.MSELoss()

        # --- Optimizer and Schedulers ---
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = self._create_lr_scheduler()
        self.kl_scheduler = KLCycleScheduler(args.kl_cycle_len)

        # --- State Management ---
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        self.log_file_path = os.path.join(args.checkpoint_dir, 'training_log.json')

        if args.resume_checkpoint or (args.evaluate and not args.train):
            checkpoint_path = args.resume_checkpoint or args.checkpoint_path
            if checkpoint_path and os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)
            elif args.evaluate:
                raise FileNotFoundError(f"Checkpoint for evaluation not found at {checkpoint_path}")

    def _load_vocab(self, path):
        with open(path, 'r') as f: return json.load(f)

    def _load_scaler(self, path):
        with open(path, 'r') as f: scaler_data = json.load(f)
        return {'mean': torch.tensor(scaler_data['mean'], device=self.device),
                'std': torch.tensor(scaler_data['std'], device=self.device)}

    def _create_lr_scheduler(self):
        num_training_steps = self.args.epochs * len(self.train_loader) if self.args.train else 1

        def lr_lambda(current_step: int):
            if current_step < self.args.warmup_steps:
                return float(current_step) / float(max(1, self.args.warmup_steps))
            progress = float(current_step - self.args.warmup_steps) / float(
                max(1, num_training_steps - self.args.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    def _compute_loss(self, model_output, batch, kl_beta):
        losses = {}
        loss_node = F.cross_entropy(model_output["node_logits"][:,:-1,:].reshape(-1, model_output["node_logits"].size(-1)),
                                    batch.x_dense.reshape(-1), ignore_index=0)
        # print(model_output["relation_logits"].shape, batch.relations_dense.shape)
        loss_relation = F.cross_entropy(
            model_output["relation_logits"][:,:-1,:].reshape(-1, model_output["relation_logits"].size(-1)),
            batch.relations_dense.view(-1), ignore_index=-1)
        total_loss = loss_node + loss_relation
        losses.update({"loss_node": loss_node.detach(), "loss_relation": loss_relation.detach()})

        if self.config.is_vae:
            mean, logvar = model_output["mean"], model_output["logvar"]
            loss_kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch.num_graphs
            total_loss += kl_beta * loss_kl
            losses["loss_kl"] = loss_kl.detach()

        if self.args.predict_properties:
            prop_targets = (batch.properties - self.scaler['mean']) / self.scaler['std']
            loss_prop = self.property_loss_fn(model_output["property_preds"], prop_targets)
            total_loss += loss_prop
            losses["loss_prop"] = loss_prop.detach()

        losses["total_loss"] = total_loss
        return losses

    def _save_checkpoint(self, epoch, is_best):
        state = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                 'kl_scheduler_state_dict': self.kl_scheduler.state_dict(),
                 'best_val_loss': self.best_val_loss}
        torch.save(state, os.path.join(self.args.checkpoint_dir, 'checkpoint_latest.pth'))
        if is_best:
            torch.save(state, os.path.join(self.args.checkpoint_dir, 'checkpoint_best.pth'))
            print(f"Saved new best checkpoint.")

    def _load_checkpoint(self, path):
        print(f"Loading checkpoint from: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt and self.args.train:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            self.kl_scheduler.load_state_dict(ckpt['kl_scheduler_state_dict'])
            self.start_epoch = ckpt['epoch'] + 1
            self.best_val_loss = ckpt['best_val_loss']
            print(f"Resumed training from epoch {self.start_epoch - 1}, best val loss: {self.best_val_loss:.4f}")
        else:
            print("Loaded model weights for evaluation.")

    def _log_to_file(self, log_data):
        with open(self.log_file_path, 'a') as f:
            f.write(json.dumps(log_data) + '\n')

    def _run_epoch(self, epoch, is_train=True, use_test_set=False):
        self.model.train(is_train)

        if use_test_set:
            loader = self.test_loader
            mode_str = 'Test'
        else:
            loader = self.train_loader if is_train else self.valid_loader
            mode_str = 'Train' if is_train else 'Valid'

        total_loss_agg = defaultdict(float)
        progress_bar = tqdm(loader, desc=f"Epoch {epoch} [{mode_str}]")

        for i, batch in enumerate(progress_bar):
            batch = batch.to(self.device)
            kl_beta = self.kl_scheduler.step() if is_train else self.kl_scheduler.max_beta

            if is_train:
                model_output = self.model(batch)
                losses = self._compute_loss(model_output, batch, kl_beta)
                self.optimizer.zero_grad()
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
            else:
                with torch.no_grad():
                    model_output = self.model(batch)
                    losses = self._compute_loss(model_output, batch, kl_beta)

            for key, val in losses.items(): total_loss_agg[key] += val.item()

            if (i + 1) % self.args.log_interval == 0 and is_train:
                avg_losses = {k: v / (i + 1) for k, v in total_loss_agg.items()}
                log_str = f"LR: {self.lr_scheduler.get_last_lr()[0]:.6f} | KL Beta: {kl_beta:.3f} | " + \
                          " | ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
                progress_bar.set_postfix_str(log_str)
                log_entry = {'epoch': epoch, 'step': i + 1, 'mode': mode_str, **avg_losses}
                self._log_to_file(log_entry)

        return {key: val / len(loader) for key, val in total_loss_agg.items()}

    def train(self):
        print("Starting training...")
        for epoch in range(self.start_epoch, self.args.epochs):
            train_losses = self._run_epoch(epoch, is_train=True)
            val_losses = self._run_epoch(epoch, is_train=False)

            print(f"Epoch {epoch} Train Summary: " + " | ".join([f"{k}: {v:.4f}" for k, v in train_losses.items()]))
            print(f"Epoch {epoch} Valid Summary: " + " | ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()]))

            is_best = val_losses["total_loss"] < self.best_val_loss
            if is_best: self.best_val_loss = val_losses["total_loss"]
            self._save_checkpoint(epoch, is_best)
        print("Training finished.")

    def evaluate(self):
        """Runs evaluation on the test set and saves the results."""
        print("\n--- Running Evaluation on Test Set ---")
        test_losses = self._run_epoch(epoch=self.start_epoch, is_train=False, use_test_set=True)
        print("\n--- Test Set Evaluation Results ---")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in test_losses.items()]))
        print("-----------------------------------\n")
        results_path = self.args.results_path
        if not results_path:
            results_path = os.path.join(self.args.checkpoint_dir, 'vae_evaluation_results.json')

        print(f"Saving evaluation results to {results_path}")
        with open(results_path, 'w') as f:
            json.dump(test_losses, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Evaluate JTreeformer VAE")

    # --- Mode ---
    parser.add_argument('--train', action='store_true', help="Run the training pipeline.")
    parser.add_argument('--evaluate', action='store_true', help="Run evaluation on the test set.")

    # --- Paths ---
    parser.add_argument('--train_path', type=str, help="Path to training LMDB dataset.")
    parser.add_argument('--valid_path', type=str, help="Path to validation LMDB dataset.")
    parser.add_argument('--test_path', type=str, help="Path to test LMDB dataset.")
    parser.add_argument('--vocab_path', type=str, required=True, help="Path to vocabulary json file.")
    parser.add_argument('--scaler_path', type=str, default=None, help="Path to property scaler json file.")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help="Directory to save/load checkpoints.")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Checkpoint to resume training from.")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Checkpoint to use for evaluation.")

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--predict_properties', action='store_true')
    parser.add_argument('--kl_cycle_len', type=int, default=5000)
    parser.add_argument('--log_interval', type=int, default=100)

    args = parser.parse_args()

    if not args.train and not args.evaluate:
        parser.error("Must specify at least one mode: --train or --evaluate")
    if args.train and (not args.train_path or not args.valid_path):
        parser.error("--train_path and --valid_path are required for training.")
    if args.evaluate and not args.test_path:
        parser.error("--test_path is required for evaluation.")
    if args.predict_properties and not args.scaler_path:
        parser.error("--scaler_path is required when --predict_properties is set.")

    model_config = VAEConfig()
    trainer = Trainer(model_config, args)

    if args.train:
        trainer.train()

    if args.evaluate:
        trainer.evaluate()
