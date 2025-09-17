import os
import json
import argparse
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F

from models.noise_predictor import NoisePredictorMLP
from models.target_predictor import MultiTargetPredictor
from models.diffusion_model import DiffusionModel
from data_processing.dataloader import create_predictor_dataloader

class PredictorTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.targets_config = args.targets_config
        print(f"Using device: {self.device}")

        if args.train:
            self.train_loader = create_predictor_dataloader(
                data_path=args.train_data_path,
                batch_size=args.batch_size,
                targets_config=self.targets_config,
                shuffle=True
            )
            self.valid_loader = create_predictor_dataloader(
                data_path=args.valid_data_path,
                batch_size=args.batch_size,
                targets_config=self.targets_config,
                shuffle=False
            )
        if args.evaluate:
            self.test_loader = create_predictor_dataloader(
                data_path=args.test_data_path,
                batch_size=args.batch_size,
                targets_config=self.targets_config,
                shuffle=False
            )

        noise_predictor_for_diffusion = NoisePredictorMLP(latent_dim=args.latent_dim, time_embed_dim=args.time_embed_dim,
                                                 hidden_dim=args.ddpm_hidden_dim, num_layers=args.ddpm_num_layers).to(self.device)
        self.diffusion_model = DiffusionModel(noise_predictor_for_diffusion, timesteps=args.timesteps)
        self.predictor = MultiTargetPredictor(latent_dim=args.latent_dim, hidden_dim=args.pred_hidden_dim,
                                              num_layers=args.pred_num_layers, targets_config=self.targets_config).to(
            self.device)

        self.loss_fn = F.mse_loss
        if args.train:
            try:
                ckpt = torch.load(args.diffusion_checkpoint_path, map_location=self.device)
                self.diffusion_model.noise_predictor.load_state_dict(ckpt['model_state_dict'])
            except FileNotFoundError:
                print(
                    f"ERROR: Diffusion model checkpoint not found at {args.diffusion_checkpoint_path} for training. Exiting.")
                exit()
            self.diffusion_model.noise_predictor.eval()
            for param in self.diffusion_model.noise_predictor.parameters():
                param.requires_grad = False

            self.optimizer = optim.AdamW(self.predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            self.lr_scheduler = self._create_lr_scheduler()

        self.start_epoch = 0
        self.best_val_metric = float('inf')
        os.makedirs(args.checkpoint_dir, exist_ok=True)


        if args.resume_checkpoint or (args.evaluate and not args.train):
            checkpoint_path = args.resume_checkpoint or args.checkpoint_path
            if checkpoint_path and os.path.exists(checkpoint_path):
                self._load_checkpoint(checkpoint_path)
            elif args.evaluate:
                raise FileNotFoundError(f"Checkpoint for evaluation not found at {checkpoint_path}")

    def _create_lr_scheduler(self):
        num_training_steps = self.args.epochs * len(self.train_loader)

        def lr_lambda(current_step: int):
            if current_step < self.args.warmup_steps:
                return float(current_step) / float(max(1, self.args.warmup_steps))
            progress = float(current_step - self.args.warmup_steps) / float(
                max(1, num_training_steps - self.args.warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    def _compute_loss_and_metrics(self, batch, is_eval=False):
        latents, targets_dict = batch
        latents = latents.to(self.device)
        targets_dict = {k: v.to(self.device) for k, v in targets_dict.items()}
        batch_size = latents.shape[0]

        with torch.no_grad():
            t = torch.randint(0, self.args.timesteps, (batch_size,), device=self.device).long()
            noise = torch.randn_like(latents)
            noisy_latents = self.diffusion_model.q_sample(x_start=latents, t=t, noise=noise)

        predictions_dict = self.predictor(noisy_latents)

        total_loss = 0
        for name, target_value in targets_dict.items():
            total_loss += self.loss_fn(predictions_dict[name], target_value)

        metrics = {}
        if is_eval:
            metrics['avg_l1_metric'] = sum(
                F.l1_loss(predictions_dict[n], v).detach() for n, v in targets_dict.items()) / len(targets_dict)
        return total_loss, metrics

    def _save_checkpoint(self, epoch, is_best):
        state = {
            'epoch': epoch,
            'model_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_metric': self.best_val_metric
        }
        torch.save(state, os.path.join(self.args.checkpoint_dir, 'predictor_checkpoint_latest.pth'))
        if is_best:
            torch.save(state, os.path.join(self.args.checkpoint_dir, 'predictor_checkpoint_best.pth'))
            print("Saved new best predictor checkpoint.")

    def _load_checkpoint(self, path):
        print(f"Loading checkpoint from: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.predictor.load_state_dict(ckpt['model_state_dict'])

        if self.args.train:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            self.start_epoch = ckpt['epoch'] + 1
            self.best_val_metric = ckpt['best_val_metric']
            print(f"Resumed training from epoch {self.start_epoch}, best val metric: {self.best_val_metric:.4f}")
        else:
            print("Loaded model weights for evaluation.")

        if not os.path.exists(path):
            print(f"ERROR: Checkpoint not found at {path}. Exiting.")
            exit()

    def _run_epoch(self, epoch, is_train=True):
        self.predictor.train(is_train)
        loader = self.train_loader if is_train else self.valid_loader
        mode_str = 'Train' if is_train else 'Valid'
        agg_loss = 0.0
        agg_metrics = defaultdict(float)
        progress_bar = tqdm(loader, desc=f"Epoch {epoch} [{mode_str}]")

        for i, batch in enumerate(progress_bar):
            if is_train:
                loss, _ = self._compute_loss_and_metrics(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                agg_loss += loss.item()
                avg_loss = agg_loss / (i + 1)
                lr = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix_str(f"LR: {lr:.6f} | Loss: {avg_loss:.4f}")
            else:
                with torch.no_grad():
                    loss, metrics = self._compute_loss_and_metrics(batch, is_eval=True)
                    agg_loss += loss.item()
                    for key, val in metrics.items():
                        agg_metrics[key] += val.item()

        final_avg_loss = agg_loss / len(loader)
        final_avg_metrics = {key: val / len(loader) for key, val in agg_metrics.items()}
        return final_avg_loss, final_avg_metrics

    def train(self):
        print("Starting predictor training...")
        for epoch in range(self.start_epoch, self.args.epochs):
            train_loss, _ = self._run_epoch(epoch, is_train=True)
            val_loss, val_metrics = self._run_epoch(epoch, is_train=False)
            print(f"\nEpoch {epoch} Train Summary: Loss: {train_loss:.4f}")
            val_metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            print(f"Epoch {epoch} Valid Summary: Loss: {val_loss:.4f} | {val_metric_str}\n")
            current_metric = val_metrics.get('avg_l1_metric', val_loss)
            is_best = current_metric < self.best_val_metric
            if is_best:
                self.best_val_metric = current_metric
            self._save_checkpoint(epoch, is_best)
        print("Predictor training finished.")

    def evaluate(self):
        if not self.test_loader:
            print("No test data loader found. Skipping evaluation.")
            return

        print("\n--- Running Evaluation on Test Set ---")
        self.predictor.eval()
        agg_metrics = defaultdict(float)
        progress_bar = tqdm(self.test_loader, desc="Evaluating on Test Set")

        with torch.no_grad():
            for batch in progress_bar:
                _, metrics = self._compute_loss_and_metrics(batch, is_eval=True)
                for key, val in metrics.items():
                    agg_metrics[key] += val.item()

        eval_metrics = {k: v / len(self.test_loader) for k, v in agg_metrics.items()}

        print("\n--- Test Set Evaluation Results ---")
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.items()])
        print(f"{metric_str}")
        print("-----------------------------------\n")

        results_path = self.args.results_path
        if not results_path:
            results_path = os.path.join(self.args.checkpoint_dir, 'predictor_evaluation_results.json')

        print(f"Saving evaluation results to {results_path}")
        with open(results_path, 'w') as f:
            json.dump(eval_metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Evaluate a Multi-Target Predictor for Diffusion Guidance")
    parser.add_argument('--train', action='store_true', help="Run the training pipeline.")
    parser.add_argument('--evaluate', action='store_true', help="Run the evaluation pipeline on the test set.")

    # --- Paths and Checkpoints ---
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_predictor',
                        help="Directory to save predictor checkpoints.")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Predictor checkpoint to resume training from.")
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints_predictor/predictor_checkpoint_best.pth',
                        help="Predictor checkpoint to use for evaluation.")
    parser.add_argument('--diffusion_checkpoint_path', type=str, default='./checkpoints_diffusion/checkpoint_best.pth',
                        help="Path to the pre-trained diffusion model checkpoint.")
    parser.add_argument('--results_path', type=str, default=None, help="Path to save evaluation results JSON file.")

    # --- Model Architecture ---
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=2000)
    parser.add_argument('--pred_hidden_dim', type=int, default=256)
    parser.add_argument('--pred_num_layers', type=int, default=4)

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if not args.train and not args.evaluate:
        parser.error("Must specify at least one mode: --train or --evaluate")

    trainer = PredictorTrainer(args)
    if args.train:
        trainer.train()
    if args.evaluate:
        trainer.evaluate()
