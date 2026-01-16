import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional
from data_loader import DonorDataset
from ssl_encoder import SSLEncoder
from classifier import TransplantabilityClassifier
from sklearn.metrics import accuracy_score, sensitivity_score, specificity_score
from sklearn.metrics import precision_score, f1_score, roc_auc_score, confusion_matrix

class MinimalMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2, hidden_dim: int = 64,
                 num_layers: int = 1, dropout: float = 0.3, l2_weight: float = 1e-4):
        super().__init__()
        self.l2_weight = l2_weight
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
   
    def get_l2_loss(self) -> torch.Tensor:
        l2_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_weight * l2_loss

class AblationStudy:
    def __init__(self, device: torch.device, output_dir: str):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.configurations = self._define_configurations()
  
    def _define_configurations(self) -> Dict:
        return {
            'baseline_minimal': {
                'model': 'minimal_mlp',
                'hidden_dim': 32,
                'num_layers': 1,
                'dropout': 0.3,
                'l2_weight': 1e-4,
                'description': 'Minimal MLP baseline: single hidden layer (32 units), L2=1e-4, dropout=0.3'
            },
            'minimal_strong_l2': {
                'model': 'minimal_mlp',
                'hidden_dim': 32,
                'num_layers': 1,
                'dropout': 0.3,
                'l2_weight': 1e-3,
                'description': 'Minimal MLP with strong L2 regularization: L2=1e-3'
            },
            'minimal_no_dropout': {
                'model': 'minimal_mlp',
                'hidden_dim': 32,
                'num_layers': 1,
                'dropout': 0.0,
                'l2_weight': 1e-4,
                'description': 'Minimal MLP without dropout: tests regularization necessity'
            },
            'minimal_2layer': {
                'model': 'minimal_mlp',
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.3,
                'l2_weight': 1e-4,
                'description': 'Two-layer MLP: 64 hidden units per layer'
            },
            'minimal_64hidden': {
                'model': 'minimal_mlp',
                'hidden_dim': 64,
                'num_layers': 1,
                'dropout': 0.3,
                'l2_weight': 1e-4,
                'description': 'Minimal MLP with increased capacity: 64 hidden units'
            },
            'ssl_frozen': {
                'model': 'ssl_classifier',
                'freeze_encoder': True,
                'hidden_dim': 128,
                'description': 'SSL encoder frozen as feature extractor'
            },
            'ssl_finetuned': {
                'model': 'ssl_classifier',
                'freeze_encoder': False,
                'hidden_dim': 128,
                'description': 'SSL encoder fine-tuned end-to-end'
            }
        }
   
    def run_full_study(self, json_files: List[str], schema_path: str,
                      epochs: int = 100, batch_size: int = 8):
        dataset = DonorDataset(json_files, schema_path, normalize=True)
        n = len(dataset)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        for config_name, config in self.configurations.items():
            print(f"\nEvaluating configuration: {config_name}")
            print(f"Description: {config['description']}")
            metrics = self._evaluate_loocv(dataset, config, epochs, batch_size)
            self.results[config_name] = metrics
        self._save_results()
        self._print_final_summary()
  
    def _evaluate_loocv(self, dataset: DonorDataset, config: Dict,
                       epochs: int, batch_size: int) -> Dict:
        n = len(dataset)
        all_predictions = []
        all_labels = []
        all_probabilities = []
        for i in range(n):
            from torch.utils.data import Subset, DataLoader
            train_indices = list(range(n))
            train_indices.remove(i)
            test_x, test_y = dataset[i]
            test_x = test_x.unsqueeze(0).to(self.device)
            test_y = test_y.item()
            train_set = Subset(dataset, train_indices)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            if config['model'] == 'minimal_mlp':
                model = MinimalMLP(
                    input_dim=60,
                    hidden_dim=config['hidden_dim'],
                    num_layers=config['num_layers'],
                    dropout=config['dropout'],
                    l2_weight=config['l2_weight']
                ).to(self.device)
            else:
                encoder = SSLEncoder(60, [512, 256, 128], 128)
                model = TransplantabilityClassifier(
                    encoder,
                    freeze_encoder=config['freeze_encoder']
                ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            model.train()
            for epoch in range(epochs):
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device).squeeze()
                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    if config['model'] == 'minimal_mlp':
                        loss += model.get_l2_loss()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                logits = model(test_x)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1).item()
            all_predictions.append(pred)
            all_labels.append(test_y)
            all_probabilities.append(probs[0, 1].item())
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        metrics = self._compute_metrics(all_predictions, all_labels, all_probabilities)
        return metrics
 
    def _compute_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray,
                        probabilities: np.ndarray) -> Dict:
        metrics = {
            'accuracy': float(accuracy_score(ground_truth, predictions)),
            'sensitivity': float(sensitivity_score(ground_truth, predictions, average='binary', zero_division=0)),
            'specificity': float(specificity_score(ground_truth, predictions, average='binary', zero_division=0)),
            'precision': float(precision_score(ground_truth, predictions, average='binary', zero_division=0)),
            'f1': float(f1_score(ground_truth, predictions, average='binary', zero_division=0))
        }
   
        if len(np.unique(ground_truth)) > 1:
            metrics['auc_roc'] = float(roc_auc_score(ground_truth, probabilities))
        else:
            metrics['auc_roc'] = np.nan
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        return metrics
   
    def _save_results(self):
        output_path = self.output_dir / 'ablation_results.json'
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\nResults saved to {output_path}")
 
    def _print_final_summary(self):
        print("\n" + "=" * 130)
        print("ABLATION STUDY SUMMARY - LEAVE-ONE-OUT CROSS-VALIDATION RESULTS")
        print("=" * 130)
        print(f"\n{'Configuration':<30} {'Accuracy':<12} {'AUC-ROC':<12} {'Sensitivity':<12} {'Specificity':<12} {'F1':<12}")
        print("-" * 130)
        best_config = None
        best_accuracy = 0.0
        for config_name, metrics in self.results.items():
            acc = metrics['accuracy']
            auc = f"{metrics['auc_roc']:.4f}" if isinstance(metrics['auc_roc'], float) else 'N/A'
            sens = f"{metrics['sensitivity']:.4f}"
            spec = f"{metrics['specificity']:.4f}"
            f1 = f"{metrics['f1']:.4f}"
            print(f"{config_name:<30} {acc:<12.4f} {auc:<12} {sens:<12} {spec:<12} {f1:<12}")
            if acc > best_accuracy:
                best_accuracy = acc
                best_config = config_name
        print("-" * 130)
        print(f"\nBest performing configuration: {best_config} (accuracy: {best_accuracy:.4f})")
        print("=" * 130)

def main():
    parser = argparse.ArgumentParser(
        description='Ablation study for liver transplantability prediction model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python ablation.py --json_files data/example_donor.json --schema_path data/schema.json --device mps'
    )
    parser.add_argument('--json_files', type=str, nargs='+', required=True, help='JSON donor data files')
    parser.add_argument('--schema_path', type=str, required=True, help='Path to data schema JSON')
    parser.add_argument('--output_dir', type=str, default='ablation_results', help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs per LOOCV fold')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu',
                       help='Device for computation: mps, cuda, or cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")
    ablation = AblationStudy(device, args.output_dir)
    ablation.run_full_study(args.json_files, args.schema_path, args.epochs, args.batch_size)

if __name__ == '__main__':
    main()
