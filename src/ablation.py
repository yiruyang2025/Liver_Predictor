import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import argparse
from data_loader import DonorDataset,create_dataloaders
from ssl_encoder import SSLEncoder
from classifier import TransplantabilityClassifier
from train_classifier import ClassifierTrainer
from sklearn.metrics import accuracy_score,sensitivity_score,specificity_score,precision_score,f1_score,roc_auc_score,confusion_matrix
class MinimalMLP(nn.Module):
 def __init__(self,input_dim,num_classes=2,hidden_dim=64,num_layers=1,dropout=0.3,l2_weight=1e-4):
  super().__init__()
  self.l2_weight=l2_weight
  layers=[]
  prev_dim=input_dim
  for i in range(num_layers):
   layers.append(nn.Linear(prev_dim,hidden_dim))
   layers.append(nn.ReLU())
   layers.append(nn.Dropout(dropout))
   prev_dim=hidden_dim
  layers.append(nn.Linear(prev_dim,num_classes))
  self.net=nn.Sequential(*layers)
 def forward(self,x):
  return self.net(x)
 def get_l2_loss(self):
  l2_loss=0
  for param in self.parameters():
   l2_loss+=torch.sum(param**2)
  return self.l2_weight*l2_loss
class AblationStudy:
 def __init__(self,device,output_dir):
  self.device=device
  self.output_dir=Path(output_dir)
  self.output_dir.mkdir(parents=True,exist_ok=True)
  self.results={}
 def run_full_study(self,json_files,schema_path,epochs=100,batch_size=8):
  dataset=DonorDataset(json_files,schema_path,normalize=True)
  n=len(dataset)
  indices=np.arange(n)
  np.random.shuffle(indices)
  configs={
   'baseline_minimal':{
    'model':'minimal_mlp',
    'hidden_dim':32,
    'num_layers':1,
    'dropout':0.3,
    'l2_weight':1e-4,
    'description':'Minimal MLP: 1 layer, 32 hidden, L2=1e-4 (BASELINE)'
   },
   'ssl_frozen':{
    'model':'ssl_classifier',
    'freeze_encoder':True,
    'hidden_dim':128,
    'description':'SSL encoder frozen (feature extractor only)'
   },
   'ssl_finetuned':{
    'model':'ssl_classifier',
    'freeze_encoder':False,
    'hidden_dim':128,
    'description':'SSL encoder fine-tuned'
   },
   'minimal_no_dropout':{
    'model':'minimal_mlp',
    'hidden_dim':32,
    'num_layers':1,
    'dropout':0.0,
    'l2_weight':1e-4,
    'description':'Minimal MLP without dropout'
   },
   'minimal_strong_l2':{
    'model':'minimal_mlp',
    'hidden_dim':32,
    'num_layers':1,
    'dropout':0.3,
    'l2_weight':1e-3,
    'description':'Minimal MLP with strong L2=1e-3'
   },
   'minimal_2layer':{
    'model':'minimal_mlp',
    'hidden_dim':64,
    'num_layers':2,
    'dropout':0.3,
    'l2_weight':1e-4,
    'description':'2-layer MLP (64,64 hidden)'
   },
   'minimal_64hidden':{
    'model':'minimal_mlp',
    'hidden_dim':64,
    'num_layers':1,
    'dropout':0.3,
    'l2_weight':1e-4,
    'description':'Minimal MLP with 64 hidden (vs 32)'
   }
  }
  for config_name,config in configs.items():
   print(f"\n{'='*60}")
   print(f"Running: {config_name}")
   print(f"Description: {config['description']}")
   print(f"{'='*60}")
   metrics=self._run_loocv_config(
    dataset,
    config,
    epochs=epochs,
    batch_size=batch_size
   )
   self.results[config_name]=metrics
  self._save_results()
  self._print_summary()
 def _run_loocv_config(self,dataset,config,epochs,batch_size):
  n=len(dataset)
  all_preds=[]
  all_labels=[]
  all_probs=[]
  for i in range(n):
   from torch.utils.data import Subset,DataLoader
   train_indices=list(range(n))
   train_indices.remove(i)
   test_x,test_y=dataset[i]
   test_x=test_x.unsqueeze(0).to(self.device)
   test_y=test_y.item()
   train_set=Subset(dataset,train_indices)
   train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
   if config['model']=='minimal_mlp':
    model=MinimalMLP(
     input_dim=60,
     hidden_dim=config['hidden_dim'],
     num_layers=config['num_layers'],
     dropout=config['dropout'],
     l2_weight=config['l2_weight']
    ).to(self.device)
   else:
    encoder=SSLEncoder(60,[512,256,128],128)
    model=TransplantabilityClassifier(
     encoder,
     freeze_encoder=config['freeze_encoder']
    ).to(self.device)
   optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
   criterion=nn.CrossEntropyLoss()
   model.train()
   for epoch in range(epochs):
    for batch_x,batch_y in train_loader:
     batch_x,batch_y=batch_x.to(self.device),batch_y.to(self.device).squeeze()
     optimizer.zero_grad()
     logits=model(batch_x)
     loss=criterion(logits,batch_y)
     if config['model']=='minimal_mlp':
      loss+=model.get_l2_loss()
     loss.backward()
     optimizer.step()
   model.eval()
   with torch.no_grad():
    logits=model(test_x)
    probs=torch.softmax(logits,dim=1)
    pred=torch.argmax(logits,dim=1).item()
   all_preds.append(pred)
   all_labels.append(test_y)
   all_probs.append(probs[0,1].item())
  all_preds=np.array(all_preds)
  all_labels=np.array(all_labels)
  all_probs=np.array(all_probs)
  metrics={
   'accuracy':accuracy_score(all_labels,all_preds),
   'sensitivity':sensitivity_score(all_labels,all_preds,average='binary',zero_division=0),
   'specificity':specificity_score(all_labels,all_preds,average='binary',zero_division=0),
   'precision':precision_score(all_labels,all_preds,average='binary',zero_division=0),
   'f1':f1_score(all_labels,all_preds,average='binary',zero_division=0)
  }
  if len(np.unique(all_labels))>1:
   metrics['auc_roc']=roc_auc_score(all_labels,all_probs)
  else:
   metrics['auc_roc']=np.nan
  tn,fp,fn,tp=confusion_matrix(all_labels,all_preds).ravel()
  metrics['tn'],metrics['fp'],metrics['fn'],metrics['tp']=int(tn),int(fp),int(fn),int(tp)
  return metrics
 def _save_results(self):
  output_path=self.output_dir/'ablation_results.json'
  with open(output_path,'w') as f:
   json.dump(self.results,f,indent=2)
  print(f"\nResults saved to {output_path}")
 def _print_summary(self):
  print(f"\n{'='*80}")
  print("ABLATION STUDY SUMMARY")
  print(f"{'='*80}")
  print(f"{'Config':<25} {'Accuracy':<12} {'AUC-ROC':<12} {'F1':<12}")
  print(f"{'-'*80}")
  for config_name,metrics in self.results.items():
   acc=metrics['accuracy']
   auc=metrics.get('auc_roc','N/A')
   f1=metrics['f1']
   if isinstance(auc,float):
    print(f"{config_name:<25} {acc:.4f}        {auc:.4f}        {f1:.4f}")
   else:
    print(f"{config_name:<25} {acc:.4f}        {auc:<12} {f1:.4f}")
  print(f"{'='*80}")
  best_config=max(self.results.items(),key=lambda x:x[1]['accuracy'])
  print(f"\nBest performer: {best_config[0]} (Accuracy: {best_config[1]['accuracy']:.4f})")
def main():
 parser=argparse.ArgumentParser(description='Ablation Study: Minimal Models vs SSL')
 parser.add_argument('--json_files',type=str,nargs='+',required=True,help='JSON donor files')
 parser.add_argument('--schema_path',type=str,required=True,help='Schema path')
 parser.add_argument('--output_dir',type=str,default='ablation_results')
 parser.add_argument('--epochs',type=int,default=100)
 parser.add_argument('--batch_size',type=int,default=8)
 parser.add_argument('--device',type=str,default='mps' if torch.backends.mps.is_available() else 'cpu')
 args=parser.parse_args()
 device=torch.device(args.device)
 print(f"Using device: {device}")
 ablation=AblationStudy(device,args.output_dir)
 ablation.run_full_study(args.json_files,args.schema_path,args.epochs,args.batch_size)
if __name__=='__main__':
 main()
