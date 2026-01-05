import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import argparse
from data_loader import create_dataloaders
from ssl_encoder import create_ssl_encoder_with_projection
from ssl_objectives import ContrastivePretrainingObjective,HybridObjective
class SSLTrainer:
 def __init__(self,model,device,output_dir):
  self.model=model.to(device)
  self.device=device
  self.output_dir=Path(output_dir)
  self.output_dir.mkdir(parents=True,exist_ok=True)
  self.history={'loss':[],'val_loss':[]}
 def train(self,train_loader,val_loader,objective,optimizer,scheduler,epochs=100):
  best_loss=float('inf')
  patience=10
  patience_counter=0
  for epoch in range(epochs):
   self.model.train()
   train_loss=0.0
   for batch_idx,batch in enumerate(train_loader):
    x,_=batch
    x=x.to(self.device)
    optimizer.zero_grad()
    if isinstance(objective,HybridObjective):
     total_loss,loss_contrastive,loss_masking=objective(x,self.model)
    else:
     total_loss=objective(x,self.model)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
    optimizer.step()
    train_loss+=total_loss.item()
   scheduler.step()
   train_loss/=len(train_loader)
   val_loss=self._validate(val_loader,objective)
   self.history['loss'].append(train_loss)
   self.history['val_loss'].append(val_loss)
   print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
   if val_loss<best_loss:
    best_loss=val_loss
    patience_counter=0
    self._save_checkpoint('best_encoder.pt')
   else:
    patience_counter+=1
    if patience_counter>=patience:
     print(f"Early stopping at epoch {epoch+1}")
     break
  self._save_checkpoint('final_encoder.pt')
  return self.history
 def _validate(self,val_loader,objective):
  self.model.eval()
  val_loss=0.0
  with torch.no_grad():
   for batch in val_loader:
    x,_=batch
    x=x.to(self.device)
    if isinstance(objective,HybridObjective):
     total_loss,_,_=objective(x,self.model)
    else:
     total_loss=objective(x,self.model)
    val_loss+=total_loss.item()
  return val_loss/len(val_loader)
 def _save_checkpoint(self,filename):
  path=self.output_dir/filename
  torch.save(self.model.state_dict(),path)
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--json_files',type=str,nargs='+',required=True)
 parser.add_argument('--schema_path',type=str,required=True)
 parser.add_argument('--output_dir',type=str,default='./checkpoints/ssl')
 parser.add_argument('--input_dim',type=int,default=60)
 parser.add_argument('--output_dim',type=int,default=128)
 parser.add_argument('--projection_dim',type=int,default=64)
 parser.add_argument('--batch_size',type=int,default=32)
 parser.add_argument('--epochs',type=int,default=100)
 parser.add_argument('--lr',type=float,default=1e-3)
 parser.add_argument('--objective',type=str,choices=['contrastive','hybrid'],default='contrastive')
 parser.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
 args=parser.parse_args()
 device=torch.device(args.device)
 train_loader,val_loader,test_loader,feature_names=create_dataloaders(
  args.json_files,
  args.schema_path,
  batch_size=args.batch_size
 )
 model=create_ssl_encoder_with_projection(
  args.input_dim,
  hidden_dims=[512,256,128],
  output_dim=args.output_dim,
  projection_dim=args.projection_dim
 )
 optimizer=optim.Adam(model.parameters(),lr=args.lr)
 scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
 if args.objective=='contrastive':
  objective=ContrastivePretrainingObjective(temperature=0.07)
 else:
  objective=HybridObjective(contrastive_weight=0.5,masking_weight=0.5)
 trainer=SSLTrainer(model,device,args.output_dir)
 history=trainer.train(train_loader,val_loader,objective,optimizer,scheduler,args.epochs)
 with open(Path(args.output_dir)/'training_history.json','w') as f:
  json.dump(history,f)
 print(f"SSL pretraining complete. Model saved to {args.output_dir}")
if __name__=='__main__':
 main()
