import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import argparse
from data_loader import create_dataloaders
from ssl_encoder import SSLEncoder
from classifier import TransplantabilityClassifier

class ClassifierTrainer:
 def __init__(self,model,device,output_dir):
  self.model=model.to(device)
  self.device=device
  self.output_dir=Path(output_dir)
  self.output_dir.mkdir(parents=True,exist_ok=True)
  self.history={'loss':[],'acc':[],'val_loss':[],'val_acc':[]}

 def train(self,train_loader,val_loader,optimizer,scheduler,criterion,epochs=100):
  best_val_acc=0.0
  patience=15
  patience_counter=0
  for epoch in range(epochs):
   self.model.train()
   train_loss=0.0
   train_correct=0
   train_total=0
   for batch_idx,(x,y) in enumerate(train_loader):
    x,y=x.to(self.device),y.to(self.device).squeeze()
    optimizer.zero_grad()
    logits=self.model(x)
    loss=criterion(logits,y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
    optimizer.step()
    train_loss+=loss.item()
    pred=torch.argmax(logits,dim=1)
    train_correct+=(pred==y).sum().item()
    train_total+=y.shape[0]
   scheduler.step()
   train_loss/=len(train_loader)
   train_acc=train_correct/train_total
   val_loss,val_acc=self._validate(val_loader,criterion)
   self.history['loss'].append(train_loss)
   self.history['acc'].append(train_acc)
   self.history['val_loss'].append(val_loss)
   self.history['val_acc'].append(val_acc)
   print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
   if val_acc>best_val_acc:
    best_val_acc=val_acc
    patience_counter=0
    self._save_checkpoint('best_classifier.pt')
   else:
    patience_counter+=1
    if patience_counter>=patience:
     print(f"Early stopping at epoch {epoch+1}")
     break
  self._save_checkpoint('final_classifier.pt')
  return self.history

 def _validate(self,val_loader,criterion):
  self.model.eval()
  val_loss=0.0
  val_correct=0
  val_total=0
  with torch.no_grad():
   for x,y in val_loader:
    x,y=x.to(self.device),y.to(self.device).squeeze()
    logits=self.model(x)
    loss=criterion(logits,y)
    val_loss+=loss.item()
    pred=torch.argmax(logits,dim=1)
    val_correct+=(pred==y).sum().item()
    val_total+=y.shape[0]
  return val_loss/len(val_loader),val_correct/val_total

 def _save_checkpoint(self,filename):
  path=self.output_dir/filename
  torch.save(self.model.state_dict(),path)

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--json_files',type=str,nargs='+',required=True)
 parser.add_argument('--schema_path',type=str,required=True)
 parser.add_argument('--pretrained_encoder',type=str,default=None)
 parser.add_argument('--output_dir',type=str,default='./checkpoints/classifier')
 parser.add_argument('--input_dim',type=int,default=60)
 parser.add_argument('--encoder_output_dim',type=int,default=128)
 parser.add_argument('--batch_size',type=int,default=16)
 parser.add_argument('--epochs',type=int,default=100)
 parser.add_argument('--lr',type=float,default=1e-3)
 parser.add_argument('--freeze_encoder',type=bool,default=True)
 parser.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
 args=parser.parse_args()
 device=torch.device(args.device)
 train_loader,val_loader,test_loader,feature_names=create_dataloaders(
  args.json_files,
  args.schema_path,
  batch_size=args.batch_size
 )
 encoder=SSLEncoder(args.input_dim,hidden_dims=[512,256,128],output_dim=args.encoder_output_dim)
 if args.pretrained_encoder:
  encoder.load_state_dict(torch.load(args.pretrained_encoder,map_location=device))
  print(f"Loaded pretrained encoder from {args.pretrained_encoder}")
 model=TransplantabilityClassifier(
  encoder,
  encoder_output_dim=args.encoder_output_dim,
  hidden_dims=[256,128],
  num_classes=2,
  freeze_encoder=args.freeze_encoder
 )
 optimizer=optim.Adam(model.parameters(),lr=args.lr)
 scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
 criterion=nn.CrossEntropyLoss()
 trainer=ClassifierTrainer(model,device,args.output_dir)
 history=trainer.train(train_loader,val_loader,optimizer,scheduler,criterion,args.epochs)
 with open(Path(args.output_dir)/'training_history.json','w') as f:
  json.dump(history,f)
 print(f"Classifier training complete. Model saved to {args.output_dir}")

if __name__=='__main__':
 main()
