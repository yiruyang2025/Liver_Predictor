import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import argparse
from sklearn.metrics import accuracy_score,sensitivity_score,specificity_score,precision_score,f1_score,roc_auc_score,confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt
from data_loader import DonorDataset
from ssl_encoder import SSLEncoder
from classifier import TransplantabilityClassifier

class LOOCVEvaluator:
 def __init__(self,model,device):
  self.model=model.to(device)
  self.device=device
  self.model.eval()
  
 def evaluate_loocv(self,dataset):
  n=len(dataset)
  predictions=[]
  ground_truth=[]
  probabilities=[]
  
  for i in range(n):
   train_indices=list(range(n))
   train_indices.remove(i)
   test_x,test_y=dataset[i]
   test_x=test_x.unsqueeze(0).to(self.device)
   test_y=test_y.item()
   
   with torch.no_grad():
    logits=self.model(test_x)
    probs=torch.softmax(logits,dim=1)
   pred=torch.argmax(logits,dim=1).item()
   predictions.append(pred)
   ground_truth.append(test_y)
   probabilities.append(probs[0,1].item())
   
  return np.array(predictions),np.array(ground_truth),np.array(probabilities)

 def compute_metrics(self,predictions,ground_truth,probabilities):
  accuracy=accuracy_score(ground_truth,predictions)
  sensitivity=sensitivity_score(ground_truth,predictions,average='binary',zero_division=0)
  specificity=specificity_score(ground_truth,predictions,average='binary',zero_division=0)
  precision=precision_score(ground_truth,predictions,average='binary',zero_division=0)
  f1=f1_score(ground_truth,predictions,average='binary',zero_division=0)
 
  if len(np.unique(ground_truth))>1:
   auc_roc=roc_auc_score(ground_truth,probabilities)
  else:
   auc_roc=np.nan
  tn,fp,fn,tp=confusion_matrix(ground_truth,predictions).ravel()
  metrics={
   'accuracy':accuracy,
   'sensitivity':sensitivity,
   'specificity':specificity,
   'precision':precision,
   'f1':f1,
   'auc_roc':auc_roc,
   'true_negatives':int(tn),
   'false_positives':int(fp),
   'false_negatives':int(fn),
   'true_positives':int(tp)
  }
  return metrics

 def plot_roc_curve(self,ground_truth,probabilities,output_path):
  if len(np.unique(ground_truth))<2:
   print("Cannot plot ROC curve with single class")
   return
  fpr,tpr,_=roc_curve(ground_truth,probabilities)
  roc_auc=auc(fpr,tpr)
  plt.figure()
  plt.plot(fpr,tpr,color='darkorange',lw=2,label=f'ROC curve (AUC = {roc_auc:.2f})')
  plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
  plt.xlim([0.0,1.0])
  plt.ylim([0.0,1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve - Liver Transplantability')
  plt.legend(loc="lower right")
  plt.savefig(output_path)
  plt.close()

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--json_files',type=str,nargs='+',required=True)
 parser.add_argument('--schema_path',type=str,required=True)
 parser.add_argument('--classifier_path',type=str,required=True)
 parser.add_argument('--output_dir',type=str,default='./results')
 parser.add_argument('--input_dim',type=int,default=60)
 parser.add_argument('--encoder_output_dim',type=int,default=128)
 parser.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')

 args=parser.parse_args()
 device=torch.device(args.device)
 output_dir=Path(args.output_dir)
 output_dir.mkdir(parents=True,exist_ok=True)
 dataset=DonorDataset(args.json_files,args.schema_path,normalize=True)
 encoder=SSLEncoder(args.input_dim,hidden_dims=[512,256,128],output_dim=args.encoder_output_dim)
 model=TransplantabilityClassifier(encoder,encoder_output_dim=args.encoder_output_dim)
 model.load_state_dict(torch.load(args.classifier_path,map_location=device))
 model=model.to(device)

 evaluator=LOOCVEvaluator(model,device)
 predictions,ground_truth,probabilities=evaluator.evaluate_loocv(dataset)
 metrics=evaluator.compute_metrics(predictions,ground_truth,probabilities)

 print("\nLOOCV Evaluation Results:")
 print(f"Accuracy: {metrics['accuracy']:.4f}")
 print(f"Sensitivity: {metrics['sensitivity']:.4f}")
 print(f"Specificity: {metrics['specificity']:.4f}")
 print(f"Precision: {metrics['precision']:.4f}")
 print(f"F1-Score: {metrics['f1']:.4f}")
 print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
 print(f"Confusion Matrix: TP={metrics['true_positives']} FP={metrics['false_positives']} FN={metrics['false_negatives']} TN={metrics['true_negatives']}")

 with open(output_dir/'metrics.json','w') as f:
  json.dump(metrics,f,indent=2)
 evaluator.plot_roc_curve(ground_truth,probabilities,output_dir/'roc_curve.png')
  print(f"\nResults saved to {output_dir}")

if __name__=='__main__':
 main()
