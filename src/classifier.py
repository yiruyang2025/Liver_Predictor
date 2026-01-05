# prediction head / projection head / task head

import torch
import torch.nn as nn
import torch.nn.functional as F
class TransplantabilityClassifier(nn.Module):
 def __init__(self,encoder,encoder_output_dim=128,hidden_dims=[256,128],num_classes=2,dropout=0.2,freeze_encoder=False):
  super().__init__()
  self.encoder=encoder
  self.freeze_encoder=freeze_encoder
  if freeze_encoder:
   for param in self.encoder.parameters():
    param.requires_grad=False
  layers=[]
  prev_dim=encoder_output_dim
  for hidden_dim in hidden_dims:
   layers.append(nn.Linear(prev_dim,hidden_dim))
   layers.append(nn.BatchNorm1d(hidden_dim))
   layers.append(nn.ReLU())
   layers.append(nn.Dropout(dropout))
   prev_dim=hidden_dim
  layers.append(nn.Linear(prev_dim,num_classes))
  self.classifier_head=nn.Sequential(*layers)
  self.num_classes=num_classes
 def forward(self,x,return_features=False):
  features=self.encoder(x)
  logits=self.classifier_head(features)
  if return_features:
   return logits,features
  return logits
 def unfreeze_encoder(self):
  for param in self.encoder.parameters():
   param.requires_grad=True
  self.freeze_encoder=False
 def get_features(self,x):
  return self.encoder(x)
class EnsembleClassifier(nn.Module):
 def __init__(self,encoders,encoder_output_dim=128,hidden_dims=[256,128],num_classes=2,dropout=0.2):
  super().__init__()
  self.encoders=nn.ModuleList(encoders)
  self.num_encoders=len(encoders)
  layers=[]
  prev_dim=encoder_output_dim*self.num_encoders
  for hidden_dim in hidden_dims:
   layers.append(nn.Linear(prev_dim,hidden_dim))
   layers.append(nn.BatchNorm1d(hidden_dim))
   layers.append(nn.ReLU())
   layers.append(nn.Dropout(dropout))
   prev_dim=hidden_dim
  layers.append(nn.Linear(prev_dim,num_classes))
  self.classifier_head=nn.Sequential(*layers)
  self.num_classes=num_classes
 def forward(self,x,return_features=False):
  features_list=[]
  for encoder in self.encoders:
   feat=encoder(x)
   features_list.append(feat)
  concatenated_features=torch.cat(features_list,dim=1)
  logits=self.classifier_head(concatenated_features)
  if return_features:
   return logits,concatenated_features
  return logits
def create_classifier(encoder,encoder_output_dim=128,hidden_dims=[256,128],num_classes=2,freeze_encoder=True):
 return TransplantabilityClassifier(encoder,encoder_output_dim,hidden_dims,num_classes,freeze_encoder=freeze_encoder)
def create_ensemble_classifier(encoders,encoder_output_dim=128,hidden_dims=[256,128],num_classes=2):
 return EnsembleClassifier(encoders,encoder_output_dim,hidden_dims,num_classes)
