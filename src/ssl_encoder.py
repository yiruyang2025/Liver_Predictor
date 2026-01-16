import torch
import torch.nn as nn
import torch.nn.functional as F

class SSLEncoder(nn.Module):
 def __init__(self,input_dim,hidden_dims=[512,256,128],output_dim=128,dropout=0.1):
  super().__init__()
  layers=[]
  prev_dim=input_dim
  for hidden_dim in hidden_dims:
   layers.append(nn.Linear(prev_dim,hidden_dim))
   layers.append(nn.BatchNorm1d(hidden_dim))
   layers.append(nn.ReLU())
   layers.append(nn.Dropout(dropout))
   prev_dim=hidden_dim
  layers.append(nn.Linear(prev_dim,output_dim))
  self.encoder=nn.Sequential(*layers)
  self.output_dim=output_dim
  self.input_dim=input_dim
 
 def forward(self,x):
  return self.encoder(x)

class SSLEncoderWithProjection(nn.Module):
 def __init__(self,input_dim,hidden_dims=[512,256,128],output_dim=128,projection_dim=64):
  super().__init__()
  self.encoder=SSLEncoder(input_dim,hidden_dims,output_dim)
  self.projection_head=nn.Sequential(
   nn.Linear(output_dim,output_dim),
   nn.ReLU(),
   nn.Linear(output_dim,projection_dim)
  )
  self.output_dim=output_dim
  self.projection_dim=projection_dim

 def forward(self,x,return_projection=False):
  z=self.encoder(x)
  if return_projection:
   p=self.projection_head(z)
   return z,p
  return z
def create_ssl_encoder(input_dim,hidden_dims=[512,256,128],output_dim=128):
 return SSLEncoder(input_dim,hidden_dims,output_dim)
def create_ssl_encoder_with_projection(input_dim,hidden_dims=[512,256,128],output_dim=128,projection_dim=64):
 return SSLEncoderWithProjection(input_dim,hidden_dims,output_dim,projection_dim)
