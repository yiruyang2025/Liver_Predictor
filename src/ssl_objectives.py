# define model with contrastive / masking / distillation

import torch
import torch.nn.functional as F
def nt_xent_loss(z_i,z_j,temperature=0.07):
 batch_size=z_i.shape[0]
 z_i=F.normalize(z_i,dim=1)
 z_j=F.normalize(z_j,dim=1)
 z=torch.cat([z_i,z_j],dim=0)
 similarity_matrix=torch.matmul(z,z.T)/temperature
 mask=torch.eye(2*batch_size,dtype=torch.bool,device=z.device)
 similarity_matrix.masked_fill_(mask,float('-inf'))
 pos_mask=torch.zeros(2*batch_size,2*batch_size,dtype=torch.bool,device=z.device)
 for i in range(batch_size):
  pos_mask[i,batch_size+i]=True
  pos_mask[batch_size+i,i]=True
 pos_logits=similarity_matrix[pos_mask].view(2*batch_size,1)
 neg_logits=similarity_matrix[~mask].view(2*batch_size,-1)
 logits=torch.cat([pos_logits,neg_logits],dim=1)
 labels=torch.zeros(2*batch_size,dtype=torch.long,device=z.device)
 loss=F.cross_entropy(logits,labels)
 return loss
def masking_loss(x_original,x_reconstructed,mask):
 masked_original=x_original*mask
 masked_reconstructed=x_reconstructed*mask
 loss=F.mse_loss(masked_original,masked_reconstructed)
 return loss
class MaskingPretrainingObjective:
 def __init__(self,mask_ratio=0.3):
  self.mask_ratio=mask_ratio
 def __call__(self,x,model):
  batch_size=x.shape[0]
  feature_dim=x.shape[1]
  mask=torch.bernoulli(torch.ones(batch_size,feature_dim,device=x.device)*(1-self.mask_ratio))
  x_masked=x*mask
  x_reconstructed=model(x_masked)
  loss=masking_loss(x,x_reconstructed,mask)
  return loss
class ContrastivePretrainingObjective:
 def __init__(self,temperature=0.07,augmentation_fn=None):
  self.temperature=temperature
  self.augmentation_fn=augmentation_fn
 def __call__(self,x,model):
  if self.augmentation_fn is None:
   x_i,x_j=self._default_augment(x)
  else:
   x_i,x_j=self.augmentation_fn(x),self.augmentation_fn(x)
  z_i=model(x_i)
  z_j=model(x_j)
  loss=nt_xent_loss(z_i,z_j,temperature=self.temperature)
  return loss
 def _default_augment(self,x):
  noise=torch.randn_like(x)*0.1
  x_i=x+noise
  x_j=x+torch.randn_like(x)*0.1
  return x_i,x_j
class HybridObjective:
 def __init__(self,contrastive_weight=0.5,masking_weight=0.5,temperature=0.07,mask_ratio=0.3):
  self.contrastive_objective=ContrastivePretrainingObjective(temperature)
  self.masking_objective=MaskingPretrainingObjective(mask_ratio)
  self.contrastive_weight=contrastive_weight
  self.masking_weight=masking_weight
 def __call__(self,x,model):
  loss_contrastive=self.contrastive_objective(x,model)
  loss_masking=self.masking_objective(x,model)
  total_loss=self.contrastive_weight*loss_contrastive+self.masking_weight*loss_masking
  return total_loss,loss_contrastive,loss_masking
