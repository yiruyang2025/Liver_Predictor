## data_loader.py
# JSON → feature vector, inductive bias clarification = assumptions that guide learning beyond the data itself

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path

class DonorDataset(Dataset):
 def __init__(self,json_files,schema_path,normalize=True):
  self.json_files=json_files if isinstance(json_files,list) else [json_files]
  with open(schema_path,'r') as f:
   self.schema=json.load(f)
  self.normalize=normalize
  self.donors=[]
  self.labels=[]
  self.feature_names=[]
  self._load_data()
  if self.normalize:
   self._compute_statistics()

 def _load_data(self):
  for json_file in self.json_files:
   with open(json_file,'r') as f:
    donor=json.load(f)
   features=self._extract_features(donor)
   if features is not None:
    self.donors.append(features)
    label=donor.get('target',{}).get('liver_transplantability','NTX')
    self.labels.append(1 if label=='TX' else 0)

 def _extract_features(self,donor):
  features=[]
  feature_names=[]
  try:
   main=donor.get('Main',{})
   features.extend([
    self._safe_get(main,'AGE',0,extract_number=True),
    1 if main.get('GENDER')=='M' else 0,
    self._safe_get(main,'HEIGHT',170),
    self._safe_get(main,'WEIGHT',70),
    self._safe_get(main,'BMI',25),
    self._safe_get(main,'WAIST_CIRCUMFERENCE',90)
   ])
   feature_names.extend(['age','gender_male','height','weight','bmi','waist_circ'])
   blood_group=main.get('BLOOD_GROUP','O')
   abo_map={'O':0,'A':1,'B':2,'AB':3}
   features.append(abo_map.get(blood_group,0))
   rh_value=1 if main.get('RHESUS')=='+' else 0
   features.append(rh_value)
   feature_names.extend(['abo_code','rh_positive'])
   adm_circ=donor.get('AdmCircumstances',{})
   features.extend([
    1 if adm_circ.get('CARDIAC_ARREST')=='1' else 0,
    self._safe_get(adm_circ,'CARDIAC_ARREST_DURATION',0),
    self._safe_get(adm_circ,'REANIMATION_DURATION',0),
    self._safe_get(adm_circ,'GLASGOW_SCORE',3),
    1 if adm_circ.get('RESPIRE_ARREST')=='1' else 0
   ])
   feature_names.extend(['cardiac_arrest','ca_duration','reanimation_dur','glasgow','resp_arrest'])
   deathcert=donor.get('Deathcert',{})
   dcd_type_map={'I':1,'II':2,'III':3}
   dcd_type=deathcert.get('NHBD','III')
   features.append(dcd_type_map.get(dcd_type,3))
   features.append(1 if deathcert.get('CONSENTED')else 0)
   feature_names.extend(['dcd_type_code','consented'])
   serology=donor.get('Serology',{})
   serology_fields=['ANTI_HIV1','HBSAG','ANTI_HCV','ANTI_EBVIGG','ANTI_CMVIGG','SEROLOGY_HTLV_1_2','SEROLOGY_HSVIGGG','SEROLOGY_HZVIGG','SEROLOGY_TREPPALL']

   for field in serology_fields:
    val=1 if serology.get(field,'+')=='+' else 0
    features.append(val)
    feature_names.append(field.lower())
   hla=donor.get('Hla',{})
   hla_fields=['HLA_AÏ0','HLA_BÏ0','HLA_CWÏ0','HLA_DRÏ0','HLA_DQÏ0']
 
   for field in hla_fields:
    val=self._safe_get(hla,field,0,extract_number=True)
    features.append(val)
    feature_names.append(field.lower())
   abd=donor.get('Abdominal',{})
   features.extend([
    self._safe_get(abd,'LIVER_SIZE',20),
    self._safe_get(abd,'LIVER_STEATOSIS',0),
    self._safe_get(abd,'SPLENOMEGALY',11),
    self._safe_get(abd,'PANCREAS_SIZE',14),
    self._safe_get(abd,'KIDNEY_LEFT_SIZE_CM',10),
    self._safe_get(abd,'KIDNEY_RIGHT_SIZE_CM',9)
   ])
   feature_names.extend(['liver_size','liver_steatosis','splenomegaly','pancreas_size','kidney_left','kidney_right'])
   heart=donor.get('Heart',{})
   features.append(self._safe_get(heart,'ECHOCARDIOGRAPHY_EF',58))
   feature_names.append('ejection_fraction')
   blood_labs=donor.get('BloodResults',{})
   lab_fields={
    'HB':'hemoglobin',
    'HCT':'hematocrit',
    'PLATELETS':'platelets',
    'GLUCOSE':'glucose',
    'CREATININE':'creatinine',
    'UREA':'urea',
    'ASAT':'ast',
    'ALAT':'alt',
    'LDH':'ldh',
    'BILIRUBIN_TOT':'bilirubin',
    'INR':'inr',
    'FIBRINOGEN':'fibrinogen'
   }

   for lab_key,lab_name in lab_fields.items():
    val=self._safe_get(blood_labs,lab_key,0)
    features.append(val)
    feature_names.append(lab_name)
   vital=donor.get('VitalSigns',{})

   if isinstance(vital,list) and len(vital)>0:
    vital=vital[-1]
   features.extend([
    self._safe_get(vital,'HEART_RATE',100),
    self._safe_get(vital,'BLOOD_PRESSURE_SYS',120),
    self._safe_get(vital,'BLOOD_PRESSURE_DIAS',70),
    self._safe_get(vital,'TEMPERATURE',37),
    self._safe_get(vital,'URINE_OUTPUT_ML_H',100)
   ])
   feature_names.extend(['hr','sys_bp','dias_bp','temp','urine_output'])
   blood_gas=donor.get('BloodGases',{})

   if isinstance(blood_gas,list) and len(blood_gas)>0:
    blood_gas=blood_gas[-1]
   features.extend([
    self._safe_get(blood_gas,'PH',7.35),
    self._safe_get(blood_gas,'PACO2',40),
    self._safe_get(blood_gas,'PAO2',80),
    self._safe_get(blood_gas,'HCO3',24)
   ])
   feature_names.extend(['ph','paco2','pao2','hco3'])
   transp=donor.get('Transplantation',{})
   liver_status=transp.get('LIVER_STATUS','NTX')
   features.append(1 if liver_status=='TX' else 0)
   eff_ischemic=self._safe_get(transp,'LIVER_EFFECTIVE_ISCHEMIC_TIME',12)
   features.append(eff_ischemic)
   feature_names.extend(['liver_tx_planned','ischemic_time'])

   if len(self.feature_names)==0:
    self.feature_names=feature_names
   return np.array(features,dtype=np.float32)
 
  except Exception as e:
   print(f"Error extracting features: {e}")
   return None

 def _safe_get(self,obj,key,default,extract_number=False):
  val=obj.get(key,default) if isinstance(obj,dict) else default
  if val is None:return default
  if extract_number:
   try:
    import re
    match=re.search(r'\d+',str(val))
    return float(match.group()) if match else float(default)
   except:return float(default)
  try:return float(val)
  except:return float(default)

 def _compute_statistics(self):
  donors_array=np.array(self.donors)
  self.mean=np.nanmean(donors_array,axis=0)
  self.std=np.nanstd(donors_array,axis=0)
  self.std[self.std==0]=1
  for i in range(len(self.donors)):
   self.donors[i]=(self.donors[i]-self.mean)/self.std

 def __len__(self):
  return len(self.donors)

 def __getitem__(self,idx):
  return torch.FloatTensor(self.donors[idx]),torch.LongTensor([self.labels[idx]])

def create_dataloaders(json_files,schema_path,batch_size=8,train_ratio=0.7,val_ratio=0.15,seed=42):
 dataset=DonorDataset(json_files,schema_path,normalize=True)
 n=len(dataset)
 n_train=int(n*train_ratio)
 n_val=int(n*val_ratio)
 n_test=n-n_train-n_val
 indices=np.arange(n)
 np.random.seed(seed)
 np.random.shuffle(indices)
 train_indices=indices[:n_train]
 val_indices=indices[n_train:n_train+n_val]
 test_indices=indices[n_train+n_val:]

 from torch.utils.data import Subset

 train_set=Subset(dataset,train_indices)
 val_set=Subset(dataset,val_indices)
 test_set=Subset(dataset,test_indices)
 train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
 val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=False)
 test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False)

 return train_loader,val_loader,test_loader,dataset.feature_names
