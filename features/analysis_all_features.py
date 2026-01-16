#!/usr/bin/env python3
"""
Liver Transplantability Prediction - Feature Analysis (All Features)
ICML 2026 Submission
Version 1: Clustering analysis using ALL JSON features
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

class DonorDataAnalyzer:
    """Analyzer for donor medical data with comprehensive feature extraction."""
    
    def __init__(self, json_dir):
        self.json_dir = json_dir
        self.donors_data = []
        self.feature_matrix = None
        self.feature_names = []
        
    def load_all_json(self):
        """Load all JSON files from directory."""
        print(f"Loading JSON files from {self.json_dir}...")
        
        for filename in sorted(os.listdir(self.json_dir)):
            if filename.endswith('.json'):
                filepath = os.path.join(self.json_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.donors_data.append({
                            'filename': filename,
                            'data': data
                        })
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"Loaded {len(self.donors_data)} donor records")
        
    def extract_all_features(self):
        """Extract all available numerical and categorical features."""
        print("Extracting all features from JSON data...")
        
        feature_dicts = []
        
        for donor in self.donors_data:
            features = {}
            data = donor['data']
            filename = donor['filename']
            
            # Extract main demographic features
            if 'Main' in data:
                main = data['Main']
                features['donor_id'] = filename.replace('.json', '')
                
                # Age processing
                if 'AGE' in main and main['AGE']:
                    age_str = str(main['AGE'])
                    try:
                        features['age'] = float(age_str.replace('Y', '').strip())
                    except:
                        features['age'] = np.nan
                else:
                    features['age'] = np.nan
                
                # Anthropometric
                features['height'] = float(main.get('HEIGHT', np.nan)) if main.get('HEIGHT') else np.nan
                features['weight'] = float(main.get('WEIGHT', np.nan)) if main.get('WEIGHT') else np.nan
                features['bmi'] = float(main.get('BMI', np.nan)) if main.get('BMI') else np.nan
                features['waist_circumference'] = float(main.get('WAIST_CIRCUMFERENCE', np.nan)) if main.get('WAIST_CIRCUMFERENCE') else np.nan
                
                # Blood type
                features['blood_group'] = main.get('BLOOD_GROUP', 'Unknown')
                features['rhesus'] = main.get('RHESUS', 'Unknown')
                features['gender'] = main.get('GENDER', 'Unknown')
                
            # Extract hospital admission data
            if 'HospitalAdmission' in data:
                adm = data['HospitalAdmission']
                features['cause_of_admission'] = adm.get('CAUSE_OF_ADMISSION', 'Unknown')
                
                # Cardiac arrest
                if adm.get('CARDIAC_ARREST_DURATION'):
                    try:
                        features['cardiac_arrest_duration'] = float(adm['CARDIAC_ARREST_DURATION'])
                    except:
                        features['cardiac_arrest_duration'] = np.nan
                else:
                    features['cardiac_arrest_duration'] = np.nan
                
                # Respiratory arrest
                if adm.get('RESP_ARREST_DURATION'):
                    try:
                        features['respiratory_arrest_duration'] = float(adm['RESP_ARREST_DURATION'])
                    except:
                        features['respiratory_arrest_duration'] = np.nan
                else:
                    features['respiratory_arrest_duration'] = np.nan
                    
                # Glasgow score
                if adm.get('GLASGOW_SCORE'):
                    try:
                        features['glasgow_score'] = float(adm['GLASGOW_SCORE'])
                    except:
                        features['glasgow_score'] = np.nan
                else:
                    features['glasgow_score'] = np.nan
            
            # Extract laboratory results (most recent values)
            if 'LabBloodList' in data and data['LabBloodList']:
                # Get most recent lab results
                lab_results = data['LabBloodList']
                if isinstance(lab_results, list) and len(lab_results) > 0:
                    most_recent = lab_results[-1]  # Assuming last is most recent
                    
                    # Key liver function markers
                    lab_fields = {
                        'ASAT': 'asat',
                        'ALAT': 'alat',
                        'LDH': 'ldh',
                        'GGT': 'ggt',
                        'ALC_PHOSPHATASE': 'alc_phos',
                        'BILIRUBIN_TOT': 'bili_tot',
                        'BILIRUBIN_DIR': 'bili_dir',
                        'ALBUMIN': 'albumin',
                        'TOTAL_PROTEIN': 'total_protein',
                        'AMMONIUM': 'ammonium',
                        'INR': 'inr',
                        'QUICK_PT': 'quick_pt',
                        'APTT': 'aptt',
                        'FIBRINOGEN': 'fibrinogen',
                        'FACTOR_V': 'factor_v',
                        'CREATININE': 'creatinine',
                        'UREA': 'urea',
                        'SODIUM': 'sodium',
                        'POTASSIUM': 'potassium',
                        'GLUCOSE': 'glucose',
                        'CRP': 'crp',
                        'LEUCOCYTES': 'leucocytes',
                        'PLATELETS': 'platelets',
                        'HB': 'hemoglobin',
                        'HCT': 'hematocrit'
                    }
                    
                    for json_field, feature_name in lab_fields.items():
                        if json_field in most_recent and most_recent[json_field]:
                            try:
                                features[feature_name] = float(most_recent[json_field])
                            except:
                                features[feature_name] = np.nan
                        else:
                            features[feature_name] = np.nan
            
            # Extract vital signs (most recent)
            if 'VitalSignsList' in data and data['VitalSignsList']:
                vitals = data['VitalSignsList']
                if isinstance(vitals, list) and len(vitals) > 0:
                    most_recent_vitals = vitals[-1]
                    
                    vital_fields = {
                        'HEART_RATE': 'heart_rate',
                        'BP_SYSTOLIC': 'bp_sys',
                        'BP_DIASTOLIC': 'bp_dias',
                        'BP_MEAN': 'bp_mean',
                        'TEMPERATURE': 'temperature',
                        'URINE_OUTPUT': 'urine_output'
                    }
                    
                    for json_field, feature_name in vital_fields.items():
                        if json_field in most_recent_vitals and most_recent_vitals[json_field]:
                            try:
                                features[feature_name] = float(most_recent_vitals[json_field])
                            except:
                                features[feature_name] = np.nan
                        else:
                            features[feature_name] = np.nan
            
            # Extract medical history flags
            if 'MedicalHistory' in data:
                med_hist = data['MedicalHistory']
                history_flags = {
                    'HYPERTENSION': 'hist_hypertension',
                    'DIABETES': 'hist_diabetes',
                    'LIVER_DISEASE': 'hist_liver_disease',
                    'HEART_DISEASE': 'hist_heart_disease',
                    'KIDNEY_DISEASE': 'hist_kidney_disease',
                    'CANCER': 'hist_cancer'
                }
                
                for json_field, feature_name in history_flags.items():
                    if json_field in med_hist:
                        val = med_hist[json_field]
                        if val == 'Y' or val == True:
                            features[feature_name] = 1.0
                        elif val == 'N' or val == False:
                            features[feature_name] = 0.0
                        else:
                            features[feature_name] = 0.5  # Unknown
                    else:
                        features[feature_name] = 0.5
            
            # Transplantation outcome (ground truth)
            if 'TransplantationList' in data and data['TransplantationList']:
                transplant_list = data['TransplantationList']
                liver_transplanted = False
                for tx in transplant_list:
                    if tx.get('ORGAN') == 'Liver' and tx.get('STATUS') in ['TX', 'Transplanted']:
                        liver_transplanted = True
                        break
                features['transplanted'] = 1 if liver_transplanted else 0
            else:
                features['transplanted'] = 0  # Assume not transplanted if no data
            
            feature_dicts.append(features)
        
        # Create DataFrame
        self.df = pd.DataFrame(feature_dicts)
        print(f"Extracted {len(self.df.columns)} features from {len(self.df)} donors")
        
        return self.df
    
    def preprocess_features(self):
        """Preprocess features for clustering analysis."""
        print("Preprocessing features...")
        
        # Separate donor IDs and labels
        donor_ids = self.df['donor_id'].values
        labels = self.df['transplanted'].values
        
        # Remove non-numeric features for clustering
        numeric_df = self.df.drop(['donor_id', 'transplanted', 'blood_group', 
                                     'rhesus', 'gender', 'cause_of_admission'], 
                                    axis=1, errors='ignore')
        
        # Fill missing values with median
        numeric_df = numeric_df.fillna(numeric_df.median())
        
        # Standardize features
        scaler = StandardScaler()
        self.feature_matrix = scaler.fit_transform(numeric_df)
        self.feature_names = numeric_df.columns.tolist()
        
        print(f"Preprocessed feature matrix shape: {self.feature_matrix.shape}")
        
        return self.feature_matrix, donor_ids, labels
    
    def perform_clustering(self, n_clusters=3):
        """Perform K-means clustering."""
        print(f"Performing K-means clustering (k={n_clusters})...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.feature_matrix)
        
        silhouette_avg = silhouette_score(self.feature_matrix, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return cluster_labels, silhouette_avg
    
    def visualize_clusters(self, cluster_labels, donor_ids, tx_labels, output_dir='./figures'):
        """Create comprehensive cluster visualizations."""
        print("Creating visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. PCA visualization
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(self.feature_matrix)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot by cluster
        scatter1 = axes[0].scatter(pca_coords[:, 0], pca_coords[:, 1], 
                                   c=cluster_labels, cmap='viridis', 
                                   alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[0].set_title('(a) Donor Clustering (K-means, All Features)')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
        
        # Plot by transplantation outcome
        colors = ['red' if tx == 0 else 'green' for tx in tx_labels]
        axes[1].scatter(pca_coords[:, 0], pca_coords[:, 1], 
                       c=colors, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[1].set_title('(b) Transplantation Outcome')
        axes[1].grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Transplanted (TX)'),
                          Patch(facecolor='red', label='Not Transplanted (NTX)')]
        axes[1].legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/clustering_pca_all_features.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/clustering_pca_all_features.png', dpi=300, bbox_inches='tight')
        print(f"Saved PCA visualization to {output_dir}/clustering_pca_all_features.pdf")
        plt.close()
        
        # 2. t-SNE visualization
        print("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.feature_matrix)-1))
        tsne_coords = tsne.fit_transform(self.feature_matrix)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot by cluster
        scatter1 = axes[0].scatter(tsne_coords[:, 0], tsne_coords[:, 1], 
                                   c=cluster_labels, cmap='viridis', 
                                   alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel('t-SNE Dimension 1')
        axes[0].set_ylabel('t-SNE Dimension 2')
        axes[0].set_title('(a) Donor Clustering (t-SNE, All Features)')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
        
        # Plot by transplantation outcome
        axes[1].scatter(tsne_coords[:, 0], tsne_coords[:, 1], 
                       c=colors, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
        axes[1].set_xlabel('t-SNE Dimension 1')
        axes[1].set_ylabel('t-SNE Dimension 2')
        axes[1].set_title('(b) Transplantation Outcome')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/clustering_tsne_all_features.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/clustering_tsne_all_features.png', dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE visualization to {output_dir}/clustering_tsne_all_features.pdf")
        plt.close()
        
        # 3. Feature importance (PCA loadings)
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=self.feature_names
        )
        loadings['importance'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)
        top_features = loadings.nlargest(15, 'importance')
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Feature Importance (PCA Loading Magnitude)')
        plt.title('Top 15 Most Important Features (All Features)')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance_all.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/feature_importance_all.png', dpi=300, bbox_inches='tight')
        print(f"Saved feature importance to {output_dir}/feature_importance_all.pdf")
        plt.close()
        
        return pca_coords, tsne_coords

def main():
    """Main analysis workflow."""
    print("="*80)
    print("Liver Transplantability Analysis - Version 1: All Features")
    print("="*80)
    
    # Initialize analyzer
    json_dir = '/home/claude/SOAS_export/json'
    analyzer = DonorDataAnalyzer(json_dir)
    
    # Load and process data
    analyzer.load_all_json()
    df = analyzer.extract_all_features()
    
    # Save extracted features
    df.to_csv('/home/claude/extracted_features_all.csv', index=False)
    print(f"Saved extracted features to extracted_features_all.csv")
    
    # Preprocess
    feature_matrix, donor_ids, tx_labels = analyzer.preprocess_features()
    
    # Clustering
    cluster_labels, silhouette = analyzer.perform_clustering(n_clusters=3)
    
    # Visualize
    pca_coords, tsne_coords = analyzer.visualize_clusters(cluster_labels, donor_ids, tx_labels, 
                                                           output_dir='/home/claude/figures_all_features')
    
    # Summary statistics
    print("\n" + "="*80)
    print("Analysis Summary (All Features):")
    print("="*80)
    print(f"Total donors analyzed: {len(donor_ids)}")
    print(f"Number of features: {feature_matrix.shape[1]}")
    print(f"Transplanted donors: {sum(tx_labels)} ({100*sum(tx_labels)/len(tx_labels):.1f}%)")
    print(f"Non-transplanted donors: {len(tx_labels) - sum(tx_labels)} ({100*(len(tx_labels)-sum(tx_labels))/len(tx_labels):.1f}%)")
    print(f"Clustering silhouette score: {silhouette:.3f}")
    print("="*80)
    
    print("\nAnalysis complete! Figures saved to ./figures_all_features/")

if __name__ == "__main__":
    main()
