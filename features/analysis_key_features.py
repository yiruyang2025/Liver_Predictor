#!/usr/bin/env python3
"""
Liver Transplantability Prediction - Feature Analysis (Key Purple-Marked Features Only)
ICML 2026 Submission
Version 2: Clustering analysis using ONLY purple-highlighted features from DIF template
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
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

class DonorDataAnalyzerKeyFeatures:
    """
    Analyzer for donor medical data focusing on clinically important features
    identified by liver transplant surgeons (purple-highlighted in DIF template).
    """
    
    # Key features highlighted in purple in the DIF template
    KEY_FEATURES_MAPPING = {
        # Serology markers (purple-highlighted)
        'HIV': 'hiv_status',
        'Hepatitis B': 'hep_b_status', 
        'Hepatitis C': 'hep_c_status',
        'Epstein-Barr': 'ebv_status',
        'Cytomegalovirus': 'cmv_status',
        
        # Critical lab values (liver function - purple-highlighted)
        'ASAT': 'asat',
        'ALAT': 'alat',
        'BILIRUBIN_TOT': 'bilirubin_total',
        'INR': 'inr',
        'FACTOR_V': 'factor_v',
        'CREATININE': 'creatinine',
        
        # Donor characteristics (purple-highlighted)
        'AGE': 'age',
        'BMI': 'bmi',
        'CAUSE_OF_DEATH': 'cause_of_death'
    }
    
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
        
    def extract_key_features(self):
        """
        Extract ONLY the purple-highlighted clinically important features
        as identified by liver transplant surgeons.
        """
        print("Extracting key purple-highlighted features only...")
        
        feature_dicts = []
        
        for donor in self.donors_data:
            features = {}
            data = donor['data']
            filename = donor['filename']
            
            features['donor_id'] = filename.replace('.json', '')
            
            # Extract Main features
            if 'Main' in data:
                main = data['Main']
                
                # Age (purple-highlighted)
                if 'AGE' in main and main['AGE']:
                    age_str = str(main['AGE'])
                    try:
                        features['age'] = float(age_str.replace('Y', '').strip())
                    except:
                        features['age'] = np.nan
                else:
                    features['age'] = np.nan
                
                # BMI (purple-highlighted)
                features['bmi'] = float(main.get('BMI', np.nan)) if main.get('BMI') else np.nan
            
            # Serology (purple-highlighted markers)
            if 'Serology' in data:
                serology = data['Serology']
                
                # HIV status (purple-highlighted)
                hiv_markers = ['HIV_1_2_AB_HIV_1_P24_AG', 'HIV_1_AB', 'HIV_2_AB', 
                               'HIV_1_P24_AG', 'HIV_RNA_1', 'HIV_RNA_2']
                hiv_positive = False
                for marker in hiv_markers:
                    if marker in serology and serology[marker]:
                        val = str(serology[marker]).upper()
                        if 'POS' in val or 'DETECT' in val or '+' in val:
                            hiv_positive = True
                            break
                features['hiv_status'] = 1.0 if hiv_positive else 0.0
                
                # Hepatitis B (purple-highlighted)
                hep_b_markers = ['HBS_AG', 'HBC_AB', 'HBV_DNA']
                hep_b_positive = False
                for marker in hep_b_markers:
                    if marker in serology and serology[marker]:
                        val = str(serology[marker]).upper()
                        if 'POS' in val or 'DETECT' in val or '+' in val:
                            hep_b_positive = True
                            break
                features['hep_b_status'] = 1.0 if hep_b_positive else 0.0
                
                # Hepatitis C (purple-highlighted)
                hep_c_markers = ['HCV_AB', 'HCV_RNA']
                hep_c_positive = False
                for marker in hep_c_markers:
                    if marker in serology and serology[marker]:
                        val = str(serology[marker]).upper()
                        if 'POS' in val or 'DETECT' in val or '+' in val:
                            hep_c_positive = True
                            break
                features['hep_c_status'] = 1.0 if hep_c_positive else 0.0
                
                # EBV (purple-highlighted)
                ebv_positive = False
                if 'EBV_IGG' in serology and serology['EBV_IGG']:
                    val = str(serology['EBV_IGG']).upper()
                    if 'POS' in val or '+' in val:
                        ebv_positive = True
                features['ebv_status'] = 1.0 if ebv_positive else 0.0
                
                # CMV (purple-highlighted)
                cmv_positive = False
                if 'CMV_IGG' in serology and serology['CMV_IGG']:
                    val = str(serology['CMV_IGG']).upper()
                    if 'POS' in val or '+' in val:
                        cmv_positive = True
                features['cmv_status'] = 1.0 if cmv_positive else 0.0
            else:
                # Default to negative if no serology data
                features['hiv_status'] = 0.0
                features['hep_b_status'] = 0.0
                features['hep_c_status'] = 0.0
                features['ebv_status'] = 0.0
                features['cmv_status'] = 0.0
            
            # Laboratory values (purple-highlighted liver function markers)
            if 'LabBloodList' in data and data['LabBloodList']:
                lab_results = data['LabBloodList']
                if isinstance(lab_results, list) and len(lab_results) > 0:
                    most_recent = lab_results[-1]
                    
                    # ASAT (AST) - purple-highlighted
                    if 'ASAT' in most_recent and most_recent['ASAT']:
                        try:
                            features['asat'] = float(most_recent['ASAT'])
                        except:
                            features['asat'] = np.nan
                    else:
                        features['asat'] = np.nan
                    
                    # ALAT (ALT) - purple-highlighted
                    if 'ALAT' in most_recent and most_recent['ALAT']:
                        try:
                            features['alat'] = float(most_recent['ALAT'])
                        except:
                            features['alat'] = np.nan
                    else:
                        features['alat'] = np.nan
                    
                    # Bilirubin Total - purple-highlighted
                    if 'BILIRUBIN_TOT' in most_recent and most_recent['BILIRUBIN_TOT']:
                        try:
                            features['bilirubin_total'] = float(most_recent['BILIRUBIN_TOT'])
                        except:
                            features['bilirubin_total'] = np.nan
                    else:
                        features['bilirubin_total'] = np.nan
                    
                    # INR - purple-highlighted
                    if 'INR' in most_recent and most_recent['INR']:
                        try:
                            features['inr'] = float(most_recent['INR'])
                        except:
                            features['inr'] = np.nan
                    else:
                        features['inr'] = np.nan
                    
                    # Factor V - purple-highlighted
                    if 'FACTOR_V' in most_recent and most_recent['FACTOR_V']:
                        try:
                            features['factor_v'] = float(most_recent['FACTOR_V'])
                        except:
                            features['factor_v'] = np.nan
                    else:
                        features['factor_v'] = np.nan
                    
                    # Creatinine - purple-highlighted
                    if 'CREATININE' in most_recent and most_recent['CREATININE']:
                        try:
                            features['creatinine'] = float(most_recent['CREATININE'])
                        except:
                            features['creatinine'] = np.nan
                    else:
                        features['creatinine'] = np.nan
            else:
                features['asat'] = np.nan
                features['alat'] = np.nan
                features['bilirubin_total'] = np.nan
                features['inr'] = np.nan
                features['factor_v'] = np.nan
                features['creatinine'] = np.nan
            
            # Cause of death (purple-highlighted) - categorical feature
            if 'DeathCertification' in data:
                death_cert = data['DeathCertification']
                features['cause_of_death'] = death_cert.get('CAUSE_OF_DEATH', 'Unknown')
            else:
                features['cause_of_death'] = 'Unknown'
            
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
                features['transplanted'] = 0
            
            feature_dicts.append(features)
        
        # Create DataFrame
        self.df = pd.DataFrame(feature_dicts)
        print(f"Extracted {len(self.df.columns)-2} key purple-highlighted features from {len(self.df)} donors")
        print(f"Key features: {[col for col in self.df.columns if col not in ['donor_id', 'transplanted', 'cause_of_death']]}")
        
        return self.df
    
    def preprocess_features(self):
        """Preprocess key features for clustering analysis."""
        print("Preprocessing key features...")
        
        # Separate donor IDs and labels
        donor_ids = self.df['donor_id'].values
        labels = self.df['transplanted'].values
        
        # Remove non-numeric features for clustering
        numeric_df = self.df.drop(['donor_id', 'transplanted', 'cause_of_death'], 
                                    axis=1, errors='ignore')
        
        # Fill missing values with median (column-wise)
        for col in numeric_df.columns:
            median_val = numeric_df[col].median()
            if np.isnan(median_val):
                # If median is NaN (all values missing), fill with 0
                numeric_df[col] = numeric_df[col].fillna(0)
            else:
                numeric_df[col] = numeric_df[col].fillna(median_val)
        
        # Standardize features
        scaler = StandardScaler()
        self.feature_matrix = scaler.fit_transform(numeric_df)
        self.feature_names = numeric_df.columns.tolist()
        
        print(f"Preprocessed feature matrix shape: {self.feature_matrix.shape}")
        print(f"Features used: {self.feature_names}")
        
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
        """Create comprehensive cluster visualizations using key features."""
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
        axes[0].set_title('(a) Donor Clustering (K-means, Key Purple Features)')
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
        plt.savefig(f'{output_dir}/clustering_pca_key_features.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/clustering_pca_key_features.png', dpi=300, bbox_inches='tight')
        print(f"Saved PCA visualization to {output_dir}/clustering_pca_key_features.pdf")
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
        axes[0].set_title('(a) Donor Clustering (t-SNE, Key Purple Features)')
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
        plt.savefig(f'{output_dir}/clustering_tsne_key_features.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/clustering_tsne_key_features.png', dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE visualization to {output_dir}/clustering_tsne_key_features.pdf")
        plt.close()
        
        # 3. Feature importance (PCA loadings) for key features
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2'],
            index=self.feature_names
        )
        loadings['importance'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)
        loadings_sorted = loadings.sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(loadings_sorted)), loadings_sorted['importance'].values, color='steelblue')
        plt.yticks(range(len(loadings_sorted)), loadings_sorted.index)
        plt.xlabel('Feature Importance (PCA Loading Magnitude)')
        plt.title('Feature Importance: Purple-Highlighted Clinical Markers')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance_key.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/feature_importance_key.png', dpi=300, bbox_inches='tight')
        print(f"Saved feature importance to {output_dir}/feature_importance_key.pdf")
        plt.close()
        
        return pca_coords, tsne_coords

def main():
    """Main analysis workflow for key purple-highlighted features."""
    print("="*80)
    print("Liver Transplantability Analysis - Version 2: Key Purple Features Only")
    print("="*80)
    
    # Initialize analyzer
    json_dir = '/home/claude/SOAS_export/json'
    analyzer = DonorDataAnalyzerKeyFeatures(json_dir)
    
    # Load and process data
    analyzer.load_all_json()
    df = analyzer.extract_key_features()
    
    # Save extracted features
    df.to_csv('/home/claude/extracted_features_key.csv', index=False)
    print(f"Saved extracted key features to extracted_features_key.csv")
    
    # Preprocess
    feature_matrix, donor_ids, tx_labels = analyzer.preprocess_features()
    
    # Clustering
    cluster_labels, silhouette = analyzer.perform_clustering(n_clusters=3)
    
    # Visualize
    pca_coords, tsne_coords = analyzer.visualize_clusters(cluster_labels, donor_ids, tx_labels, 
                                                           output_dir='/home/claude/figures_key_features')
    
    # Summary statistics
    print("\n" + "="*80)
    print("Analysis Summary (Key Purple Features):")
    print("="*80)
    print(f"Total donors analyzed: {len(donor_ids)}")
    print(f"Number of key features: {feature_matrix.shape[1]}")
    print(f"Transplanted donors: {sum(tx_labels)} ({100*sum(tx_labels)/len(tx_labels):.1f}%)")
    print(f"Non-transplanted donors: {len(tx_labels) - sum(tx_labels)} ({100*(len(tx_labels)-sum(tx_labels))/len(tx_labels):.1f}%)")
    print(f"Clustering silhouette score: {silhouette:.3f}")
    print("="*80)
    
    print("\nAnalysis complete! Figures saved to ./figures_key_features/")

if __name__ == "__main__":
    main()
