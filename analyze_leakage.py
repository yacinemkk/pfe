import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
import matplotlib.pyplot as plt

# Add project root to sys path
sys.path.insert(0, str(Path(__file__).parent.absolute()))
from src.data.json_preprocessor import JsonIoTDataProcessor
from config.config import JSON_DATA_DIR

def analyze_leakage():
    print("Loading data...")
    processor = JsonIoTDataProcessor()
    
    # Load a smaller subset of data for quick analysis
    df = processor.load_json_files(JSON_DATA_DIR, max_records=400000)
    
    print("\nFiltering classes...")
    df = processor.filter_classes(df)
    
    print("\nPreparing features...")
    X_continuous, X_binary, y_str, _ = processor.prepare_features(df)
    X = np.concatenate([X_continuous, X_binary], axis=1)
    
    y_encoded = processor.label_encoder.fit_transform(y_str)
    feature_names = processor.feature_names
    
    print(f"\nData shape: X={X.shape}, y={y_encoded.shape}")
    print(f"Number of distinct classes: {len(np.unique(y_encoded))}")
    
    print("\nCalculating ANOVA F-values...")
    f_values, p_values = f_classif(X, y_encoded)
    
    print("\nCalculating Mutual Information...")
    # Sample down if too large for MI computation
    sample_size = min(50000, X.shape[0])
    indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample, y_sample = X[indices], y_encoded[indices]
    
    mi_scores = mutual_info_classif(X_sample, y_sample)
    
    print("\nCalculating Random Forest Feature Importances...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
    rf.fit(X_sample, y_sample)
    rf_importances = rf.feature_importances_
    
    # Compile results into a dataframe
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'ANOVA_F_Value': f_values,
        'Mutual_Information': mi_scores,
        'RF_Importance': rf_importances
    })
    
    # Sort by RF Importance
    results_rf = results_df.sort_values(by='RF_Importance', ascending=False)
    print("\nTop 15 Features by Random Forest Importance:")
    print(results_rf[['Feature', 'RF_Importance']].head(15).to_string(index=False))
    
    # Sort by MI
    results_mi = results_df.sort_values(by='Mutual_Information', ascending=False)
    print("\nTop 15 Features by Mutual Information:")
    print(results_mi[['Feature', 'Mutual_Information']].head(15).to_string(index=False))
    
    # Sort by ANOVA F-value
    results_f = results_df.sort_values(by='ANOVA_F_Value', ascending=False)
    print("\nTop 15 Features by ANOVA F-value:")
    print(results_f[['Feature', 'ANOVA_F_Value']].head(15).to_string(index=False))

if __name__ == "__main__":
    analyze_leakage()
