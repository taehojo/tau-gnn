import pandas as pd
import numpy as np
from datetime import datetime

def calculate_annual_change_features(df, tau_regions):
    """Calculate annual change rates and other longitudinal features"""
    
    processed_data = []
    
    for idx, row in df.iterrows():
        # Check if subject has both V1 and V2 data
        has_v1 = all([not pd.isna(row.get(f'V1_{region}_SUVR', np.nan)) for region in tau_regions])
        has_v2 = all([not pd.isna(row.get(f'V2_{region}_SUVR', np.nan)) for region in tau_regions])
        
        if not (has_v1 and has_v2):
            continue
            
        # Calculate time interval
        v1_date = row.get('V1_Tau_PVC_ScanDate')
        v2_date = row.get('V2_Tau_PVC_ScanDate')
        
        if pd.notna(v1_date) and pd.notna(v2_date):
            if isinstance(v1_date, str):
                v1_date = pd.to_datetime(v1_date)
            if isinstance(v2_date, str):
                v2_date = pd.to_datetime(v2_date)
            
            time_interval = (v2_date - v1_date).days / 365.25
        else:
            time_interval = 1.0
            
        if time_interval <= 0:
            continue
            
        features = []
        
        # 1. Annual change rates
        for region in tau_regions:
            v1_val = row[f'V1_{region}_SUVR']
            v2_val = row[f'V2_{region}_SUVR']
            annual_change = (v2_val - v1_val) / time_interval
            features.append(annual_change)
        
        # 2. Baseline values (V1 SUVR - will be normalized later with StandardScaler)
        for region in tau_regions:
            v1_val = row[f'V1_{region}_SUVR']
            features.append(v1_val)
        
        # 3. Percent change
        for region in tau_regions:
            v1_val = row[f'V1_{region}_SUVR']
            v2_val = row[f'V2_{region}_SUVR']
            if v1_val > 0.01:
                pct_change = ((v2_val - v1_val) / v1_val) * 100
            else:
                pct_change = 0
            features.append(pct_change)
        
        # 4. Additional engineered features
        annual_changes = features[:len(tau_regions)]
        features.append(np.std(annual_changes))
        features.append(np.max(annual_changes))
        features.append(np.mean(annual_changes))
        
        # 5. Demographics (will be normalized later with StandardScaler)
        age = row.get('BL_Age', 70)
        if isinstance(age, str) and '>' in age:
            age = 90
        features.append(float(age))
        
        features.append(1 if row.get('PTGENDER') == 1 else 0)  # Male=1
        
        education = row.get('PTEDUCAT', 16)
        if pd.isna(education):
            education = 16
        features.append(float(education))
        
        # 6. APOE ε4 carrier status
        apoe_status = row.get('APOEGrp', 0)  # 0=non-carrier, 1=carrier
        if pd.isna(apoe_status):
            apoe_status = 0
        features.append(int(apoe_status))
        
        # Label
        dx = row['BL_DXGrp']
        label = 1 if dx in [1.0, 5.0] else 0
        
        processed_data.append({
            'RID': row['RID'],
            'features': features,
            'label': label,
            'time_interval': time_interval
        })
    
    return processed_data


def get_feature_names(tau_regions):
    """Get feature names for interpretation"""
    names = []
    
    # Annual changes
    names.extend([f'{region}_annual_change' for region in tau_regions])
    # Baseline values  
    names.extend([f'{region}_baseline' for region in tau_regions])
    # Percent changes
    names.extend([f'{region}_percent_change' for region in tau_regions])
    # Engineered features
    names.extend(['change_std', 'change_max', 'change_mean'])
    # Demographics
    names.extend(['age', 'gender', 'education', 'apoe_e4'])
    
    return names