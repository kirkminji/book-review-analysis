import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
from scipy import stats

def train_and_save():
    # 1. Data Load
    df = pd.read_csv('books_ml_dataset_v4.csv')
    df['ymw'] = df['ymw'].astype(str)
    df = df.sort_values(['product_code', 'ymw']).reset_index(drop=True)
    
    # 2. Feature Definitions (matching v4 notebook)
    feature_cols = [c for c in df.columns if c not in ['product_code', 'ymw', 'y_sales_score']]
    
    # 3. Lag Feature Creation
    df_lag = df.copy()
    for lag in [1, 2, 3, 4]:
        df_lag[f'y_lag{lag}'] = df_lag.groupby('product_code')['y_sales_score'].shift(lag)
    
    # Drop NAs for lag1 (following Part 2 logic)
    df_lag = df_lag.dropna(subset=['y_lag1']).reset_index(drop=True)
    
    # Updated feature list with lag1
    final_feature_cols = feature_cols + ['y_lag1']
    
    # 4. Time-based Split
    df_lag_sorted = df_lag.sort_values('ymw').reset_index(drop=True)
    split_idx = int(len(df_lag_sorted) * 0.8)
    
    train_data = df_lag_sorted.iloc[:split_idx]
    y_train_score = train_data['y_sales_score']
    y_train_class = (y_train_score > 0).astype(int)
    X_train = train_data[final_feature_cols]
    
    # 5. Scaling
    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    # 6. Model Training
    # Regression Model
    reg_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    reg_model.fit(X_train, y_train_score) # Using unscaled for LightGBM
    
    # Classification Model
    clf_model = lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1)
    clf_model.fit(X_train, y_train_class) # Using unscaled for LightGBM
    
    # 7. Save Models and Metadata
    os.makedirs('models', exist_ok=True)
    joblib.dump(reg_model, 'models/best_reg_model.pkl')
    joblib.dump(clf_model, 'models/best_clf_model.pkl')
    joblib.dump(scaler, 'models/robust_scaler.pkl')
    joblib.dump(final_feature_cols, 'models/feature_cols.pkl')
    
    print("Models and scaler saved successfully.")
    print(f"Final feature count: {len(final_feature_cols)}")

if __name__ == "__main__":
    train_and_save()
