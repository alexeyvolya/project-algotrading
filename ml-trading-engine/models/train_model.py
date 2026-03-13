import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import yaml
from datetime import datetime

class ModelTrainer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.prediction_horizon = self.config['ml']['prediction_horizon']
        self.train_window = self.config['ml']['train_window']
        self.random_state = self.config['ml']['random_state']
        
    def create_target(self, df):
        """Create target variable: 1 if price increases in next N candles, 0 otherwise"""
        future_price = df['close'].shift(-self.prediction_horizon)
        current_price = df['close']
        df['target'] = (future_price > current_price).astype(int)
        return df.dropna()
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Drop non-feature columns
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target']]
        X = df[feature_cols]
        y = df['target']
        return X, y
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        return model
    
    def walk_forward_validation(self, X, y):
        """Perform walk-forward validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            model = self.train_xgboost(X_train, y_train, X_val, y_val)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            scores.append(accuracy)
        
        return np.mean(scores), np.std(scores)
    
    def feature_importance(self, model, feature_names):
        """Get feature importance"""
        importance = model.get_booster().get_score(importance_type='gain')
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    def train_and_save(self, df, model_path):
        """Main training pipeline"""
        # Create target
        df = self.create_target(df)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Walk-forward validation
        mean_score, std_score = self.walk_forward_validation(X, y)
        print(f"Walk-forward validation accuracy: {mean_score:.4f} ± {std_score:.4f}")
        
        # Train final model on all data
        model = self.train_xgboost(X, y, X, y)
        
        # Feature importance
        importance = self.feature_importance(model, X.columns)
        print("Top 10 features:")
        for feat, imp in importance[:10]:
            print(f"  {feat}: {imp:.4f}")
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        return model, importance