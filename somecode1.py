import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')

class PnLPredictor:
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.feature_importance = {}
        self.pca = None
        
    def load_data(self, data=None, train_path=None, test_path=None, test_size=0.2, random_state=42):
        """Load training and test data from variable or files"""
        if data is not None:
            # Split the provided data into train/test
            print(f"Original data shape: {data.shape}")
            
            # Stratified split to maintain target distribution
            # For regression, we can bin the target for stratification
            target_col = 'pnl'  # Assuming 'pnl' is the target column
            if target_col in data.columns:
                # Create bins for stratification
                data['target_bins'] = pd.qcut(data[target_col], q=5, labels=False, duplicates='drop')
                
                from sklearn.model_selection import train_test_split
                self.train_df, self.test_df = train_test_split(
                    data, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=data['target_bins']
                )
                
                # Remove the temporary binning column
                self.train_df = self.train_df.drop('target_bins', axis=1)
                self.test_df = self.test_df.drop('target_bins', axis=1)
            else:
                # Simple random split if target column not found
                self.train_df, self.test_df = train_test_split(
                    data, 
                    test_size=test_size, 
                    random_state=random_state
                )
                
            print(f"Training data shape: {self.train_df.shape}")
            print(f"Test data shape: {self.test_df.shape}")
            
        elif train_path:
            self.train_df = pd.read_csv(train_path)
            if test_path:
                self.test_df = pd.read_csv(test_path)
            print(f"Training data shape: {self.train_df.shape}")
            if hasattr(self, 'test_df'):
                print(f"Test data shape: {self.test_df.shape}")
    
    def explore_data(self):
        """Comprehensive data exploration"""
        print("=== DATA EXPLORATION ===")
        
        # Basic info
        print("\nTraining Data Info:")
        print(self.train_df.info())
        
        # Target variable analysis
        target_col = 'pnl'  # Assuming this is the target column name
        if target_col in self.train_df.columns:
            print(f"\nTarget Variable ({target_col}) Statistics:")
            print(self.train_df[target_col].describe())
            
            # Plot target distribution
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.hist(self.train_df[target_col], bins=50, alpha=0.7)
            plt.title('PnL Distribution')
            plt.subplot(1, 2, 2)
            plt.boxplot(self.train_df[target_col])
            plt.title('PnL Box Plot')
            plt.tight_layout()
            plt.show()
        
        # Missing values
        missing_data = self.train_df.isnull().sum()
        if missing_data.sum() > 0:
            print("\nMissing Values:")
            print(missing_data[missing_data > 0])
        
        # Correlation analysis for numeric columns
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.train_df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.show()
    
    def feature_engineering(self):
        """Advanced feature engineering for financial data"""
        print("=== FEATURE ENGINEERING ===")
        
        # Combine train and test for consistent feature engineering
        if hasattr(self, 'test_df'):
            combined_df = pd.concat([self.train_df, self.test_df], ignore_index=True, sort=False)
            is_train = [True] * len(self.train_df) + [False] * len(self.test_df)
        else:
            combined_df = self.train_df.copy()
            is_train = [True] * len(self.train_df)
        
        # Identify market factor columns (numeric columns excluding target and ID)
        exclude_cols = ['pnl', 'scenario_id', 'segment_description_level_1', 
                       'segment_description_level_2', 'segment_description_level_3', 
                       'segment_description_level_4', 'sub_id']
        
        market_cols = [col for col in combined_df.columns 
                      if combined_df[col].dtype in ['float64', 'int64'] 
                      and col not in exclude_cols]
        
        print(f"Identified {len(market_cols)} market factor columns")
        
        # 1. Basic market factor features
        if market_cols:
            # Portfolio exposure magnitude
            combined_df['total_exposure'] = combined_df[market_cols].abs().sum(axis=1)
            combined_df['net_exposure'] = combined_df[market_cols].sum(axis=1)
            combined_df['long_exposure'] = combined_df[market_cols].clip(lower=0).sum(axis=1)
            combined_df['short_exposure'] = combined_df[market_cols].clip(upper=0).abs().sum(axis=1)
            
            # Portfolio concentration
            combined_df['exposure_concentration'] = (combined_df[market_cols].abs() / 
                                                   (combined_df[market_cols].abs().sum(axis=1) + 1e-8)).max(axis=1)
            
            # Number of active positions
            combined_df['active_positions'] = (combined_df[market_cols].abs() > 1e-6).sum(axis=1)
            
        # 2. Sector/Asset class groupings
        # Group similar market factors
        equity_cols = [col for col in market_cols if any(x in col.lower() for x in ['spy', 'gbp', 'ftse', 'stoxx'])]
        fx_cols = [col for col in market_cols if any(x in col.lower() for x in ['usd', 'eur', 'gbp', 'jpy'])]
        rates_cols = [col for col in market_cols if any(x in col.lower() for x in ['std', 'none', 'mm'])]
        
        if equity_cols:
            combined_df['equity_exposure'] = combined_df[equity_cols].sum(axis=1)
            combined_df['equity_exposure_abs'] = combined_df[equity_cols].abs().sum(axis=1)
            
        if fx_cols:
            combined_df['fx_exposure'] = combined_df[fx_cols].sum(axis=1)
            combined_df['fx_exposure_abs'] = combined_df[fx_cols].abs().sum(axis=1)
            
        if rates_cols:
            combined_df['rates_exposure'] = combined_df[rates_cols].sum(axis=1)
            combined_df['rates_exposure_abs'] = combined_df[rates_cols].abs().sum(axis=1)
        
        # 3. Statistical features
        if market_cols:
            combined_df['exposure_mean'] = combined_df[market_cols].mean(axis=1)
            combined_df['exposure_std'] = combined_df[market_cols].std(axis=1)
            combined_df['exposure_skew'] = combined_df[market_cols].skew(axis=1)
            combined_df['exposure_kurt'] = combined_df[market_cols].kurtosis(axis=1)
            
        # 4. Segment description features
        segment_cols = ['segment_description_level_1', 'segment_description_level_2', 
                       'segment_description_level_3', 'segment_description_level_4']
        
        for col in segment_cols:
            if col in combined_df.columns:
                # Label encode categorical variables
                le = LabelEncoder()
                combined_df[f'{col}_encoded'] = le.fit_transform(combined_df[col].fillna('missing'))
                
                # Count of each category
                category_counts = combined_df[col].value_counts()
                combined_df[f'{col}_count'] = combined_df[col].map(category_counts).fillna(0)
        
        # 5. PCA features for dimensionality reduction
        from sklearn.decomposition import IncrementalPCA

        # 5. PCA features for dimensionality reduction
        if len(market_cols) > 10:
            print("Creating PCA features using IncrementalPCA...")
            market_data = combined_df[market_cols].fillna(0)

            # Use only top N most variable columns to reduce memory usage
            n_top_vars = 50
            variances = market_data.var().sort_values(ascending=False)
            top_var_cols = variances.head(n_top_vars).index.tolist()
    
            # Standardize
            scaler = StandardScaler()
            market_scaled = scaler.fit_transform(market_data[top_var_cols])

            # Use IncrementalPCA to avoid memory blowup
            self.pca = IncrementalPCA(n_components=10, batch_size=10000)
            pca_features = self.pca.fit_transform(market_scaled)

            for i in range(pca_features.shape[1]):
                combined_df[f'pca_{i}'] = pca_features[:, i]

            print(f"PCA explained variance (first 5 components): {self.pca.explained_variance_ratio_[:5]}")
        
        # 6. Interaction features (top correlations with target if available)
        if 'pnl' in combined_df.columns and market_cols:
            train_data = combined_df[is_train]
            correlations = train_data[market_cols + ['pnl']].corr()['pnl'].abs().sort_values(ascending=False)
            top_features = correlations.head(5).index.tolist()
            top_features = [f for f in top_features if f != 'pnl']
            
            if len(top_features) >= 2:
                print(f"Creating interactions with top features: {top_features[:3]}")
                for i in range(min(3, len(top_features))):
                    for j in range(i+1, min(3, len(top_features))):
                        feat1, feat2 = top_features[i], top_features[j]
                        combined_df[f'{feat1}_x_{feat2}'] = combined_df[feat1] * combined_df[feat2]
        
        # Split back into train and test
        self.train_processed = combined_df[is_train].copy()
        if hasattr(self, 'test_df'):
            self.test_processed = combined_df[~np.array(is_train)].copy()
        
        # Get feature columns (exclude target and IDs)
        self.feature_cols = [col for col in self.train_processed.columns 
                           if col not in ['pnl', 'scenario_id'] and 
                           not col.startswith('segment_description_level')]
        
        print(f"Created {len(self.feature_cols)} features total")
        
        return self.train_processed, getattr(self, 'test_processed', None)
    
    def prepare_features(self, df, fit_scaler=True):
        """Prepare features with proper scaling"""
        features = df[self.feature_cols].copy()
        
        # Handle missing values
        features = features.fillna(features.median())
        
        # Scale features
        if fit_scaler:
            self.scaler = RobustScaler()
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)
            
        return pd.DataFrame(features_scaled, columns=self.feature_cols, index=features.index)
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val, params=None):
        """Train LightGBM model with optimal parameters"""
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'random_state': 42
            }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, params=None):
        """Train XGBoost model"""
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'lambda': 1,
                'alpha': 0.1,
                'random_state': 42
            }
        
        train_data = xgb.DMatrix(X_train, label=y_train)
        val_data = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            train_data,
            evals=[(train_data, 'train'), (val_data, 'val')],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return model
    
    def cross_validate_model(self, X, y, model_type='lgb', n_splits=5):
        """Perform cross-validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        oof_predictions = np.zeros(len(X))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n=== Fold {fold + 1} ===")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            if model_type == 'lgb':
                model = self.train_lightgbm(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
            elif model_type == 'xgb':
                model = self.train_xgboost(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                val_pred = model.predict(xgb.DMatrix(X_val_fold))
            else:
                raise ValueError("model_type must be 'lgb' or 'xgb'")
            
            oof_predictions[val_idx] = val_pred
            fold_score = np.sqrt(mean_squared_error(y_val_fold, val_pred))
            cv_scores.append(fold_score)
            print(f"Fold {fold + 1} RMSE: {fold_score:.6f}")
        
        overall_score = np.sqrt(mean_squared_error(y, oof_predictions))
        print(f"\nOverall CV RMSE: {overall_score:.6f}")
        print(f"CV RMSE std: {np.std(cv_scores):.6f}")
        
        return overall_score, oof_predictions
    
    def train_ensemble(self, train_features, train_target):
        """Train ensemble of different models"""
        print("=== TRAINING ENSEMBLE ===")
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_features, train_target, test_size=0.2, random_state=42
        )
        
        models = {}
        
        # 1. LightGBM
        print("\nTraining LightGBM...")
        lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val)
        models['lgb'] = lgb_model
        
        # 2. XGBoost
        print("\nTraining XGBoost...")
        xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
        models['xgb'] = xgb_model
        
        # 3. Ridge Regression
        print("\nTraining Ridge...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        models['ridge'] = ridge
        
        # 4. Random Forest
        print("\nTraining Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models['rf'] = rf
        
        # Evaluate individual models
        print("\n=== MODEL EVALUATION ===")
        val_predictions = {}
        for name, model in models.items():
            if name == 'lgb':
                pred = model.predict(X_val, num_iteration=model.best_iteration)
            elif name == 'xgb':
                pred = model.predict(xgb.DMatrix(X_val))
            else:
                pred = model.predict(X_val)
            
            val_predictions[name] = pred
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            print(f"{name.upper()} RMSE: {rmse:.6f}")
        
        # Simple ensemble (average)
        ensemble_pred = np.mean(list(val_predictions.values()), axis=0)
        ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        print(f"ENSEMBLE RMSE: {ensemble_rmse:.6f}")
        
        self.models = models
        return models
    
    def predict(self, test_features):
        """Make predictions using ensemble"""
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'lgb':
                pred = model.predict(test_features, num_iteration=model.best_iteration)
            elif name == 'xgb':
                pred = model.predict(xgb.DMatrix(test_features))
            else:
                pred = model.predict(test_features)
            predictions[name] = pred
        
        # Ensemble prediction (simple average)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        return ensemble_pred, predictions
    
    def run_full_pipeline(self, data=None, train_path=None, test_path=None, test_size=0.2):
        """Run the complete pipeline"""
        print("=== STARTING PNL PREDICTION PIPELINE ===")
        
        # Load data
        self.load_data(data=data, train_path=train_path, test_path=test_path, test_size=test_size)
        
        # Explore data
        self.explore_data()
        
        # Feature engineering
        train_processed, test_processed = self.feature_engineering()
        
        # Prepare features
        train_features = self.prepare_features(train_processed, fit_scaler=True)
        train_target = train_processed['pnl']
        
        # Train ensemble
        self.train_ensemble(train_features, train_target)
        
        # Make predictions on test set
        if hasattr(self, 'test_df'):
            test_features = self.prepare_features(test_processed, fit_scaler=False)
            ensemble_pred, individual_preds = self.predict(test_features)
            
            # Evaluate on test set (since we have true labels)
            test_target = test_processed['pnl']
            test_rmse = np.sqrt(mean_squared_error(test_target, ensemble_pred))
            print(f"\n=== TEST SET EVALUATION ===")
            print(f"Test RMSE: {test_rmse:.6f}")
            
            # Individual model performance on test set
            for name, pred in individual_preds.items():
                model_rmse = np.sqrt(mean_squared_error(test_target, pred))
                print(f"{name.upper()} Test RMSE: {model_rmse:.6f}")
            
            # Create results dataframe
            results = pd.DataFrame({
                'actual': test_target,
                'predicted': ensemble_pred,
                'residual': test_target - ensemble_pred
            })
            
            # Plot predictions vs actual
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.scatter(results['actual'], results['predicted'], alpha=0.5)
            plt.plot([results['actual'].min(), results['actual'].max()], 
                    [results['actual'].min(), results['actual'].max()], 'r--')
            plt.xlabel('Actual PnL')
            plt.ylabel('Predicted PnL')
            plt.title('Predictions vs Actual')
            
            plt.subplot(1, 2, 2)
            plt.hist(results['residual'], bins=50, alpha=0.7)
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title('Residual Distribution')
            plt.tight_layout()
            plt.show()
            
            return results
        
        return None

# Usage example for your scenario:
"""
# With your existing data variable (180k rows, 22 columns)
predictor = PnLPredictor()

# Option 1: Use 80% for training, 20% for testing
results = predictor.run_full_pipeline(data=data, test_size=0.2)

# Option 2: Custom split (e.g., 90% train, 10% test)
results = predictor.run_full_pipeline(data=data, test_size=0.1)

# Option 3: For cross-validation without holdout test set
predictor.load_data(data=data, test_size=0.0)  # Use all data for training
predictor.explore_data()
train_processed, _ = predictor.feature_engineering()
train_features = predictor.prepare_features(train_processed, fit_scaler=True)
train_target = train_processed['pnl']

# 5-fold cross-validation
cv_score, oof_preds = predictor.cross_validate_model(train_features, train_target, 'lgb', n_splits=5)
"""

print("PnL Prediction Pipeline Ready for your data variable!")
print("Usage:")
print("1. predictor = PnLPredictor()")
print("2. results = predictor.run_full_pipeline(data=data, test_size=0.2)")
print("3. This will use 80% for training and 20% for testing with stratified split")
