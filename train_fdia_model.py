import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class EnhancedFDIATrainer:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.best_params_ = None
        self.X_test = None
        self.y_test = None

    def load_data(self, csv_path):
       
        print("Loading and optimizing 500,000 samples...")
        chunks = pd.read_csv(csv_path, chunksize=100000)
        
        X_list, y_list = [], []
        for chunk in chunks:
            
            float_cols = [f'V_{i}' for i in range(42)] + [f'I_{i}' for i in range(42)]
            chunk[float_cols] = chunk[float_cols].astype('float32')
            chunk['label'] = chunk['label'].astype('int8')
            
            X_list.append(chunk.drop('label', axis=1))
            y_list.append(chunk['label'])
        
        X = pd.concat(X_list)
        y = pd.concat(y_list)
        self.feature_names = X.columns.tolist()
        return X, y

    def create_preprocessor(self):
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('transformer', PowerTransformer(method='yeo-johnson'))
        ])

    def train_enhanced_model(self, X, y):
       
        print("Training enhanced model...")
        
    
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
       
        self.preprocessor = self.create_preprocessor()
        X_train = self.preprocessor.fit_transform(X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        
  
        param_dist = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_iter': [200, 300, 400],
            'max_leaf_nodes': [63, 127, 255],
            'min_samples_leaf': [20, 50, 100],
            'l2_regularization': [0, 0.1, 0.5]
        }
        
      
        scorer = make_scorer(f1_score, pos_label=1)
        

        search = RandomizedSearchCV(
            HistGradientBoostingClassifier(
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=42,
                verbose=2
            ),
            param_distributions=param_dist,
            n_iter=10,
            scoring=scorer,
            cv=3,
            n_jobs=-1,
            random_state=42
        )
        
        print("\nStarting hyperparameter optimization...")
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        
  
        print("\nBest Parameters:", self.best_params_)
        print("\nFinal Evaluation:")
        y_pred = self.model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred, target_names=['Normal', 'Attack']))
        
       
        self.plot_feature_importance()

    def plot_feature_importance(self):
        
        try:
           
            from sklearn.inspection import permutation_importance
            
      
            n_samples = min(1000, len(self.X_test))
            X_subset = self.X_test[:n_samples]
            y_subset = self.y_test[:n_samples]
            
            result = permutation_importance(
                self.model, 
                X_subset,
                y_subset,
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )
            
            sorted_idx = result.importances_mean.argsort()
            top_n = min(20, len(self.feature_names))
            
            plt.figure(figsize=(12, 8))
            plt.title(f"Top {top_n} Important Features (Permutation Importance)")
            plt.barh(
                range(top_n),
                result.importances_mean[sorted_idx][-top_n:],
                xerr=result.importances_std[sorted_idx][-top_n:],
                align='center'
            )
            plt.yticks(
                range(top_n),
                [self.feature_names[i] for i in sorted_idx[-top_n:]]
            )
            plt.xlabel("Permutation Importance")
            plt.tight_layout()
            plt.savefig('feature_importance_enhanced.png')
            plt.show()
        except Exception as e:
            print(f"\nWarning: Could not plot feature importance - {e}")

    def save_model(self, path='enhanced_fdia_model.joblib'):
        
        dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'params': self.best_params_,
            'features': self.feature_names
        }, path)
        print(f"\nEnhanced model successfully saved to {path}")

if __name__ == "__main__":
    trainer = EnhancedFDIATrainer()
    
    try:
    
        X, y = trainer.load_data('fdia_dataset.csv')
        
        
        trainer.train_enhanced_model(X, y)
        
      
        trainer.save_model()
        print("\nEnhanced training completed successfully!")
        
    except Exception as e:
        print(f"\nError during enhanced training: {e}")