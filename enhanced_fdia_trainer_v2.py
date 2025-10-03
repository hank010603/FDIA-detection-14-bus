# enhanced_fdia_trainer_v2.py
"""
Enhanced FDIA Trainer v2
- Usage example:
  python enhanced_fdia_trainer_v2.py --csv mitm_output_stream.csv --out models/enhanced_fdia_model_v2.joblib --n-iter 12 --use-smote

Notes:
- Expects CSV with columns: V_0..V_41, I_0..I_41, label
- Saves a joblib bundle dict with keys: model, preprocessor, params, features, classes
"""
import argparse
import time
import os
import numpy as np
import pandas as pd
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV

# Try to import imblearn.SMOTE (optional)
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except Exception:
    _SMOTE_AVAILABLE = False

RNG = 42
NUM_BUSES = 42

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default='fdia_dataset.csv', help='path to CSV with V/I and label')
    p.add_argument('--out', type=str, default='enhanced_fdia_model_v2.joblib', help='output joblib path')
    p.add_argument('--chunksize', type=int, default=100000, help='chunksize when reading CSV')
    p.add_argument('--n-iter', type=int, default=10, help='n_iter for RandomizedSearchCV')
    p.add_argument('--use-smote', action='store_true', help='apply SMOTE oversampling (requires imblearn)')
    p.add_argument('--calibrate', action='store_true', help='wrap final estimator with CalibratedClassifierCV for better probabilities')
    p.add_argument('--max-samples', type=int, default=None, help='(optional) limit number of samples to read (for quick tests)')
    return p.parse_args()

class EnhancedFDIATrainerV2:
    def __init__(self, csv_path, out_path, chunksize=100000, use_smote=False, calibrate=False, max_samples=None, n_iter=10):
        self.csv_path = csv_path
        self.out_path = out_path
        self.chunksize = chunksize
        self.use_smote = use_smote and _SMOTE_AVAILABLE
        self.calibrate = calibrate
        self.max_samples = max_samples
        self.n_iter = n_iter

        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.best_params_ = None
        self.classes_ = None

    def load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        print(f"[+] Loading CSV in chunks from: {self.csv_path}")
        reader = pd.read_csv(self.csv_path, chunksize=self.chunksize)
        X_parts = []
        y_parts = []
        total = 0
        for chunk in reader:
            # Ensure header compatibility: try to find V_ and I_ columns
            v_cols = [c for c in chunk.columns if str(c).startswith('V_')]
            i_cols = [c for c in chunk.columns if str(c).startswith('I_')]
            if len(v_cols) < NUM_BUSES or len(i_cols) < NUM_BUSES:
                # try fallback: assume first 42 V then next 42 I if header absent
                # but prefer explicit names
                cols = list(chunk.columns)
                if len(cols) >= NUM_BUSES*2 + 1:
                    v_cols = cols[:NUM_BUSES]
                    i_cols = cols[NUM_BUSES:NUM_BUSES*2]
                else:
                    raise ValueError("CSV doesn't contain expected V_/I_ columns and not enough columns to infer.")
            # keep ordering V_0..V_41 then I_0..I_41
            v_cols_sorted = sorted(v_cols, key=lambda s: int(s.split('_')[-1]) if '_' in s and s.split('_')[-1].isdigit() else s)
            i_cols_sorted = sorted(i_cols, key=lambda s: int(s.split('_')[-1]) if '_' in s and s.split('_')[-1].isdigit() else s)
            features = v_cols_sorted + i_cols_sorted
            # label column detection
            label_col = None
            for candidate in ['label', 'ground_truth', 'y', 'target']:
                if candidate in chunk.columns:
                    label_col = candidate
                    break
            if label_col is None:
                raise ValueError("No label column found. Expect 'label' or 'ground_truth' in CSV.")

            X_chunk = chunk[features].astype(float)
            y_chunk = chunk[label_col].astype(int)

            X_parts.append(X_chunk)
            y_parts.append(y_chunk)
            total += len(X_chunk)
            print(f"  - loaded chunk with {len(X_chunk)} rows (total {total})")

            if self.max_samples and total >= self.max_samples:
                print(f"[+] Reached max_samples limit: {self.max_samples}. Stopping read.")
                break

        X = pd.concat(X_parts, ignore_index=True)
        y = pd.concat(y_parts, ignore_index=True)
        self.feature_names = X.columns.tolist()
        print(f"[+] Finished loading. Total samples: {len(X)}, features: {len(self.feature_names)}")
        return X, y

    def create_preprocessor(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('transformer', PowerTransformer(method='yeo-johnson'))
        ])

    def train(self, X, y):
        print("[+] Starting training pipeline...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RNG, stratify=y
        )
        print(f"  train / test sizes: {len(X_train)} / {len(X_test)}")

        self.preprocessor = self.create_preprocessor()
        X_train_t = self.preprocessor.fit_transform(X_train)
        X_test_t = self.preprocessor.transform(X_test)

        # Optional SMOTE (oversample minority class)
        if self.use_smote:
            if not _SMOTE_AVAILABLE:
                print("[!] SMOTE requested but imblearn not installed. Skipping SMOTE.")
            else:
                print("[+] Applying SMOTE to training set...")
                sm = SMOTE(random_state=RNG, n_jobs=-1)
                X_train_t, y_train = sm.fit_resample(X_train_t, y_train)
                print(f"  After SMOTE train size: {len(X_train_t)} (class counts: {np.bincount(y_train)})")

        # sample weights (balanced) as fallback / additional measure
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

        # Hyperparameter search space
        param_dist = {
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'max_iter': [150, 200, 300, 400],
            'max_leaf_nodes': [31, 63, 127, 255],
            'min_samples_leaf': [10, 20, 50, 100],
            'l2_regularization': [0, 0.01, 0.1, 0.5]
        }

        base_clf = HistGradientBoostingClassifier(
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RNG,
            verbose=0
        )

        search = RandomizedSearchCV(
            base_clf,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            scoring='f1_macro',
            cv=3,
            n_jobs=-1,
            random_state=RNG,
            verbose=2
        )

        print("[+] Running RandomizedSearchCV (this may take a while)...")
        search_start = time.time()
        # pass sample_weight to fit
        search.fit(X_train_t, y_train, sample_weight=sample_weight)
        search_end = time.time()
        print(f"[+] Hyperparameter search finished in {search_end - search_start:.1f}s")

        best = search.best_estimator_
        self.best_params_ = search.best_params_
        print("[+] Best params:", self.best_params_)

        # Optional calibration for better probabilities
        if self.calibrate:
            print("[+] Calibrating classifier with CalibratedClassifierCV (sigmoid)...")
            calibrated = CalibratedClassifierCV(best, cv='prefit', method='sigmoid')
            calibrated.fit(X_train_t, y_train, sample_weight=sample_weight)
            self.model = calibrated
        else:
            self.model = best

        # final evaluation
        print("[+] Evaluating on test set...")
        y_pred = self.model.predict(X_test_t)
        try:
            y_prob = self.model.predict_proba(X_test_t)[:, 1]
        except Exception:
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print("\n=== FDIA Detection Metrics (Test set) ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print("\nDetailed report:")
        print(classification_report(y_test, y_pred, zero_division=0, target_names=['Normal', 'Attack']))

        # store classes
        self.classes_ = np.unique(y)

        # attempt feature importance plotting (permutation)
        self.try_plot_feature_importance(X_test_t, y_test)

    def try_plot_feature_importance(self, X_test_t, y_test):
        try:
            from sklearn.inspection import permutation_importance
            n_samples = min(1000, X_test_t.shape[0])
            res = permutation_importance(self.model, X_test_t[:n_samples], y_test[:n_samples],
                                         n_repeats=5, random_state=RNG, n_jobs=-1)
            sorted_idx = np.argsort(res.importances_mean)
            top_n = min(20, len(self.feature_names))
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 7))
            plt.title("Top {} Permutation Importances".format(top_n))
            plt.barh(range(top_n), res.importances_mean[sorted_idx][-top_n:], xerr=res.importances_std[sorted_idx][-top_n:])
            plt.yticks(range(top_n), [self.feature_names[i] for i in sorted_idx[-top_n:]])
            plt.xlabel("Permutation importance")
            plt.tight_layout()
            plt.savefig('feature_importance_v2.png')
            print("[+] Saved feature importance to feature_importance_v2.png")
            plt.close()
        except Exception as e:
            print("[!] Could not compute or plot feature importance:", e)

    def save_model(self):
        bundle = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'params': self.best_params_,
            'features': self.feature_names,
            'classes': self.classes_.tolist() if self.classes_ is not None else None
        }
        dump(bundle, self.out_path)
        print(f"[+] Model bundle saved to: {self.out_path}")

def main():
    args = parse_args()
    trainer = EnhancedFDIATrainerV2(
        csv_path=args.csv,
        out_path=args.out,
        chunksize=args.chunksize,
        use_smote=args.use_smote,
        calibrate=args.calibrate,
        max_samples=args.max_samples,
        n_iter=args.n_iter
    )

    start = time.time()
    X, y = trainer.load_data()
    trainer.train(X, y)
    trainer.save_model()
    elapsed = time.time() - start
    print(f"[+] Total elapsed time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
