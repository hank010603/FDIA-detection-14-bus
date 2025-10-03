# csv_to_model_predict.py
import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default='mitm_output_stream.csv', help='path to csv produced by mitm')
parser.add_argument('--model', type=str, default='enhanced_fdia_model.joblib', help='path to joblib model bundle')
parser.add_argument('--chunksize', type=int, default=5000)
args = parser.parse_args()

CSV_PATH = args.csv
MODEL_PATH = args.model
CHUNK = args.chunksize
NUM_BUSES = 42

# load model bundle
bundle = joblib.load(MODEL_PATH)
if isinstance(bundle, dict):
    model = bundle.get('model', None)
    preprocessor = bundle.get('preprocessor', None)
else:
    model = bundle
    preprocessor = None

if model is None:
    raise RuntimeError("Model not found in joblib bundle")

y_true_all = []
y_pred_all = []
probs_all = []

cols = [f'V_{i}' for i in range(NUM_BUSES)] + [f'I_{i}' for i in range(NUM_BUSES)] + ['ground_truth', 'timestamp', 'metadata_json']

print(f"[+] Reading CSV: {CSV_PATH}")
reader = pd.read_csv(CSV_PATH, chunksize=CHUNK, names=None)  # assume header present

first_chunk = True
for chunk in reader:
    # If header present, pandas will infer column names; to be safe, try to detect
    if first_chunk:
        # if 'ground_truth' not in columns, try to rename assuming default header
        if 'ground_truth' not in chunk.columns:
            # assume CSV has header we wrote, set explicit names
            chunk.columns = cols
        first_chunk = False

    # ensure we have the feature columns
    if not set([f'V_{i}' for i in range(NUM_BUSES)] + [f'I_{i}' for i in range(NUM_BUSES)]).issubset(set(chunk.columns)):
        # try to handle case where header repeated or extra columns - attempt to select by prefix
        v_cols = [c for c in chunk.columns if str(c).startswith('V_')][:NUM_BUSES]
        i_cols = [c for c in chunk.columns if str(c).startswith('I_')][:NUM_BUSES]
        if len(v_cols) < NUM_BUSES or len(i_cols) < NUM_BUSES:
            print("[!] Chunk missing expected V/I columns; skipping")
            continue
        features_df = pd.concat([chunk[v_cols], chunk[i_cols]], axis=1)
    else:
        features_df = chunk[[f'V_{i}' for i in range(NUM_BUSES)] + [f'I_{i}' for i in range(NUM_BUSES)]]

    X = features_df.values.astype(float)
    if preprocessor is not None:
        try:
            Xp = preprocessor.transform(X)
        except Exception as e:
            print("[!] Preprocessor transform error:", e)
            Xp = X
    else:
        Xp = X

    try:
        preds = model.predict(Xp)
    except Exception as e:
        print("[!] Model predict error:", e)
        break

    prob_pos = None
    if hasattr(model, 'predict_proba'):
        try:
            prob_pos = model.predict_proba(Xp)[:, 1]
        except:
            prob_pos = None

    # collect metrics if ground_truth exists
    if 'ground_truth' in chunk.columns:
        y_true = chunk['ground_truth'].astype(int).values
        y_pred = np.array(preds).astype(int)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        if prob_pos is not None:
            probs_all.append(prob_pos)

# concat results and print metrics
if len(y_true_all) > 0:
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    print("\n=== Batch Evaluation ===")
    print(f"Samples: {len(y_true_all)}")
    print(f"Accuracy : {accuracy_score(y_true_all, y_pred_all):.4f}")
    print(f"Precision: {precision_score(y_true_all, y_pred_all, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true_all, y_pred_all, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_true_all, y_pred_all, zero_division=0):.4f}")
    print("\nDetailed report:")
    print(classification_report(y_true_all, y_pred_all, zero_division=0))
else:
    print("[!] No ground_truth found in CSV - predictions done but no evaluation printed.")
