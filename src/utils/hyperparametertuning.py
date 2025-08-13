import os
import re
import joblib
import pandas as pd
from pathlib import Path

class HyperparameterTuning:
    def __init__(self):
        self.model_dir = Path(__file__).resolve().parents[2] / "models" / "rf"

    def get_dataframe(self):
        rows = []
        pattern = re.compile(r'rf_fold(\d+)_auc([\d.]+)\.joblib')

        for model_path in self.model_dir.glob("*.joblib"):
            match = pattern.match(model_path.name)
            if match:
                fold = int(match.group(1))
                auc = float(match.group(2))
                try:
                    model = joblib.load(model_path)
                    params = model.get_params()
                    params.update({'fold': fold, 'auc': auc})
                    rows.append(params)
                except Exception as e:
                    print(f"Skipping {model_path.name}: {e}")

        return pd.DataFrame(rows)

if __name__ == "__main__":
    tuner = HyperparameterTuning()
    df = tuner.get_dataframe()
    print(df.sort_values(by='auc'))
