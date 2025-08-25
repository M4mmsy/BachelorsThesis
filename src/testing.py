from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm
import numpy as np
import pandas as pd

# Dummy data
X = pd.DataFrame(np.random.rand(100, 5))
y = np.random.choice([0,1], 100)

outer_cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=4, random_state=42)
outer_splits = outer_cv.get_n_splits(X, y)
splits_iter = outer_cv.split(X, y)

for fold_idx, (train_idx, test_idx) in enumerate(tqdm(splits_iter, total=outer_splits, desc="Outer CV folds"), 1):
    print(f"Fold {fold_idx}: train size={len(train_idx)}, test size={len(test_idx)}")