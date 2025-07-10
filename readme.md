# Handwritten Digit Classification with SVM

Classify MNIST digits (0 – 9) using a Support Vector Machine with an RBF kernel, plus k‑NN and multinomial logistic‑regression baselines.  The workflow loads the **original IDX files** locally, performs standard scaling (and optional PCA), tunes hyper‑parameters by cross‑validation, evaluates on the official test set, then saves the best model for instant reuse.

---

## Project structure

```text
├── data/                        # four MNIST IDX files (download or place here)
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
├── mnist_idx_svm.py            # main Python script
├── requirements.txt            # pip‑installable dependencies
└── README.md                   # you are here
```

---

## Requirements

| Package        | Tested version | Why it matters                                    |
| -------------- | -------------- | ------------------------------------------------- |
| `Python`       | 3.9 +          | any modern interpreter                            |
| `numpy`        | 1.25           | fast array maths                                  |
| `scikit‑learn` | 1.5            | SVM, k‑NN, logistic regression, PCA, GridSearchCV |
| `matplotlib`   | 3.9            | confusion‑matrix plot                             |
| `joblib`       | 1.5            | model persistence                                 |

Install everything in one shot:

```bash
python -m pip install -r requirements.txt
```

---

## Quick‑start

```bash
# 1. Place the four IDX files inside ./data/

# 2. Train SVM + baselines (default full grid)
python mnist_idx_svm.py

Script saves the tuned pipeline to `mnist_rbf_svm.joblib` and prints a confusion matrix.

### Reusing the trained model

```python
from joblib import load
clf = load("mnist_rbf_svm.joblib")
probs = clf.predict_proba(new_images)  # if probability=True was set
```

---

## Script overview (`mnist_idx_svm.py`)

| Block                    | Purpose                                                           |
| ------------------------ | ----------------------------------------------------------------- |
| **Imports**              | bring in `pathlib`, `numpy`, plotting, scikit‑learn classes       |
| ``                       | parse binary IDX files → NumPy arrays in one pass                 |
| **Data load**            | read four files, sanity‑check shapes                              |
| **Pre‑processing**       | flatten to 784‑D, cast to `float32`                               |
| **Pipeline**             | `StandardScaler` → (optional) `PCA` → `SVC(kernel='rbf')`         |
| **Hyper‑parameter grid** | search `C ∈ {1,5,10,50}`, `γ ∈ {1e‑3,5e‑4,1e‑4}`                  |
| ``                       | 5‑fold stratified CV, parallel across CPU cores                   |
| **Evaluation**           | hold‑out accuracy, `classification_report`, confusion matrix      |
| **Baselines**            | k‑NN (k = 3, distance‑weighted) & multinomial logistic regression |
| **Persistence**          | save best pipeline with `joblib.dump`                             |

---

## Results

| Model                     | Test accuracy (typical) |
| ------------------------- | ----------------------- |
| **RBF‑SVM (tuned)**       | **≈ 98.5 %**            |
| k‑NN (k = 3)              | ≈ 97.2 %                |
| Multinomial Logistic Reg. | ≈ 92–93 %               |

*Digits 4 vs 9 and 3 vs 5 are the most frequent confusions; PCA lowers accuracy by \~0.2 pp but yields a \~6× speed‑up.*

---

## Customising & speeding up

| Need                | Change                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------- |
| Faster search       | Enable PCA (uncomment), or down‑sample training to 10 k images during grid search then refit full data. |
| Probability outputs | Set `probability=True` in `SVC`; beware double fit time.                                                |
| Larger grids        | Swap to `RandomizedSearchCV(n_iter=30)` to explore wider ranges efficiently.                            |
| Memory tight        | Keep PCA and set `pca_whiten=True`.                                                                     |

---

## References

- LeCun, Y. et al., *Gradient‑based learning applied to document recognition*. Proceedings of the IEEE, 1998.  (MNIST original paper)
- scikit‑learn user guide – [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)

---

## © 2025  Shashidhar Reddy

