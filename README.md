# KNN Classification with Dimensionality Reduction

KNN classifier with PCA and NCA dimensionality reduction for medical classification.

## ✨ Features

- EDA with correlation analysis & visualizations
- Outlier detection (Local Outlier Factor)
- KNN with hyperparameter tuning (GridSearchCV)
- PCA and NCA dimensionality reduction
- Decision boundary visualizations

## 📦 Dependencies

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## 🚀 Quick Start

```bash
python main.py
```

Data path: `../DATAS/data.csv`

## 📈 Results

| Method | Test Accuracy |
|--------|--------------|
| Baseline KNN | ~95% |
| PCA + KNN | ~94% |
| NCA + KNN | ~96% ⭐ |

**Key Finding:** NCA outperforms PCA for KNN (distance-metric optimized)

## 📁 Project Structure

```
├── main.py              # Main script
├── data.csv             # Dataset
├── requirements.txt     # Dependencies
└── outputs/             # Visualizations & results
```

## 🔧 Hyperparameter Tuning

10-fold cross-validation with:
- `n_neighbors`: 1-30
- `weights`: uniform, distance
- `metric`: Minkowski (p=1,2,∞)

## 📊 Outputs

- Correlation heatmaps
- Decision boundaries (PCA & NCA)
- Confusion matrices & accuracy scores
- Misclassified samples visualization

## 📝 Notes

- Data: Binary medical classification (~569 samples, 30 features)
- Outliers removed: ~30 samples (LOF method)
- Train/test split: 70/30
- Random state: Fixed for reproducibility
