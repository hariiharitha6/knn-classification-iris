"""
Task 6: K-Nearest Neighbors (KNN) Classification
Dataset: Iris Dataset
Tools: Scikit-learn, Pandas, Matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load & Explore Dataset
# ─────────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 55)
print("       K-Nearest Neighbors — Iris Classification")
print("=" * 55)
print(f"\nDataset shape : {df.shape}")
print(f"Classes       : {list(iris.target_names)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nClass distribution:\n{df['species'].value_counts()}")

# ─────────────────────────────────────────────
# 2. Feature Normalization
# ─────────────────────────────────────────────
X = df[iris.feature_names].values
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTrain size: {X_train_scaled.shape[0]}  |  Test size: {X_test_scaled.shape[0]}")

# ─────────────────────────────────────────────
# 3. Experiment with Different K Values
# ─────────────────────────────────────────────
k_range = range(1, 21)
train_accuracies, test_accuracies = [], []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_accuracies.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    test_accuracies.append(accuracy_score(y_test,  knn.predict(X_test_scaled)))

best_k = k_range[np.argmax(test_accuracies)]
best_acc = max(test_accuracies)
print(f"\nBest K = {best_k}  |  Test Accuracy = {best_acc:.4f}")

# ─────────────────────────────────────────────
# 4. Final Model with Best K
# ─────────────────────────────────────────────
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred = knn_best.predict(X_test_scaled)

print(f"\nClassification Report (K={best_k}):\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ─────────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.suptitle("KNN Classification — Iris Dataset", fontsize=18, fontweight='bold', y=0.98)

colors = ['#E63946', '#457B9D', '#2A9D8F']
cmap   = plt.cm.RdYlGn

# --- Plot 1: K vs Accuracy ---
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(k_range, train_accuracies, 'b-o', label='Train', markersize=5)
ax1.plot(k_range, test_accuracies,  'r-s', label='Test',  markersize=5)
ax1.axvline(best_k, color='green', linestyle='--', label=f'Best K={best_k}')
ax1.set_xlabel('K'); ax1.set_ylabel('Accuracy')
ax1.set_title('K vs Accuracy')
ax1.legend(); ax1.grid(alpha=0.3)

# --- Plot 2: Confusion Matrix ---
ax2 = fig.add_subplot(3, 3, 2)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title(f'Confusion Matrix (K={best_k})')

# --- Plot 3: Feature Distributions ---
ax3 = fig.add_subplot(3, 3, 3)
for i, (cls, col) in enumerate(zip(iris.target_names, colors)):
    subset = df[df['species'] == cls]['sepal length (cm)']
    ax3.hist(subset, alpha=0.7, label=cls, color=col, bins=12)
ax3.set_xlabel('Sepal Length (cm)'); ax3.set_ylabel('Count')
ax3.set_title('Sepal Length Distribution'); ax3.legend(); ax3.grid(alpha=0.3)

# --- Plot 4: Sepal Decision Boundary ---
ax4 = fig.add_subplot(3, 3, 4)
feat_idx = [0, 1]  # sepal length & width
X2 = X[:, feat_idx]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)
scaler2 = StandardScaler()
X2_train_s = scaler2.fit_transform(X2_train)
X2_test_s  = scaler2.transform(X2_test)
knn2 = KNeighborsClassifier(n_neighbors=best_k)
knn2.fit(X2_train_s, y2_train)
h = 0.02
x_min, x_max = X2_train_s[:,0].min()-1, X2_train_s[:,0].max()+1
y_min, y_max = X2_train_s[:,1].min()-1, X2_train_s[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax4.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
scatter_colors = [colors[c] for c in y2_train]
ax4.scatter(X2_train_s[:,0], X2_train_s[:,1], c=scatter_colors, edgecolors='k', s=30)
ax4.set_xlabel('Sepal Length (scaled)'); ax4.set_ylabel('Sepal Width (scaled)')
ax4.set_title(f'Decision Boundary — Sepal (K={best_k})')
patches = [mpatches.Patch(color=colors[i], label=iris.target_names[i]) for i in range(3)]
ax4.legend(handles=patches, fontsize=7)

# --- Plot 5: Petal Decision Boundary ---
ax5 = fig.add_subplot(3, 3, 5)
feat_idx2 = [2, 3]  # petal length & width
X3 = X[:, feat_idx2]
X3_train, X3_test, y3_train, _ = train_test_split(X3, y, test_size=0.2, random_state=42)
scaler3 = StandardScaler()
X3_train_s = scaler3.fit_transform(X3_train)
knn3 = KNeighborsClassifier(n_neighbors=best_k)
knn3.fit(X3_train_s, y3_train)
x_min2, x_max2 = X3_train_s[:,0].min()-1, X3_train_s[:,0].max()+1
y_min2, y_max2 = X3_train_s[:,1].min()-1, X3_train_s[:,1].max()+1
xx2, yy2 = np.meshgrid(np.arange(x_min2,x_max2,h), np.arange(y_min2,y_max2,h))
Z2 = knn3.predict(np.c_[xx2.ravel(), yy2.ravel()]).reshape(xx2.shape)
ax5.contourf(xx2, yy2, Z2, alpha=0.3, cmap=plt.cm.Set1)
scatter_colors3 = [colors[c] for c in y3_train]
ax5.scatter(X3_train_s[:,0], X3_train_s[:,1], c=scatter_colors3, edgecolors='k', s=30)
ax5.set_xlabel('Petal Length (scaled)'); ax5.set_ylabel('Petal Width (scaled)')
ax5.set_title(f'Decision Boundary — Petal (K={best_k})')
ax5.legend(handles=patches, fontsize=7)

# --- Plot 6: Pairplot-style scatter ---
ax6 = fig.add_subplot(3, 3, 6)
for i, (cls, col) in enumerate(zip(iris.target_names, colors)):
    mask = y == i
    ax6.scatter(X[mask, 2], X[mask, 3], label=cls, color=col, alpha=0.7, edgecolors='k', s=40)
ax6.set_xlabel('Petal Length (cm)'); ax6.set_ylabel('Petal Width (cm)')
ax6.set_title('Petal Features — All Classes'); ax6.legend(); ax6.grid(alpha=0.3)

# --- Plot 7: Test Accuracy bar across K ---
ax7 = fig.add_subplot(3, 3, 7)
bar_colors = ['green' if k == best_k else '#457B9D' for k in k_range]
ax7.bar(k_range, test_accuracies, color=bar_colors, edgecolor='k', linewidth=0.5)
ax7.set_xlabel('K'); ax7.set_ylabel('Test Accuracy')
ax7.set_title('Test Accuracy for Each K'); ax7.grid(axis='y', alpha=0.3)
ax7.set_xticks(list(k_range))

# --- Plot 8: Sepal scatter ---
ax8 = fig.add_subplot(3, 3, 8)
for i, (cls, col) in enumerate(zip(iris.target_names, colors)):
    mask = y == i
    ax8.scatter(X[mask, 0], X[mask, 1], label=cls, color=col, alpha=0.7, edgecolors='k', s=40)
ax8.set_xlabel('Sepal Length (cm)'); ax8.set_ylabel('Sepal Width (cm)')
ax8.set_title('Sepal Features — All Classes'); ax8.legend(); ax8.grid(alpha=0.3)

# --- Plot 9: Per-class accuracy bar ---
ax9 = fig.add_subplot(3, 3, 9)
report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
cls_names = iris.target_names
f1_scores = [report[c]['f1-score'] for c in cls_names]
ax9.bar(cls_names, f1_scores, color=colors, edgecolor='k')
ax9.set_ylim(0, 1.1); ax9.set_ylabel('F1-Score')
ax9.set_title(f'Per-Class F1-Score (K={best_k})'); ax9.grid(axis='y', alpha=0.3)
for i, v in enumerate(f1_scores):
    ax9.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('knn_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: knn_results.png")
print("\nDone!")
