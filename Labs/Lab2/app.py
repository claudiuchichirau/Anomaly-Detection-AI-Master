import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from pyod.utils.data import generate_data_clusters
from pyod.utils.utility import standardizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score 
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.datasets import make_blobs
from scipy.io import loadmat
from pyod.models.combination import average, maximization

# Ex.1 

# 1D Linear Expression
a, b = 2, 5
n_samples = 100
x = np.random.rand(n_samples, 1) * 10
eps = np.random.normal(0, 1, size=(n_samples, 1))
y = a * x + b + eps


X_1D = np.hstack((x, np.ones_like(x)))  
XTX_inv_1D = np.linalg.pinv(X_1D.T @ X_1D)
leverage_scores_1D = np.sum(X_1D * (X_1D @ XTX_inv_1D), axis=1)


plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Data points")
plt.scatter(x[leverage_scores_1D > np.quantile(leverage_scores_1D, 0.9)], 
            y[leverage_scores_1D > np.quantile(leverage_scores_1D, 0.9)], 
            color="red", label="High leverage points")
plt.legend()
plt.title("Leverage Scores for 1D Linear Model")
plt.show()

# 2D Linear Expression
a, b, c = 2, -3, 5 
x1 = np.random.rand(n_samples, 1) * 10
x2 = np.random.rand(n_samples, 1) * 10
eps = np.random.normal(0, 1, size=(n_samples, 1))
y = a * x1 + b * x2 + c + eps


X_2D = np.hstack((x1, x2, np.ones_like(x1)))
XTX_inv_2D = np.linalg.pinv(X_2D.T @ X_2D)
leverage_scores_2D = np.sum(X_2D * (X_2D @ XTX_inv_2D), axis=1)


fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(x1, x2, c=leverage_scores_2D, cmap="viridis", label="Data points")
plt.colorbar(sc, label="Leverage score")
high_leverage = leverage_scores_2D > np.quantile(leverage_scores_2D, 0.9)
ax.scatter(x1[high_leverage], x2[high_leverage], color="red", label="High leverage points")
ax.legend()
plt.title("Leverage Scores for 2D Linear Model")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# Ex.2
X, y = generate_data_clusters(
    n_train=400,          
    n_test=200,            
    n_clusters=2,          
    n_features=2,          
    contamination=0.1,    
    return_in_clusters=True)

X = np.array(X)
y = np.array(y)
X = np.concatenate(X, axis=0)  # single array
y = np.concatenate(y, axis=0)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

y_train_labels = np.zeros(len(y_train), dtype=int)
y_train_labels[-int(0.1 * len(y_train)):] = 1  # 10% as outliers for training
y_test_labels = np.zeros(len(y_test), dtype=int)
y_test_labels[-int(0.1 * len(y_test)):] = 1  # 10% as outliers for test

neighbors = [5, 10, 20]  
results = []

for n in neighbors:
    knn = KNN(n_neighbors=n)
    knn.fit(X_train)

    y_train_pred = knn.labels_  
    y_test_pred = knn.predict(X_test)  

    train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    results.append((n, train_bal_acc, test_bal_acc))

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"KNN with n_neighbors={n}")

    axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="k")
    axs[0, 0].set_title("Ground Truth - Training Data")

    axs[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred, cmap="coolwarm", edgecolors="k")
    axs[0, 1].set_title("Predicted Labels - Training Data")

    axs[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k")
    axs[1, 0].set_title("Ground Truth - Test Data")

    axs[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap="coolwarm", edgecolors="k")
    axs[1, 1].set_title("Predicted Labels - Test Data")

    plt.tight_layout()
    plt.show()

for n, train_bal_acc, test_bal_acc in results:
    print(f"n_neighbors = {n} | Balanced Accuracy (Train): {train_bal_acc:.2f} | Balanced Accuracy (Test): {test_bal_acc:.2f}")


# Ex.3
n_samples_1 = 200
n_samples_2 = 100
centers = [(-10, -10), (10, 10)]
std_devs = [2, 6]  

X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers, cluster_std=std_devs, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

neighbors = [5, 10]  
results = []

for n in neighbors:
    # KNN
    knn = KNN(n_neighbors=n, contamination=0.07)
    knn.fit(X_train)
    y_train_pred_knn = knn.labels_ 
    y_test_pred_knn = knn.predict(X_test)  

    # LOF
    lof = LOF(n_neighbors=n, contamination=0.07)
    lof.fit(X_train)
    y_train_pred_lof = lof.labels_  
    y_test_pred_lof = lof.predict(X_test) 

    results.append((n, y_train_pred_knn, y_test_pred_knn, y_train_pred_lof, y_test_pred_lof))

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("KNN vs LOF on Clusters with Different Densities")

for i, (n, y_train_pred_knn, y_test_pred_knn, y_train_pred_lof, y_test_pred_lof) in enumerate(results):
    # KNN Training 
    axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred_knn, cmap="coolwarm", edgecolors="k", label=f"KNN (n_neighbors={n})")
    axs[0, 0].set_title("KNN Training Predictions")
    axs[0, 0].set_xlim(-15, 15)
    axs[0, 0].set_ylim(-15, 15)

    # KNN Testing 
    axs[0, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred_knn, cmap="coolwarm", edgecolors="k", label=f"KNN (n_neighbors={n})")
    axs[0, 1].set_title("KNN Testing Predictions")
    axs[0, 1].set_xlim(-15, 15)
    axs[0, 1].set_ylim(-15, 15)

    # LOF Training 
    axs[1, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred_lof, cmap="coolwarm", edgecolors="k", label=f"LOF (n_neighbors={n})")
    axs[1, 0].set_title("LOF Training Predictions")
    axs[1, 0].set_xlim(-15, 15)
    axs[1, 0].set_ylim(-15, 15)

    # LOF Testing 
    axs[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred_lof, cmap="coolwarm", edgecolors="k", label=f"LOF (n_neighbors={n})")
    axs[1, 1].set_title("LOF Testing Predictions")
    axs[1, 1].set_xlim(-15, 15)
    axs[1, 1].set_ylim(-15, 15)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# Ex.4
data = loadmat('cardio.mat')
X = data['X']  
y = data['y'].flatten()  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

n_neighbors_range = range(30, 121, 10) 
train_scores = []
test_scores = []

for n_neighbors in n_neighbors_range:
    knn = KNN(n_neighbors=n_neighbors, contamination=0.1)  # 10% contamination
    knn.fit(X_train)
    
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    
    train_ba = balanced_accuracy_score(y_train, y_train_pred)
    test_ba = balanced_accuracy_score(y_test, y_test_pred)
    
    train_scores.append(train_ba)
    test_scores.append(test_ba)
    
    print(f"KNN (n_neighbors={n_neighbors}) - Train BA: {train_ba:.4f}, Test BA: {test_ba:.4f}")

train_scores_normalized = standardizer(np.array(train_scores).reshape(-1, 1))
test_scores_normalized = standardizer(np.array(test_scores).reshape(-1, 1))

train_predictions = []
test_predictions = []

for n_neighbors in n_neighbors_range:
    knn = KNN(n_neighbors=n_neighbors, contamination=0.1)  # 10% contamination
    knn.fit(X_train)
    
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    
    train_predictions.append(y_train_pred)
    test_predictions.append(y_test_pred)


train_predictions_array = np.array(train_predictions)  
test_predictions_array = np.array(test_predictions)    

y_train_pred_avg = np.mean(train_predictions_array, axis=0) > 0.5 
y_test_pred_avg = np.mean(test_predictions_array, axis=0) > 0.5    


y_train_pred_max = np.max(train_predictions_array, axis=0)          
y_test_pred_max = np.max(test_predictions_array, axis=0)            

print(f"\nAverage Strategy - Train BA: {balanced_accuracy_score(y_train, y_train_pred_avg):.4f}, Test BA: {balanced_accuracy_score(y_test, y_test_pred_avg):.4f}")
print(f"Maximization Strategy - Train BA: {balanced_accuracy_score(y_train, y_train_pred_max):.4f}, Test BA: {balanced_accuracy_score(y_test, y_test_pred_max):.4f}")