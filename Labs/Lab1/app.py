import numpy as np
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

#1
X_train, X_test, y_train, y_test = generate_data(
    n_train=400,        
    n_test=100,         
    n_features=2,       # 2-dimensional dataset
    contamination=0.1, 
    random_state=42     
)

plt.figure(figsize=(8, 6))

# normal samples (y_train == 0)
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 
            color='blue', label='Normal')

# outliers (y_train == 1)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
            color='red', label='Outlier')


plt.title('Training Data - Normal Samples and Outliers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.show()



#2
X_train, X_test, y_train, y_test = generate_data(
    n_train=400,       
    n_test=100,         
    n_features=2,       # 2-dimensional dataset
    contamination=0.1,  
    random_state=42     
)

knn = KNN(contamination=0.1)  
knn.fit(X_train)             

y_train_pred = knn.predict(X_train)  # predictions for training data
y_test_pred = knn.predict(X_test)    # predictions for test data

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
tn_test, fp_test, fn_test, tp_test = cm_test.ravel()

train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)

print(f"2.\nTraining Balanced Accuracy: {train_balanced_accuracy:.2f}")
print(f"Testing Balanced Accuracy: {test_balanced_accuracy:.2f}")

print("\nTraining Confusion Matrix:")
print(f"TN: {tn_train}, FP: {fp_train}, FN: {fn_train}, TP: {tp_train}")

print("\nTesting Confusion Matrix:")
print(f"TN: {tn_test}, FP: {fp_test}, FN: {fn_test}, TP: {tp_test}")

# ROC curve for the test set
y_test_scores = knn.decision_function(X_test)  # Get the decision function scores
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)
roc_auc = auc(fpr, tpr)

# plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# different contamination rate
knn_high_contamination = KNN(contamination=0.2)
knn_high_contamination.fit(X_train)

y_test_pred_high = knn_high_contamination.predict(X_test)
cm_test_high = confusion_matrix(y_test, y_test_pred_high)
tn_test_high, fp_test_high, fn_test_high, tp_test_high = cm_test_high.ravel()

test_balanced_accuracy_high = balanced_accuracy_score(y_test, y_test_pred_high)

print(f"\nTesting Balanced Accuracy with High Contamination: {test_balanced_accuracy_high:.2f}")
print("\nTesting Confusion Matrix with High Contamination:")
print(f"TN: {tn_test_high}, FP: {fp_test_high}, FN: {fn_test_high}, TP: {tp_test_high}")


#3
X_train, X_test, y_train, y_test = generate_data(
    n_train=1000,       
    n_test=0,         
    n_features=1,       # unidimensional dataset
    contamination=0.1,  
    random_state=42     
)

# Z-scores for the training data
mean = np.mean(X_train)
std_dev = np.std(X_train)
z_scores = (X_train - mean) / std_dev

threshold = np.quantile(np.abs(z_scores), 0.9) 

# classify anomalies based on the Z-score threshold
y_train_pred = (np.abs(z_scores) > threshold).astype(int)

cm_train = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = cm_train.ravel()

train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)

print(f"\n\n3.\nZ-score Threshold: {threshold:.2f}")
print(f"Training Balanced Accuracy: {train_balanced_accuracy:.2f}")

print("\nTraining Confusion Matrix:")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")


#4
np.random.seed(42)  # For reproducibility

n_samples = 1000
contamination_rate = 0.1
n_features = 3  # 3D dataset

means = [5, 10, 15]  
std_devs = [1, 2, 3] 

# 90% normal points 
X_normal = np.column_stack([np.random.normal(loc=mean, scale=std, size=int(n_samples * (1 - contamination_rate)))
                            for mean, std in zip(means, std_devs)])

# 10% outliers
X_outliers = np.column_stack([np.random.normal(loc=mean, scale=std*5, size=int(n_samples * contamination_rate))
                              for mean, std in zip(means, std_devs)])

X_train = np.vstack([X_normal, X_outliers])

# 0 - normal, 1 - outliers
y_train = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_outliers))])

means = np.mean(X_train, axis=0)
std_devs = np.std(X_train, axis=0)
z_scores = (X_train - means) / std_devs

combined_z_scores = np.sqrt(np.sum(z_scores**2, axis=1))
threshold = np.quantile(combined_z_scores, 0.9)  

y_train_pred = (combined_z_scores > threshold).astype(int)

cm_train = confusion_matrix(y_train, y_train_pred)
tn, fp, fn, tp = cm_train.ravel()

train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)


print(f"\n\n4.\nZ-score Threshold: {threshold:.2f}")
print(f"Training Balanced Accuracy: {train_balanced_accuracy:.2f}")

print("\nTraining Confusion Matrix:")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")