# Loan-Approval-Prediction-using-Machine-Learning
This project demonstrates a Loan Approval Prediction system using a Random Forest Classifier trained on a synthetic dataset. The goal is to classify whether a loan should be approved or rejected based on various applicant features.
# Import necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=3, random_state=42)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = rfc.predict(X_test)
# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
Accuracy: 0.975
Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.99      0.98       101
           1       0.99      0.96      0.97        99

    accuracy                           0.97       200
   macro avg       0.98      0.97      0.97       200
weighted avg       0.98      0.97      0.97       200

Confusion Matrix:
[[100   1]
 [  4  95]]
