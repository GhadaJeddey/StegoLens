import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
import joblib

#loading the csv file
csv_file = pd.read_csv('dataset//combined_features.csv')

# Seperate the labels from the features 
label = csv_file['label']
features = csv_file.drop(columns=['label'])



#divide the training set and the learning set 
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

#training the model
model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)

print('training the model ...')
model.fit(x_train, y_train)
print("model trained")

joblib.dump(model,"audio_detection_model.pkl")



'''  
Stats: 


print("Features shape:", features.shape)
print("Labels shape:", label.shape)

    
print('training the model ...')
model.fit(x_train, y_train)
print("model trained")

#prediction
y_pred = model.predict(x_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Full classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Clean", "Stego"], yticklabels=["Clean", "Stego"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


'''







