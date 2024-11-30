import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


dataset_path = "All_Diets.csv" 
data = pd.read_csv(dataset_path)


data.fillna(data.median(), inplace=True)


data = pd.get_dummies(data, drop_first=True)  


X = data.drop("Outcome", axis=1) 
y = data["Outcome"]  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


svm_model = SVC(kernel='linear')  
svm_model.fit(X_train, y_train)


y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of SVM Classifier: {accuracy * 100:.2f}%")
