
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def main():
    
    data = pd.read_csv('your_dataset.csv')
    
   
    print("Dataset Information:")
    print(data.info())
    print("\nDataset Description:")
    print(data.describe())
    print("\nMissing Values:\n", data.isnull().sum())

    
    print("\nHandling missing values...")
    imputer = SimpleImputer(strategy='mean')  
    data.iloc[:, :] = imputer.fit_transform(data)
    print("Missing values handled successfully.\n")

    
    print("Encoding categorical variables...")
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le  
    print("Categorical encoding completed.\n")

   
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data.drop('target', axis=1))  
    X = pd.DataFrame(scaled_features, columns=data.columns.drop('target'))
    y = data['target']  
    print("Feature scaling completed.\n")

    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split successfully.\n")

    
    print("Training the Random Forest model...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    print("Model training completed.\n")

    
    print("Making predictions on the test set...")
    y_pred = rf.predict(X_test)
    print("Predictions completed.\n")

   
    print("Evaluating model performance...")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()
