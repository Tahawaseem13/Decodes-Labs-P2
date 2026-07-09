import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

# Load Iris Dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)


df["species"] = iris.target

df["species"] = df["species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print("========== First 5 Rows ==========")
print(df.head())

print("\n========== Dataset Shape ==========")
print(df.shape)

print("\n========== Column Names ==========")
print(df.columns)

print("\n========== Dataset Information ==========")
print(df.info())

print("\n========== Statistical Summary ==========")
print(df.describe())

print("\n========== Species Count ==========")
print(df["species"].value_counts())

X = iris.data

y = iris.target

print("\n========== Features (First 5 Rows) ==========")
print(X[:5])

print("\n========== Labels (First 5 Values) ==========")
print(y[:5])
# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("\n========== Training Data ==========")
print(X_train.shape)

print("\n========== Testing Data ==========")
print(X_test.shape)


model = KNeighborsClassifier(n_neighbors=5)


model.fit(X_train, y_train)

print("\n========== Model Trained Successfully ==========")

y_pred = model.predict(X_test)

print("\n========== Actual Labels ==========")
print(y_test)

print("\n========== Predicted Labels ==========")
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\n========== Accuracy ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

print("\n========== Confusion Matrix ==========")
print(cm)

print("\n========== Classification Report ==========")
print(classification_report(y_test, y_pred))

f1 = f1_score(y_test, y_pred, average="weighted")

print("\n========== F1 Score ==========")
print(f"F1 Score: {f1:.2f}")