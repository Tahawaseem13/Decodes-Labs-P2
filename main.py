import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


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

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=iris.target_names
)

plt.title("Confusion Matrix")

plt.show()


print("\n========== Predict Your Own Flower ==========")

sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))

# Create Input
new_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

# Scale Input
new_flower = scaler.transform(new_flower)

# Predict
prediction = model.predict(new_flower)

species = iris.target_names[prediction[0]]

print("\n========== Prediction ==========")
print(f"Predicted Species: {species.capitalize()}")



plt.figure(figsize=(6,4))

df["species"].value_counts().plot(kind="bar")

plt.title("Iris Species Distribution")
plt.xlabel("Species")
plt.ylabel("Count")

plt.show()