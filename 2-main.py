"""Letter recognition (A-Z) with an Artificial Neural Network.

Dataset: UCI Letter Recognition. Each row is a letter image described by
16 numeric features (pixel statistics). The ANN predicts which of the 26
capital letters it is.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

cols = ["letter", "x_box", "y_box", "width", "height", "onpix", "x_bar", "y_bar",
        "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x_ege", "xegvy", "y_ege", "yegvx"]
df = pd.read_csv("data/letter-recognition.data", header=None, names=cols)
X = df.drop(columns="letter")
y = df["letter"].map(lambda c: ord(c) - ord("A"))   # A..Z -> 0..25

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_train)       # ANNs need scaled inputs
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# Input(16) -> Dense(128) -> Dense(64) -> Output(26 letters)
ann = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                    max_iter=300, early_stopping=True, random_state=42)
ann.fit(X_train, y_train)

pred = ann.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, target_names=[chr(65 + i) for i in range(26)], digits=2))

sample = X_test[:5]
to_letter = lambda a: [chr(int(i) + ord("A")) for i in a]
print("First 5 predictions:", to_letter(ann.predict(sample)), "actual:", to_letter(y_test[:5]))
