#importing the necessary files
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import struct
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries

# Load dataset
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
df.fillna(df.mean(), inplace=True)
df.dropna(inplace=True)

# Sample first 50 rows & first 7 columns for visualization
df_sample = df.iloc[:50, :7]

# Load MNIST Files
image_file = "images-idx3-ubyte"
label_file = "labels-idx1-ubyte"

def read_idx_images(filename):
    with open(filename, 'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        rows, cols = struct.unpack(">II", f.read(8))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)
    return images

def read_idx_labels(filename):
    with open(filename, 'rb') as f:
        _, num_items = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

if os.path.exists(image_file) and os.path.exists(label_file):
    X = read_idx_images(image_file)
    y = read_idx_labels(label_file)
else:
    print("❌ Error: MNIST files not found!")

# Flatten images for Machine Learning
X_flat = X.reshape(X.shape[0], -1) / 255.0  # Normalize pixel values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
model.fit(X_train, y_train)

# Predict Labels
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Print Accuracy Score
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"🔹 Log Loss: {log_loss(y_test, y_pred_proba):.4f}")
print("🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# Display Predictions
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Pred: {y_pred[i]} | Actual: {y_test[i]}")
    plt.axis("off")
plt.show()

# Automatic Image Grid
batch_size = int(input("Enter the number of images per batch: "))
num_batches = int(input("Enter the number of batches to display: "))
x = math.ceil(math.sqrt(batch_size))

for batch in range(num_batches):
    plt.figure(figsize=(10, 10))
    for j in range(batch_size):
        index = batch * batch_size + j
        if index < len(X):
            plt.subplot(x, x, j + 1)
            plt.imshow(X[index], cmap="gray")
            plt.title(f"Label: {y[index]}")
            plt.axis("off")
    plt.show()




import shap
import numpy as np
import matplotlib.pyplot as plt

# Select test image
num_sample = 5  
image = X_test[num_sample].reshape(1, -1)  # Ensure 1D input

# Select background samples for SHAP KernelExplainer
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]  

# Initialize SHAP KernelExplainer
explainer = shap.KernelExplainer(model.predict_proba, background)

# Compute SHAP values
shap_values = explainer.shap_values(image)

# Get predicted class
predicted_class = np.argmax(model.predict_proba(image))

# Extract SHAP values for the predicted class
shap_image = shap_values[0][:, predicted_class].reshape(28, 28)

# Normalize SHAP values for better visualization
shap_image = (shap_image - shap_image.min()) / (shap_image.max() - shap_image.min())

# 🔥 **Improved Visualization**
plt.figure(figsize=(10, 5))

# Show original image
plt.subplot(1, 2, 1)
plt.imshow(image.reshape(28, 28), cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Show SHAP heatmap overlaid on the original image
plt.subplot(1, 2, 2)
plt.imshow(image.reshape(28, 28), cmap="gray", alpha=0.5)  # Add transparency
plt.imshow(shap_image, cmap="jet", alpha=0.6)  # Jet colormap for better visibility
plt.colorbar(label="SHAP Value")
plt.title(f"SHAP Explanation (Class {predicted_class})")
plt.axis("off")

plt.show()


