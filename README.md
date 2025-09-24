## **Human Activity Recognition Using UCI HAR Dataset with CNN-LSTM**

This project focuses on building a human activity recognition model using deep learning techniques. The model uses the **UCI Human Activity Recognition (HAR) dataset** and employs a **CNN-LSTM architecture** to classify human activities based on accelerometer and gyroscope sensor data.

### **Project Overview**

The goal is to build a model that can predict human activities such as walking, sitting, and standing from sensor data. The dataset contains measurements from 30 subjects performing six different activities. The model combines **Convolutional Neural Networks (CNN)** for feature extraction and **Long Short-Term Memory (LSTM)** networks for temporal analysis of the data.

---

### **Dataset**

* **Source**: UCI HAR Dataset ([link](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones))
* **Classes**: 6 activities (walking, walking\_upstairs, walking\_downstairs, sitting, standing, laying)
* **Features**: Acceleration and gyroscope data in 3D (X, Y, Z axes)
* **Sample Length**: 128 samples per window

---

### **Project Structure**

```
/project
│
├── /data                 # Processed data files (CSV)
│   ├── employees_uci_plus_realtime_windows.csv    # Processed UCI HAR dataset with real-time data appended
│   └── employees_30_uci_windows.csv               # UCI HAR dataset (only)
│
├── /model                # Model training code and weights
│   ├── gait_cnn_lstm_best.h5    # Best model checkpoint
│   └── gait_cnn_lstm_model    # Saved model in TensorFlow format
│
├── /notebooks             # Jupyter notebooks for model training and evaluation
│   └── human_activity_recognition.ipynb   # Full notebook with training and evaluation code
│
├── /assets                # PowerPoint slides and other documentation
│   └── activity_recognition_presentation.pptx    # Project presentation slides
│
└── README.md             # This file
```

---

### **Installation**

To run this project locally, you will need the following dependencies:

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Required packages**:

   * TensorFlow
   * Scikit-learn
   * Numpy, Pandas
   * Matplotlib, Seaborn

---

### **Data Preparation**

1. **Download the UCI HAR dataset**:

   * The dataset is automatically downloaded and unzipped if not already present when running the script.
   * Alternatively, you can manually download it from [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) and place it in the `UCI HAR Dataset` directory.

2. **Data Processing**:

   * The data is windowed into 128-sample segments.
   * The features are scaled using **StandardScaler**, and **LabelEncoder** is used for subject ID encoding.

3. **Real-time Data** (Optional):

   * You can append your own real-time sensor data by uploading CSV files to the `/realtime_csvs` directory and providing a mapping of filenames to subject IDs.

---

### **Model**

The model is a **CNN-LSTM** architecture consisting of:

* **CNN layers** to extract features from the time-series data.
* **LSTM layers** to capture temporal dependencies.
* **Fully connected layers** for classification.

**Key Hyperparameters**:

* **Conv1D**: 64 filters, kernel size = 3
* **MaxPooling1D**: Pool size = 2
* **LSTM**: 128 units
* **Dropout**: 0.3
* **Optimizer**: Adam
* **Loss function**: Sparse categorical crossentropy

---

### **Training**

The model is trained on the processed data with a validation split of 10% using the **EarlyStopping** callback to prevent overfitting.

```python
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=40,
    batch_size=64,
    callbacks=[es, mc],
    verbose=2
)
```

---

### **Evaluation**

Once trained, the model is evaluated on the test set. The evaluation includes:

* **Test Accuracy**: Measures the model’s classification performance.
* **Confusion Matrix**: Visualizes the classification performance for each activity.

---

### **Real-time Activity Prediction**

The model can be used for real-time activity recognition from a raw 128-sample window of accelerometer data. The model's output is a predicted activity with a confidence score.

```python
def check_in_from_raw_window(raw_window, threshold=0.6):
    """
    raw_window: numpy array shape (128,3) in same units as training windows (ax,ay,az)
    threshold: confidence threshold to allow access
    """
    if raw_window.shape != (ntimesteps, nfeatures):
        raise ValueError(f"raw_window must be shape {(ntimesteps, nfeatures)}")
    scaled = scaler.transform(raw_window.reshape(-1,3)).reshape(1, ntimesteps, nfeatures)
    probs = model.predict(scaled)[0]
    idx = np.argmax(probs)
    prob = probs[idx]
    subj_label = le.inverse_transform([idx])[0]
    if prob >= threshold:
        print(f"✅ ACCESS GRANTED -> subject {subj_label} (confidence {prob:.2f})")
    else:
        print(f"❌ ACCESS DENIED -> top={subj_label}, conf={prob:.2f} < {threshold}")
    return subj_label, float(prob)
```
### **Results**

Below are some important results from the model training and evaluation:

Model Architecture Summary:
The architecture consists of CNN layers for feature extraction and LSTM layers for sequence learning.
<img width="1564" height="625" alt="Screenshot 2025-09-24 213407" src="https://github.com/user-attachments/assets/bf3dee0e-9f31-4a6b-bdd4-1a4befd276a0" />


Training Accuracy Curve:
The training accuracy curve shows how the model learned over multiple epochs.
<img width="677" height="378" alt="accuracy" src="https://github.com/user-attachments/assets/235ad9d3-2935-43ca-a55a-b297067090bf" />

<img width="636" height="872" alt="Screenshot 2025-09-24 213724" src="https://github.com/user-attachments/assets/db3ddd5c-8659-4f2a-9866-705d3e6431b3" />

Training Loss Curve:
The training loss curve tracks the model's performance during training.
<img width="677" height="378" alt="train_val_loss" src="https://github.com/user-attachments/assets/651a4b7b-6aa5-4761-b946-e14f4ac7634c" />


Confusion Matrix:
The confusion matrix below shows how well the model is classifying the activities.
---
<img width="793" height="710" alt="confusion matrix" src="https://github.com/user-attachments/assets/950d5eb0-e47a-4cc7-b029-b2a5e9359c7b" />

<img width="493" height="91" alt="Screenshot 2025-09-24 213758" src="https://github.com/user-attachments/assets/fa9234df-2f3a-48d7-bc04-9f2049f6b66d" />

### **Conclusion**

* The project successfully implements a CNN-LSTM model for human activity recognition using the UCI HAR dataset.
* The model achieves good classification accuracy and can be applied to real-time activity recognition.
* Future work can involve improving the model's generalization with more data and experimenting with different architectures.

---

### **References**

* [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
* **CNN-LSTM models** for time-series classification ([Keras documentation](https://keras.io/))


