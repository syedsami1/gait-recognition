# Activity Recognition using the UCI HAR Dataset

## Overview

This project is focused on building an activity recognition model using the UCI HAR (Human Activity Recognition) dataset. The goal is to classify various physical activities (e.g., walking, sitting, standing) based on sensor data from accelerometers and gyroscopes. The solution leverages deep learning techniques, specifically a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks.

### Key Points:
- **Dataset**: The UCI HAR dataset containing accelerometer and gyroscope readings from 30 participants.
- **Model**: A hybrid CNN-LSTM architecture for time-series classification.
- **Libraries Used**: 
  - `TensorFlow`, `Keras` for deep learning
  - `NumPy`, `Pandas` for data manipulation
  - `Scikit-learn` for preprocessing and evaluation
  - `Matplotlib`, `Seaborn` for visualization

---

## Table of Contents
1. [Installation Instructions](#installation-instructions)
2. [Dataset](#dataset)
3. [Approach](#approach)
4. [Model Building](#model-building)
5. [Results](#results)
6. [How to Use](#how-to-use)
7. [Files](#files)

---

## Installation Instructions

Follow these steps to set up the project environment:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/gait-recognition.git
   cd gait-recognition
