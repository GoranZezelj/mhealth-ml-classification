# MHEALTH Dataset - Activity Classification using Machine Learning

This repository contains the source code used for the master's thesis:

**"Performance Comparison of Machine Learning Algorithms for Classification on the MHEALTH Dataset using Python"**

## 📄 Description

The goal of this project is to evaluate and compare the performance of various supervised machine learning algorithms for human activity recognition based on data from wearable sensors (MHEALTH dataset).

The study focuses on classification accuracy, precision, recall, F1-score, and confusion matrix analysis across different algorithms.

## 🧠 Machine Learning Algorithms Used

- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost
- LightGBM
- Naive Bayes

## 📁 Project Structure

```bash
.
├── data/                      # Contains the MHEALTH dataset
├── models/                    # Training scripts for each ML algorithm
├── results/                   # Generated results (metrics, graphs, matrices)
├── requirements.txt           # List of required Python libraries
├── README.md                  # Project documentation

⚙️ How to Run

    Clone this repository:

git clone https://github.com/your-username/mhealth-ml-classification.git
cd mhealth-ml-classification

    Install dependencies:

pip install -r requirements.txt

    Train and evaluate models:

python models/train_random_forest.py
python models/train_svm.py
...

    View results in the results/ directory.

📊 Evaluation Metrics

    Accuracy

    Precision

    Recall

    F1-score

    Confusion Matrix (with cross-class error analysis)

📦 Dataset

The MHEALTH dataset is available at the UCI Machine Learning Repository:
🔗 https://archive.ics.uci.edu/ml/datasets/mhealth+dataset
📚 Thesis Summary

This repository was created as part of a master's thesis focused on human activity recognition using machine learning techniques. The aim was to determine which algorithm performs best on real-world time-series sensor data from mobile and wearable devices.
👤 Author

Goran Žeželj
Master's Thesis
📧 Email: [gzezelj@yahoo.com]
