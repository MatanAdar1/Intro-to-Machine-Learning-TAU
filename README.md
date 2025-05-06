*Intro to Machine Learning – TAU*
Author: Matan Adar
Course: Introduction to Machine Learning
Institution: Tel Aviv University

🧠 Overview
This repository contains theoretical and practical assignments completed during the Intro to Machine Learning course at TAU. It includes analytical problem-solving, mathematical proofs, and hands-on implementations of machine learning models such as a single-layer neural network and the ID3 decision tree algorithm.

📁 Contents
1. 📄 Theoretical Assignments (PDFs)
HW1: Naive Bayes, Bayesian decision-making, PAC learnability, VC-dimension analysis.

HW2: Convexity proofs, Jensen's inequality, and logistic loss exploration.

HW3: SVM optimization, kernel analysis, and decision tree metrics like entropy and Gini.

Gradient Descent & ID3 Report: Comparison of single-layer NN vs ID3 using the Wisconsin Breast Cancer dataset.

2. 🧪 Python Implementations
part1-SLNN.py:
Implementation of a single-layer neural network with sigmoid activation. Trains using cross-entropy loss and manual gradient descent. Includes multiple learning rates and weight initialization strategies, along with loss plots and accuracy evaluation.

part2-ID3.py:
ID3 decision tree built from scratch. Handles numeric data, splits on information gain, includes recursive tree construction and prediction, with tree visualization and accuracy calculation.

See the PDF comparison report and Python logs for full details.

📈 Visualizations
Train/test loss curves per epoch

Grid plots comparing different learning rates and initialization types

Final weights, biases, and accuracy metrics

Plots are saved as PNG files and discussed in the report (תרגיל פייתון מבוא ללמידת מכונה-מתןPDF).

🛠️ Tech Stack
Python 3

NumPy

Matplotlib

scikit-learn (for dataset loading only)

🚀 How to Run
Clone the repository

Run part1-SLNN.py or part2-ID3.py with Python 3.

Outputs include terminal results and .png graphs.

