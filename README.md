# 🧠 PyTorch MLP Optimization & Regularization

This project implements a modular **deep learning training pipeline** for Multi-Layer Perceptrons (MLPs) using PyTorch. It explores optimization dynamics, weight decay regularization, and evaluation methods. The goal is to demonstrate best practices in model building, training, and analysis.

---

## 🔧 **Features**

* **Model Architecture**

  * Flexible MLP builder with configurable depth and hidden units
  * Modular `nn.Module` design for reuse

* **Training Pipeline**

  * Custom training loop with `SGD` and `CrossEntropyLoss`
  * Epoch-level loss tracking and optimizer updates

* **Regularization**

  * Implementation of **weight decay** applied only to weight parameters
  * Comparison of regularized vs non-regularized training

* **Evaluation**

  * Accuracy calculation on validation/test sets
  * Parameter extraction utilities for model introspection

* **Experimentation**

  * Easy hyperparameter tuning: learning rate, layers, hidden size, weight decay
  * PCA integration for analyzing parameter/feature spaces

---

## 📌 **Technologies Used**

* Python
* Jupyter Notebook
* **PyTorch** (`torch`, `torchvision`)
* **scikit-learn**
* **numpy**, **matplotlib**

---

## 📈 **Methodology**

* **MLP Construction**

  * Function `build_mlp()` creates feed-forward models with arbitrary layers
* **Training**

  * `train_epoch()` handles forward pass, backpropagation, and optimizer updates
* **Weight Decay**

  * `sgd_weight_decay_weights_only()` implements selective weight decay to prevent bias/variance terms from being penalized
* **Evaluation**

  * `evaluate()` computes classification accuracy
  * `extract_model_params()` flattens all trainable parameters for analysis
* **Experimentation**

  * Torchvision datasets (e.g., MNIST/Fashion-MNIST) used as examples
  * PCA via `sklearn.decomposition` for dimensionality reduction in analysis

---

## 🧠 **Learning Goals**

* Understand **MLP construction** and training loops in PyTorch
* Apply **regularization techniques** like weight decay correctly
* Practice writing **reusable training utilities** for deep learning
* Explore model parameter analysis with PCA and evaluation metrics

---

## 🧮 **Future Enhancements**

* Add support for other optimizers (Adam, RMSprop)
* Implement early stopping and learning rate scheduling
* Extend evaluation with confusion matrices and precision/recall
* Add visualization of training dynamics (loss/accuracy curves)

---

## 📂 Dataset

This project uses datasets from **torchvision** (e.g., MNIST, Fashion-MNIST).
They are automatically downloaded via the PyTorch `datasets` API.
