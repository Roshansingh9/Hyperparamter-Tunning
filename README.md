# Deep Learning Practice: Hyperparameter Tuning for Diabetes Prediction

This repository contains a Jupyter Notebook where I practice deep learning concepts, focusing on hyperparameter tuning using the Pima Indians Diabetes dataset. The notebook demonstrates how to build, train, and optimize neural networks with Keras and Keras Tuner.

## Overview

- **Goal:** Predict diabetes using a neural network and explore the impact of different hyperparameters.
- **Techniques Practiced:**
  - Data preprocessing and scaling
  - Building neural networks with Keras
  - Hyperparameter tuning (optimizers, number of neurons, number of layers, activation functions, dropout rates)
  - Model evaluation

## File Structure

- `Hyperparamter Tunning/demo.ipynb` — Main notebook with all code and experiments.
- `Hyperparamter Tunning/diabetes.csv` — Dataset used for training and testing.
- Output directories for Keras Tuner trials (e.g., `No_of_neurons_hyperparam_tuning`, `No_of_layers_hyperparam_tuning`, `hyperparam_tuning`).

## Requirements

- Python 3.7+
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- tensorflow
- keras
- keras-tuner (or kerastuner)
- colorama (optional, for colored output)

Install dependencies with:

```sh
pip install pandas numpy scikit-learn tensorflow keras keras-tuner colorama
```

## How to Use

1. Open `demo.ipynb` in Jupyter Notebook or VS Code.
2. Run the cells step by step to:
   - Load and preprocess the data
   - Build and train baseline models
   - Tune hyperparameters (optimizer, neurons, layers, etc.)
   - Evaluate model performance

## What I Learned

- How to use Keras Tuner for hyperparameter optimization
- The effect of different optimizers, layer sizes, and architectures on model accuracy
- Best practices for structuring deep learning experiments

## Dataset

The [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) contains medical data for predicting diabetes.

## References

- [Keras Tuner Documentation](https://keras.io/keras_tuner/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---

\*This notebook is for learning and experimentation
