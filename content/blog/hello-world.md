---
title: "Multi-Layer Perceptrons: An Introduction"
date: 2026-03-25
description: "An introduction to multi-layer perceptrons."
author: "Noah Reef"
tags: ["deep-learning", "neural-networks", "machine-learning"]
---

<style>
  .container {
    max-width: 100% !important;
  }
  figure.center {
    margin-left: auto !important;
    margin-right: auto !important;
    width: 60%;
  }
  figure.center img {
    width: 100%;
    height: auto;
  }
</style>

It would be an understatement to say that Deep Learning, and the multi-layer perceptron (MLP) in particular, has had a profound impact on the field of artificial intelligence and scientific computing. Though it is not the most sophisticated nor the most widely used neural network architecture, it is the foundation for many more complex architectures. In this post, we will explore the basics of MLPs, including their structure, how they work, and how to build one from scratch.

## The Perceptron

A perceptron is a linear classifier that is used to classify data into two categories. It is a simple model that is based on the idea of a neuron, which is a biological cell that is used to process information. The perceptron is a mathematical model that is used to process information in a way that is similar to how a neuron processes information. We often picture it with the diagram shown below:

{{< figure src="/images/perceptron.png" alt="Multi-Layer Perceptron" position="center" caption="A diagram of a multi-layer perceptron (MLP) architecture." >}}

## Perceptron Implementation

Below is a simple Python implementation of a Perceptron classifier:

```python
import numpy as np

class Perceptron:
    """A simple perceptron classifier for binary classification."""
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the perceptron on input features X and labels y."""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Apply the perceptron learning rule
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Compute the linear output
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Apply the step function
                y_predicted = self.step_function(linear_output)
                
                # Update weights and bias if misclassified
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
    
    @staticmethod
    def step_function(x: float) -> int:
        """Heaviside step function."""
        return 1 if x >= 0 else 0
```

This implementation follows the classic perceptron algorithm: initialize weights to zero, then iteratively update them for each misclassified sample using the learning rate and the difference between the predicted and actual labels.
