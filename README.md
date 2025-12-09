# Linear Regression in Java
This project implements Linear Regression from scratch in Java, without using any machine learning libraries.

## Features
* Linear Regression using Batch Gradient Descent
* Categorical feature handling via internal label encoding
* Custom DataFrame abstraction for tabular data
* Supports:
    * training from CSV
    * single-row prediction
    * batch prediction
* Mean Squared Error (MSE) loss
* Clean separation of:
    * data storage
    * encoding
    * model logic

## Project Structure
```txt
.
├── DataFrame.java        // Custom tabular data structure
├── EncodeData.java       // Categorical encoder
├── LinearRegression.java // Linear Regression model
├── Test.java // Test / demo
└── README.md
```

## Algorithm Details
### Hypothesis Function
```math
\hat{y} = \mathbf{w}^T \mathbf{x} + b
```
Where:
* `w` = learned Weights
* `b` = learned Bias
### Loss Function (MSE)
```math
\mathcal{L}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```
### Optimization
* Batch Gradient Descent
* Weight update:
```math
\mathbf{w}^{(t+1)}
= \mathbf{w}^{(t)} - \alpha \cdot \frac{2}{n}
\sum_{i=1}^{n} (\hat{y}_i - y_i)\mathbf{x}_i
```
* Bias update:
```math
b^{(t+1)}
= b^{(t)} - \alpha \cdot \frac{2}{n}
\sum_{i=1}^{n} (\hat{y}_i - y_i)
```
## Categorical Encoding
* Automatically detects non-numeric columns
* Encodes categories per feature
* Encoder is learned during training and reused during prediction
* Unknown categories during inference throw an explicit error

This prevents:
* silent data leakage
* inconsistent feature mappings

# Possible Extensions
* One-Hot Encoding
* Feature normalization
* Logistic Regression
* Ridge / Lasso Regression
* R² and MAE metrics
* Mini-batch gradient descent
