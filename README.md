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
├── LinearRegressionTest.java // Test / demo
└── README.md
```

## Algorithm Details
### Hypothesis Function
```math
\[
\hat{y} = \mathbf{w}^T \mathbf{x} + b
\]
```