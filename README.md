# Iris Flower Classification

This project involves building a machine learning model to classify Iris flowers into three species Iris-setosa, Iris-versicolor, and Iris-virginica using the famous **Iris dataset**. The project includes data preprocessing, exploratory data analysis (EDA), training of multiple models, and selection of the best model.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Model Training](#model-training)
5. [Results](#results)
6. [Saving the Model](#saving-the-model)
7. [How to Run](#how-to-run)
8. [Requirements](#requirements)

## Project Overview

The goal of this project is to classify Iris flowers into three different species based on four features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

We train and evaluate the following machine learning models:
- Logistic Regression
- k-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

The best model is selected based on accuracy and other evaluation metrics, and is then saved for future use.

## Dataset

The dataset used is the **Iris dataset**, which contains 150 samples of iris flowers, with 50 samples for each of the three species. The dataset has the following features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
- Species (Target variable)

The dataset can be accessed from the following link:
[Iris Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/iris)

## Exploratory Data Analysis (EDA)

EDA was performed to understand the structure of the dataset and visualize relationships between features. Key steps include:
- Checking for null values.
- Plotting histograms for each feature.
- Creating scatterplots for feature pair comparisons grouped by species.
- Generating correlation matrices to check for multicollinearity.
- Creating box plots to examine the distribution of each feature.

## Model Training

Four machine learning models were trained and evaluated:

1. **Logistic Regression**
2. **k-Nearest Neighbors (KNN)**
3. **Decision Tree**
4. **Random Forest**

The dataset was split into training and testing sets (80% training, 20% testing). After feature scaling, each model was trained, and accuracy and confusion matrices were computed to evaluate their performance.

### Model Evaluation Summary:

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 96.67%   |
| k-Nearest Neighbors | 90.00%   |
| Decision Tree       | 93.33%   |
| Random Forest       | 96.67%   |

The **Random Forest** model was selected as the best-performing model based on accuracy and robustness.

## Results

Here are the results of the best model (**Random Forest**) on the test dataset:

| Actual Species   | Predicted Species |
|------------------|-------------------|
| Iris-versicolor  | Iris-versicolor    |
| Iris-virginica   | Iris-virginica     |
| Iris-virginica   | Iris-virginica     |
| Iris-versicolor  | Iris-versicolor    |
| Iris-setosa      | Iris-setosa        |

You can find the complete evaluation metrics such as accuracy, confusion matrix, and classification report in the [notebook](https://github.com/IsharaParanagamaGedara/Iris-Flower-Classification/blob/main/iris-dataset-analysis-classification.ipynb).

## Saving the Model

The best-performing model (**Random Forest**) was saved as a `.pkl` file for future use. You can load and use this model for new predictions without retraining.

```python
# Saving the Random Forest model
joblib.dump(rf_model, 'random_forest_model.pkl')
```

To load the model:
```python
# Loading the saved model
loaded_model = joblib.load('random_forest_model.pkl')
```

## How to Run

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/IsharaParanagamaGedara/Iris-Flower-Classification.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook or Python script to train the models and evaluate them.


## Requirements

Here are the key dependencies for this project:

- Python 3.x
- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib
- joblib



