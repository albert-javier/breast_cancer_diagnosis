# Breast Cancer Dataset Analysis

AI-powered tool for accurate breast cancer analysis. 

Streamlit App can be found in https://breastcancerdiagnosis.streamlit.app

This repository provides an analysis and implementation of machine learning models using the Breast Cancer Wisconsin Dataset, a built-in dataset in scikit-learn. The dataset is widely used for exploring classification techniques in predicting whether a tumor is malignant or benign.

# Dataset Overview
The Breast Cancer Wisconsin Dataset consists of features computed from digitized images of fine needle aspirates (FNAs) of breast masses. The goal is to classify tumors as either malignant or benign based on these features.



| **Feature**                | **Description**                                                                     | **Type**    |
|-----------------------------|-------------------------------------------------------------------------------------|-------------|
| `mean_radius`              | Mean of distances from the center to points on the perimeter of the tumor           | Numeric     |
| `mean_texture`             | Standard deviation of gray-scale values                                             | Numeric     |
| `mean_perimeter`           | Mean size of the tumor's perimeter                                                  | Numeric     |
| `mean_area`                | Mean size of the tumor's area                                                       | Numeric     |
| `mean_smoothness`          | Mean of local variation in radius lengths                                           | Numeric     |
| `mean_compactness`         | Mean of `(perimeter^2 / area - 1.0)`                                                | Numeric     |
| `mean_concavity`           | Mean of severity of concave portions of the tumor contour                           | Numeric     |
| `mean_concave_points`      | Mean of the number of concave portions of the tumor contour                         | Numeric     |
| `mean_symmetry`            | Mean symmetry of the tumor                                                          | Numeric     |
| `mean_fractal_dimension`   | Mean of "coastline approximation" - 1D to 2D ratio of the tumor contour             | Numeric     |
| `radius_error`             | Standard error of the tumor's radius                                                | Numeric     |
| `texture_error`            | Standard error of the tumor's texture                                               | Numeric     |
| `perimeter_error`          | Standard error of the tumor's perimeter                                             | Numeric     |
| `area_error`               | Standard error of the tumor's area                                                  | Numeric     |
| `smoothness_error`         | Standard error of the tumor's smoothness                                            | Numeric     |
| `compactness_error`        | Standard error of the tumor's compactness                                           | Numeric     |
| `concavity_error`          | Standard error of the tumor's concavity                                             | Numeric     |
| `concave_points_error`     | Standard error of the number of concave portions                                    | Numeric     |
| `symmetry_error`           | Standard error of the tumor's symmetry                                              | Numeric     |
| `fractal_dimension_error`  | Standard error of the tumor's fractal dimension                                     | Numeric     |
| `worst_radius`             | "Worst" or largest mean value for the tumor's radius                                | Numeric     |
| `worst_texture`            | "Worst" or largest mean value for the tumor's texture                               | Numeric     |
| `worst_perimeter`          | "Worst" or largest mean value for the tumor's perimeter                             | Numeric     |
| `worst_area`               | "Worst" or largest mean value for the tumor's area                                  | Numeric     |
| `worst_smoothness`         | "Worst" or largest mean value for the tumor's smoothness                            | Numeric     |
| `worst_compactness`        | "Worst" or largest mean value for the tumor's compactness                           | Numeric     |
| `worst_concavity`          | "Worst" or largest mean value for the tumor's concavity                             | Numeric     |
| `worst_concave_points`     | "Worst" or largest mean value for the number of concave portions                    | Numeric     |
| `worst_symmetry`           | "Worst" or largest mean value for the tumor's symmetry                              | Numeric     |
| `worst_fractal_dimension`  | "Worst" or largest mean value for the tumor's fractal dimension                     | Numeric     |
| `target`                   | Classification target: `0` = Malignant, `1` = Benign  

## ANN Model Used

MLPClassifier with 
Best Parameters: {'activation': 'tanh', 'hidden_layer_sizes': (50,), 'learning_rate': 'constant', 'solver': 'sgd'}




## References
1. [Scikit-learn Breast Cancer Dataset Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
