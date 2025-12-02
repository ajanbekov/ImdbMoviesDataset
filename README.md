# IMDB Movies Dataset Analysis

## Overview

This project utilizes the IMDB Movies Dataset available on Kaggle, which can be accessed [here](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows). The goal is to analyze the dataset and explore potential relationships between various factors.

The Jupyter Notebook included in this repository contains the code for analysis. It encompasses the following key aspects:

* **Preprocessing:** Handling missing values and correcting data types to ensure analysis readiness.
* **Visualization:** Visualizing the dataset to gain insights into the distribution of variables.
* **Correlation Analysis:** Conducting correlation analysis to understand the relationships between variables.
* **Trend Analysis:** Performing trend analysis over time to examine how variables change.
* **Classification Models:** Building and evaluating models such as Random Forest, Extra Trees, Decision Tree, LGBM, and Gradient Boosting for tasks like movie genre prediction and popularity classification.
* **Regression Models:** Developing regression models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) to predict movie popularity levels and gross revenue.
* **Model Evaluation:** Utilizing metrics such as classification_report, R² and RMSE values, feature importance, confusion matrix, and Actual vs. Predicted plots to assess performance.

## Dataset Information

The IMDB Movies Dataset contains comprehensive information about the top 1000 movies, consisting of 16 columns such as Poster_Link, Series_Title, Released_Year, Certificate, Runtime, Genre, IMDB_Rating, Overview, Meta_score, Director, Star1, Star2, Star3, Star4, No_of_Votes, and Gross.

## Usage

To replicate the analysis and explore the code, follow these steps:

1. Clone this repository to your local machine.
2. Download the dataset from the Kaggle link provided above.
3. Install the required libraries (e.g., pandas, numpy, scikit-learn, matplotlib, seaborn).
4. Open the Jupyter Notebook and execute the cells sequentially.
