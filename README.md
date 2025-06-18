# Movie Success Predictor
This machine learning project utilizes the TMDB 5000 Movie Dataset to build classification models to predict whether a movie will be a success or failure and regression models to predict a movie's revenue. The purpose of this project is to see how different machine learning models perform on the same tasks.

## Dataset
This project uses the TMDB 5000 Movie Dataset from Kaggle, which includes data such as genres, cast, production countries, budget, and revenue.

## Preprocessing Overview
### The data was preprocessed in the following ways:

Merged movie and credits datasets

Extracted the release month and year from the release date

Removed entries with zero revenue

One-hot encoded: <br>
Top 20 genres<br>
Top 20 production countries<br>
Top 20 cast<br>

Defines a custom classification target:<br>
A movie is a Success if its revenue > 1.5 * its budget, else the movie is a Failure<br>

## Models & Evaluation
### Classification (Success vs Fail)

Logistic Regression <br>
Random Forest Classifier <br>
Support Vector Machine <br>
K-Nearest Neighbors <br>

Evaluated using 10-fold Stratified Cross-Validation and accuracy.
<br><br>

### Regression (Revenue Prediction)

K-Nearest Neighbors Regressor <br>
Linear Regression <br>

Evaluated using 10-fold Cross-Validation and R² score.

