# Resampling Imbalanced Lending Data

## Overview of the Analysis

### Purpose of the Analysis
The purpose of the analysis is to determine a machine learning model that accurately classifies data that is inherently imbalanced. Credit risk is inherently imbalanced because healthy loans easily outnumber risky loans. The model needs to accurately identify the creditworthiness of borrowers. Imbalanced classes can cause models to predict the larger class (healthy loans) much better than the smaller class (high-risk loans) because it has more healthy loan data to train on. Therefore, this analysis will evaluate and train models with imbalanced classes with a goal of increasing the accuracy of identification of the smaller class. 

### Financial Information and What Nees to be Predicted
The financial information that is being used for this analysis is a dataset of historical lending activity from a peer-to-peer lending services company. The data includes the following information columns: loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks, and total debt. There is also a loan status column that lists a '0' for healthy loans and a '1' for high-risk loans. There are 77,536 total rows of data. 

We need to train our model using this historical lending data so that it can learn which factors cause the loan status to be a '0' (healthy loan) or a '1' (high-risk loan). Then, we need to use the trained model to predict whether a new loan (one it hasn't trained on) will be a healthy or a high-risk loan. It is much faster for a machine to classify new loans in this way compared to a human, so we want to train the machine to be able to make an accurate prediction about the new loans.   

### Basic Information About the Variables We are Trying to Predict 
To determine how many loans in the data were healthy vs. high-risk loans, I use the ```value_counts()``` function on the loan status (or y) column. The number of healthy loans is 75,036 while the number of high-risk loans is 2,500. As you can see, this is an imbalanced class with many more healthy loans than high-risk loans. Since the loan status is the "y" column (the variable we are trying to predict), the rest of the columns are considered "features" and are the "X" column. The "X" column has the features or the information that is needed for making the prediction (i.e., debt to income, total debt, etc. as mentioned above). 

### Stages of the Machine Learning Process
The five main stages of the machine learning process are as follows:
1. Decide on a model and create a model instance: The model used in this analysis was Logistic Regression.
```model = LogisticRegression(random_state=1)```

2. Split the dataset into training and testing sets, and preprocess the data.

3. Fit/train the training data to the model.

4. Use/test the model for predictions.

5. Evaluate the model performance.


### Methods Used
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

---

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

---

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
