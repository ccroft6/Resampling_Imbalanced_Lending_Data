# Credit Risk Classification 

## Overview of the Analysis

### Purpose of the Analysis
The purpose of the analysis is to determine a machine learning model that accurately classifies data that is inherently imbalanced. Credit risk is inherently imbalanced because healthy loans easily outnumber risky loans. The model needs to accurately identify the creditworthiness of borrowers. Imbalanced classes can cause models to predict the larger class (healthy loans) much better than the smaller class (high-risk loans) because it has more healthy loan data to train on. Therefore, this analysis will evaluate and train models with imbalanced classes with a goal of increasing the accuracy of identification of the smaller class. 

### Financial Information and What Nees to be Predicted
The financial information that is being used for this analysis is a dataset of historical lending activity from a peer-to-peer lending services company. The data includes the following information columns: loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks, and total debt. There is also a loan status column that lists a '0' for healthy loans and a '1' for high-risk loans. There are 77,536 total rows of data. 

We need to train our model using this historical lending data so that it can learn which factors cause the loan status to be a '0' (healthy loan) or a '1' (high-risk loan). Then, we need to use the trained model to predict whether a new loan (one it hasn't trained on) will be a healthy or a high-risk loan. It is much faster for a machine to classify new loans in this way compared to a human, so we want to train the machine to be able to make an accurate prediction about the new loans.   

### Basic Information About the Variables We are Trying to Predict 
To determine how many loans in the data were healthy vs. high-risk loans, I use the ```value_counts()``` function on the loan status (or y) column. The number of healthy loans is 75,036 while the number of high-risk loans is 2,500. As you can see, this is an imbalanced class with many more healthy loans than high-risk loans. Since the loan status is the "y" column (the variable we are trying to predict), the rest of the columns are considered "features" and are the "X" column. The "X" column has the features or the information that is needed for making the prediction (i.e., debt to income, total debt, etc. as mentioned above). 

```# Check the balance of our target values
y.value_counts()

0    75036
1     2500
Name: loan_status, dtype: int64
```

### Stages of the Machine Learning Process
The main stages of the machine learning process are as follows:
1. Decide on a model and create a model instance: The model used in this analysis is Logistic Regression.
Example:
```lr_model = LogisticRegression(random_state=1)```

2. Split the dataset into training and testing sets, and preprocess the data. The default uses 75% of the data for training and 25% of the data for testing. 
Example:
```# Import the train_test_learn module
from sklearn.model_selection import train_test_split

# Split the data using train_test_split
# Assign a random_state of 1 to the function
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

3. Fit/train the training data to the model. In the fit stage, the algorithm learns from the training data. 
Example:
```# Fit the model using training data
lr_model.fit(X_train, y_train)
```

4. Use the model for predictions. In the predict stage, the algorithm uses the trained model to predict new data. If we give the model new data that's similar enough to the data that it's gotten before, it can guess (or predict) the outcome for that data.
Example:
```# Make a prediction using the testing data
original_y_pred = lr_model.predict(X_test)
```

5. Evaluate the model performance. To evaluate the model performance, we can use the `accuracy_score` function, the `confusion_matrix` function, and the `classification_report` function. The **accuracy score** provides the percentage of correct predictions, both true positives and true negatives, made by the model divided by the total number of outcomes. The **confusion matrix** contains the number of times that a model correctly predicted a positive or a negative value and the number of times that it didn't. The **classification report** provides the precision and recall scores. The **precision** is the percentage of true positives in a single category predicted by the model divided by the total number of all positive predictions. The **recall** is the percentage of true positives in a single category that the model correctly categorized divided by the total number of actual transactions in that category (true positives and false negatives). 

### Methods Used
#### Logistic Regression
The model used for this analysis is Logistic Regression. It is a machine learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (i.e., high-risk loan) or 0 (i.e., healthy loan). Logistic regression requires quite large sample sizes and requires the dependent variable to be binary.

#### Resampling Technique 
Because the data is imbalanced, the resampling technique of oversampling was used. Oversampling is when you create more instances of the smaller class label. You resample enough of the original instances to make the size of the smaller class equal to the size of the larger class in the training set. When adding this technique, the machine learning steps are followed:
1. Create a model instance.
Example:
```# Instantiate the random oversampler model
# Assign a random_state parameter of 1 to the model
random_oversampler_model = RandomOverSampler(random_state=1)

# Fit the original training data to the random_oversampler model
X_resampled, y_resampled = random_oversampler_model.fit_resample(X_train, y_train)

# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
resample_lr_model = LogisticRegression(random_state=1)
```
2. Fit the model. 
Example:
```# Fit the model using the resampled training data
resample_lr_model.fit(X_resampled, y_resampled)
```
3. Predict using the model.
Example:
```
# Make a prediction using the testing data
resampled_y_pred = resample_lr_model.predict(X_test)
```

---

## Results

This section compares the balanced accuracy scores and the precision and recall scores of both machine learning models.

* Machine Learning Model 1 - Logistic Regression Model with Original Data:
  * **Accuracy Score:** The accuracy score for Model 1 is 95.2% 
  * **Precision:** The precision is 85%. It correctly predicted the high-risk loans out of all the high-risk loan predictions 85% of the time.
  * **Recall:** The recall is 91%. It correctly predicted 91% of loans as high-risk out of all those that were actually high-risk. 

* Machine Learning Model 2 - Logistic Regression Model with Oversampled Data):
  * **Accuracy Score:** The accuracy score for Model 2 is 99.4% 
  * **Precision:** The precision is 84%. It correctly predicted the high-risk loans out of all the high-risk loan predictions 84% of the time.
  * **Recall:** The recall is 99%. It correctly predicted 99% of loans as high-risk out of all those that were actually high-risk. 

---

## Summary
Based on the evaluation results of the two models, the model with resampled data (model 2) performs the best. Model 2 has an accuracy score of 99.4% while model 1 has an accuracy score of 95.2%. This means the model with resampled data is better at predicting true positives and true negatives. 

The recall for model 2 (99%) is higher than the recall for model 1 (91%). This means the resampled data is better at correctly classifying the high-risk loans out of those that are truly high-risk.  

The precision for model 1 (85%) is slightly higher than the precision for model 2 (84%). This means the original data was a little better at correctly classifying the high-risk loans out of all the high-risk predictions.

In all, model 2 (resampled data) is better at detecting the creditworthiness of borrowers compared to model 1 (imbalanced data). It is more important to predict the 1's (high-risk loans), so we focus our evaluation of performance on those. An overall accuracy of 99.4% is great, so I would recommend the use of the Logistic Regression model with the resampled (oversampled) data.  

---

## Contributors 
Catherine Croft

Email: catherinecroft1014@gmail.com

LinkedIn: [catherine-croft](https://www.linkedin.com/in/catherine-croft-4715481aa/)

---

## License 
MIT