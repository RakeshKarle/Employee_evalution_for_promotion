# Employee_evalution_for_promotion

Developed a machine learning model to predict employee promotions with up to 94.22% accuracy. Explored data, applied log transformation, and built Decision Tree, Logistic Regression, and Random Forest models. Utilized automated ML (EvalML) achieving 93.86% accuracy. Demonstrates strong skills in data preprocessing, EDA, and model building.

# Detailed description of the project:

1. Data Preprocessing: started by importing necessary libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn. Then, loaded the dataset using Pandas and checked its dimensions, which showed that it has 54,808 rows and 13 columns.

2. Checked for missing values in the dataset and dropped rows with missing values.Handled missing values by removing the corresponding rows.

3. Separated the numerical and categorical features from the dataset.

4. Exploratory Data Analysis (EDA): To gain insights into the data and understand its distribution, performed exploratory data analysis using Seaborn and Matplotlib.Created boxplots to visualize the distribution of numerical features and pie charts to visualize the distribution of categorical features.

5. Also applied log transformation to some numerical features (age, length_of_service, and avg_training_score) to make their distributions closer to normal.

6. Feature Scaling : Before building the machine learning models applied standard scaling to the dataset using Scikit-learn's StandardScaler. Feature scaling ensures that all features have the same scale, which can improve the performance of some machine learning algorithms.

7. Model Building: splited the dataset into training and testing sets using Scikit-learn's train_test_split method. The training set contains 37,104 samples, while the testing set contains 9,276 samples.

8. Implemented three classification algorithms to build the promotion prediction model:

9. Decision Tree Classifier: Achieved an accuracy of 89.04%, precision of 66.86%, recall of 69.82%, F1 score of 68.16%, and AUC of 69.80%.

10. Logistic Regression: Achieved an accuracy of 94.22%, precision of 93.08%, recall of 68.09%, F1 score of 74.58%, and AUC of 79.89%.

11. Random Forest Classifier: Achieved an accuracy of 93.85%, precision of 93.86%, recall of 68.09%, F1 score of 74.58%, and AUC of 79.89%.

12. Automated Machine Learning (AutoML):
Explored using EvalML, an open-source library for automated machine learning, to automate the process of model selection and hyperparameter tuning.Created an AutoMLSearch object, specifying the training data, target variable, and the problem type ('binary' for binary classification).

13. AutoML automatically searched for the best pipeline, which combines various preprocessing steps and machine learning algorithms to achieve the highest performance on the given problem. The best pipeline achieved an accuracy of 93.86% on the test set.

14. Saved the best pipeline as a file ('model.pkl') for future use.

Successfully tackled the problem of employee promotion prediction.Explored and visualized the dataset, preprocessed the data, built multiple machine learning models using various algorithms, and finally, explored automated machine learning using EvalML.
