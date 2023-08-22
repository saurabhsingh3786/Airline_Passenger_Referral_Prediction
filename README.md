# Airline_Passenger_Referral_Prediction
Predicting aircraft passenger referal and excavating the main influencing factors can help airlines improve their services and gain.


![WhatsApp Image 2023-08-21 at 14 25 05](https://github.com/ShriyaChouhan/Airline_Passenger_Referral_Prediction/assets/128309746/ed85e541-994c-4414-9fed-e1d216ac6e65)


Customer referral is a crucial aspect of business growth and success, and the airline industry is no exception. Satisfied passengers who have had positive experiences with an airline are more likely to refer the airline to their friends, family, and colleagues. Identifying these potential advocates can help airlines improve customer satisfaction and loyalty and attract new customers.

In this project, we will use machine learning algorithms to predict whether a passenger will refer an airline to others. We will use a dataset that includes past passengers and their referral behavior, as well as various features such as age, gender, flight class, and route information.

Our first step will be to perform preprocess the data by handling missing values, encoding categorical variables, and scaling numeric features and after that exploratory data analysis to gain insights into the data and identify any patterns or correlations .

We will then apply several machine learning algorithms, including logistic regression, decision tree, and support vector machines, to predict the likelihood of a passenger becoming a referral. We will also perform feature engineering and selection to improve the performance of our models.

Finally, we will evaluate our models using metrics such as accuracy, precision, recall, and F1 score. We will also use techniques such as cross-validation and grid search to tune our hyperparameters and ensure our models generalize well to new data.

***What's in the Project***
1. EDA
2. Data Cleaning
3. Hypothesis Testing
4. Feature Engineering
5. Data Preprocessing
6. Model Builiding
7. Model Evaluation
8. Feature Work
9. Conclusion

***Objective:*** The main objective is to predict whether passengers will refer the airline to their friends.

**Description Of Features:**
* airline: Name of the airline.
* overall: Overall point given to the trip between 1 to 10.
* author: Author of the trip
* reviewdate: Date of the Review
* customer review: Review of the customers in free text format
* aircraft: Type of the aircraft
* travellertype: Type of traveler (e.g. business, leisure)
* cabin: Cabin at the flight
* date flown: Flight date
* seatcomfort: Rated between 1-5
* cabin service: Rated between 1-5
* foodbev: Rated between 1-5
* entertainment: Rated between 1-5
* groundservice: Rated between 1-5
* valueformoney: Rated between 1-5
* recommended: Binary, target variable

***Data Preparation:***
* Dropping rows having entire row as NAN
* Dropping columns which don't add value for the analysis
* Dropping duplicates

***EDA***
The primary goal of EDA is to support the analysis of data prior to making any conclusions. It may aid in the detection of apparent errors, as well as a deeper understanding of data patterns, the detection of outliers or anomalous events, and the discovery of interesting relationships between variables.

***Feature Engineering & Data Pre-processing***
1. **Handling Missing Values & Missing Value Imputation**
    **Median Imputation:---**
       Median imputation involves replacing missing values with the median of the non-missing values in the same column. The median is the middle value of a dataset when it is sorted in ascending order. It is robust to outliers and can be a better measure of central tendency when data is skewed.
3. **Handling Outliers**
     As there is no outliers in our dataset, so didnt apply any technique to remove them.
4. **Categorical Encoding**
    Here 'customer_review' contains textual data so we will handle it in textual data processing and 'author', 'arrival_city', 'departure_city' features are not so of important use in ml model implementation so we would drop them further when doing feature manipulation.
   We use label encoding for categorical columns because most machine learning algorithms can only operate on numerical data. Label encoding converts categorical data into numerical data by assigning a unique integer to each category. This allows the machine learning algorithm to understand the relationship between the different categories.
Here are some of the advantages of using label encoding:
* It is easy to implement and understand.
* It is relatively efficient.
* It can be used with a variety of machine learning algorithms.
5.  ***Textual Data Preprocessing***
    In our dataset, there is a feature 'customer_review' which contains textual data so we will convert that text review into numeric review so that we can use it in our feature selection as it would be very helpful to know which review is providing recommendation. For this whole process we use NLP(NATURAL LANGUAGE PROCESSING) for reviews.

6. ***Feature Manipulation & Selection***
  In the code for feature selection in the merged dataset, we used the SelectKBest method with the ANOVA (Analysis of Variance) score function. Let's discuss the feature selection methods used and the reasons for choosing them:

  SelectKBest with ANOVA: SelectKBest is a feature selection method from scikit-learn that selects the top 'k' features based on univariate statistical tests. The ANOVA score function is used specifically for regression tasks (predicting continuous target variables) and evaluates the relationship between each feature and the target variable using ANOVA F-values.

 Reason for using SelectKBest with ANOVA: We chose this method because the target variable 'Sales' is a continuous numerical variable in the regression task. The ANOVA F-values help us assess the statistical significance of each feature's relationship with the target. By selecting the top 'k' features, we aim to keep the most informative features and reduce the model's complexity, which can help prevent overfitting.

7. ***Data Splitting***
 The test_size parameter in the train_test_split function controls the proportion of the data that should be allocated to the testing set when splitting the dataset into training and testing sets. In the above code, test_size=0.2 is used, which means that 20% of the data will be allocated to the testing set, and the remaining 80% will be used for training.The commonly used splitting ratios are 80:20 (test_size=0.2) and 70:30 (test_size=0.3). These ratios strike a good balance between having enough data for training and obtaining a reliable evaluation on the testing set.

8. ***ML Model Implementation***
    * Logistic Regression
    * Decision Tree
    * K-Nearest Neighbour
    * Naïve Bayes Classifier
    * Support Vector Machine
   
     The evaluation metrics that are generally considered for a positive business impact are precision, recall, and ROC AUC score. Let's briefly discuss each metric and their business impact:

    1. **Precision:--** Precision measures the accuracy of the positive predictions made by the model. It's especially important when the cost of false positives (misclassifying a negative instance as positive) is high. In scenarios where you want to avoid false positives, precision is a crucial metric. For example, in medical diagnoses or fraud detection, you want to minimize false positives to avoid unnecessary treatments or alerts.

    2. **Recall:--** Recall (also known as sensitivity or true positive rate) measures the ability of the model to correctly identify positive instances. It's important when the cost of false negatives (misclassifying a positive instance as negative) is high. High recall is desired in scenarios where missing a positive instance can have severe consequences. For example, in disease detection, you want to catch as many cases as possible.

    3. **ROC AUC Score:--** Receiver Operating Characteristic Area Under the Curve (ROC AUC) score is a measure of the model's ability to distinguish between positive and negative classes. It's useful for evaluating the overall performance of the model across different thresholds. A high ROC AUC score indicates that the model is good at ranking positive instances higher than negative ones.

In a positive business impact scenario, you would want to strike a balance between precision and recall. It depends on the specific context and the relative costs of false positives and false negatives. If both precision and recall are high, it suggests that the model is effectively identifying positive instances while minimizing false positives and false negatives.


***Conclusion***
In this project, we embarked on a comprehensive journey to predict airline passenger referrals using a variety of machine learning models. We began by conducting thorough exploratory data analysis (EDA), delving into the dataset to gain valuable insights into the various features and their relationships with the target variable.

**Our EDA unveiled several significant findings:--**

**Traveller Types and Cabin Class:--** Passengers who travel for Business purposes tend to rate cabin service higher on average compared to other traveller types. Moreover, passengers in Business Class exhibit higher ratings for cabin service than those in Economy Class. Seat Comfort and Travel Purpose: Passengers traveling for Solo Leisure purposes tend to rate seat comfort higher on average compared to those traveling for Business reasons. Food and Beverage Ratings: We observed that the frequencies of food and beverage ratings of 2, 4, and 5 are equally distributed, indicating a balanced distribution. Moving forward, we harnessed this information to construct and evaluate multiple machine learning models:

**Logistic Regression:--** Our initial Logistic Regression model yielded an accuracy of 96%, demonstrating promising results out of the gate. Decision Tree: While Decision Tree exhibited commendable performance with an accuracy of around 94%, the tuned model showcased significant improvement with an accuracy of 95.98%. K-Nearest Neighbours: This model showcased a robust accuracy of 95.97%, slightly decreased to 95.92% with hyperparameter tuning. Naïve Bayes Classifier: The Naïve Bayes Classifier achieved an accuracy of 96.09%, with slight performance enhancement to 96.40% following tuning. Support Vector Machine: Our Support Vector Machine model, with an accuracy of 96.57%, proved to be a strong contender for prediction. In terms of feature importance, it was evident that certain attributes had more influence on the models' predictions. These varied across models, but common factors such as seat comfort, cabin service, and overall ratings consistently stood out.

After careful consideration and meticulous evaluation, we chose the Support Vector Machine (SVM) model with hyperparameter tuning as our final prediction model. With an accuracy of 96.57%, this model showcased remarkable performance. Its ability to handle complex relationships in the data and accommodate non-linear decision boundaries made it an optimal choice for our task.

Furthermore, we delved into the SVM model's feature importance using coefficient analysis. This insight provided us with a better understanding of how different attributes contribute to the model's predictions. However, it's important to note that SVM models may not offer the most intuitive feature importance interpretation due to the nature of their decision boundaries.

In conclusion, our journey through data exploration, model building, and evaluation has enabled us to construct a reliable prediction model for airline passenger referrals. The insights gained from this project can provide valuable guidance to the airline industry in understanding passenger preferences and optimizing their services to enhance customer satisfaction and referrals.
