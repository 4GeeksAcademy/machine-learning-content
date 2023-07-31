## Summary of Supervised Learning models

The following is a brief review of the different models studied and when and why they are used, as a practical guide to always know how to choose the best option:

### Classifiers and returners

Depending on the nature of the model and its mathematical definition, they could be used for classification, prediction (regression) or both:

| **Model** | **Classification** | **Regression** |
|:----------|:-------------------|:---------------|
| Logistic Regression | ✅ | ❌ |
| Linear Regression | ❌ | ✅ |
| Regularized Linear Regression | ❌ | ✅ |
| Decision Tree | ✅ | ✅ |
| Random Forest | ✅ | ✅ |
| Boosting | ✅ | ✅ |
| Naive Bayes | ✅ | ❌ |
| K Nearest Neighbors | ✅ | ✅ |

In addition, we can easily implement it using the following functions:

| **Model** | **Classification** | **Regression** |
|:----------|:-------------------|:---------------|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | - |
| Linear Regression | - | `sklearn.linear_model.LinearRegression` |
| Regularized Linear Regression | - | `sklearn.linear_model.Lasso`<br />`sklearn.linear_model.Ridge` |
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` | `sklearn.tree.DecisionTreeRegressor` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | `sklearn.ensemble.RandomForestRegressor` |
| Boosting | `sklearn.ensemble.GradientBoostingClassifier`<br />`xgboost.XGBClassifier` | `sklearn.ensemble.GradientBoostingRegressor`<br />`xgboost.XGBRegressor` |
| Naive Bayes | `sklearn.naive_bayes.BernoulliNB`<br />`sklearn.naive_bayes.MultinomialNB`<br />`sklearn.naive_bayes.GaussianNB` | - |
| K Nearest Neighbors | `sklearn.neighbors.KNeighborsClassifier` | `sklearn.neighbors.KNeighborsRegressor` |

### Description and when to use

Knowing what the role of each model is and when we can/should use it is vital to perform our work efficiently and professionally. Below is a comparison chart that addresses this information:

**Model** **Utility** **Recommended Use** **Examples of Use Cases** **Use Cases** **Specifics of Use Cases** **Specifics of Use Cases** **Specifics of Use Cases** **Specifics of Use Cases

| **Model** | **Utility** | **Recommended use** | **Examples of use cases** |
|:----------|:------------|:--------------------|:--------------------------|
| Logistic Regression | Used to classify binary or multiclass (less common) events. | Useful when the relationship between the characteristics and the target variable is linear. Requires that the features be linearly independent. | Classification of emails as spam or non-spam. Disease detection based on symptoms and medical tests. Predicting a customer's probability of buying a product. |
| Linear Regression | Used to predict continuous numerical values. | Useful when the relationship between the characteristics and the target variable is linear. Requires that the characteristics have a significant correlation with the target variable to obtain good results. | Predicting the price of a house based on its size, number of rooms and location. Estimation of a student's academic performance based on his or her hours of study and previous grades. |
| Regularized Linear Regression | Similar to linear regression but including a parameter to avoid overfitting. | Useful when there is multicollinearity between characteristics or to avoid over-fitting of the traditional model. | Prediction of the price of a car based on characteristics such as year of manufacture, make, model, applying regularization to avoid overfitting. Estimation of an employee's salary based on work experience and education level, with regularization to reduce the influence of irrelevant characteristics. |
| Decision Tree | Used to classify or predict continuous numerical values. | Useful when the relationships between the characteristics and the target variable are nonlinear or complex. Can handle numerical and categorical features without the need for standardization. | Prediction of customer loyalty based on purchase history. Classification of movies according to their genre and characteristics. Fraud detection in financial transactions. |
| Random Forest | Used to classify or predict continuous numerical values. Combines multiple decision trees. | Useful when the dataset is large and complex, avoiding over-fitting and improving accuracy. | Image classification for target recognition. Prediction of housing prices based on multiple features. Diagnosis of diseases based on multiple medical tests. |
| Boosting | Used to classify or predict continuous numerical values. Combines multiple decision trees created sequentially to correct for errors in previous models. | Useful when more accurate models than individual models are desired and sufficient computational power is available. | Sentiment analysis in text to classify opinions as positive or negative. Detection of anomalous behavior in security systems. Predicting customer revenue based on multiple factors. |
| Naive Bayes | Used to classify binary or multiclass events. | Useful when there is conditional independence between features (since it is the foundation of the model). Works well when the data set contains categorical features or represents word frequencies. | Text classification problems and categorization tasks. Classification of product reviews as positive or negative. |
| K nearest neighbors | Used to classify or predict continuous numerical values. | Useful when you have a dataset with non-linear relationships and when the local structure of the data is important. The data set must be standardized. | Recommendation of similar products on an e-commerce site. Classification of diseases based on symptoms and medical history. Predicting the price of a house based on similar prices of nearby properties. |

Apart from the example cases and their definitions, the developer's and professional's criteria prevails, and depending on the use case, the data and its characteristics, sometimes even models that are not optimized for that purpose can be useful.