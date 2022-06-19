# Feature encoding for categorical variables

This is an important phase in the data cleaning process. Feature encoding involves replacing classes in a categorical variable with real numbers. For example, classroom A, classroom B and classroom C could be encoded as 2,4,6.

**When should we encode our features? Why?**

We encode features when they are categorical. We convert categorical data numerically because math is generally done using numbers. A big part of natural language processing is converting text to numbers. Just like that, our algorithms cannot run and process data if that data is not numerical. We should always encode categorical features so that they can be processed by numerical algorithms, and so machine learning algorithms can learn from them. 

## Types of categorical variables

There are two types of categorical variables, nominal and ordinal. Before we dive into feature encoding, it is important that we first contrast the difference between a nominal variable and an ordinal variable.

As we already explained in the Feature Engineering lecture, a nominal variable is a categorical variable where its data does not follow a logical ordering. Some examples are:

-Gender (Male or Female)

-Colours (Red, Blue, Green)

-Political party (Democratic or Republican)

An ordinal variable, on the other hand, is also a categorical variable except its data follows a logical ordering. Some examples of ordinal data include:

-Socioeconomic status (low income, middle income or high income)

-Education level (high school, bachelor’s degree, master’s degree or PhD)

-Satisfaction rating (extremely dislike, dislike, neutral, like or extremely like)

## Methods to encode categorical data:

#### 1. Nominal Encoding

Each category is assigned a numeric value not representing any order. For example: [black, red, white] could be encoded to [3,7,11].
We will explore two different ways to encode nominal variables, one using Scikit-learn OneHotEncoder and the other using Pandas get_dummies.

**Scikit-learn OneHotEncoder**


Each category is transformed into a new binary feature, with all records being marked 1 for True or 0 for False. For example: [Florida, Virginia, Massachussets] could be encoded to state_Florida = [1,0,0], state_Virginia = [0,1,0], state_Massachussets = [0,0,1].

In the case of a feature named 'Gender' with the unique values of Male and Female, OneHotEncoder creates two columns to represent the two categories in the gender column, one for male and one for female.

Female passengers will receive a value of 1 in the female column and a value of 0 in the male column. Conversely, male passengers will receive a value of 0 in the female column and a value of 1 in the male column.

```py
from sklearn.preprocessing import OneHotEncoder

#Instantiate OneHotEncoder
ohe = OneHotEncoder()

#Apply OneHotEncoder to the gender column
ohe.fit_transform(data[['Gender']])

#Verify categories created in OneHotEncoder
ohe.categories_
```

**Pandas get_dummies method**

Similarly, get_dummies encodes the categorical 'Sex' feature by creating two columns to represent the two categories.

```py
pd.get_dummies(data['Gender']).head()
```

**Difference between OneHotEncoder() and Get_dummies**

-Under OneHotEncoder the original dataframe remains the same size.

-OneHotEncoder can be incorporated as part of a machine learning pipeline.

-Under OneHotEncoder we can use the GridSearch function in Scikit-learn to choose the best preprecessing parameters.

#### 2. Ordinal Encoding

Each category is assigned a numeric value representing an order. For example: [small, medium, large, extra-large] could be encoded to [1,2,3,4].

In this section, we will again consider two approaches to encoding ordinal variables, one using Scikit-learn OrdinalEncoder and the other using Pandas map method.

**Scikit-learn OrdinalEncoder**

OrdinalEncoder assigns incremental values to the categories of an ordinal variable. This helps machine learning algorithms to pick up on an ordinal variable and subsequently use the information that it has learned to make more accurate predictions.

In order to use OrdinalEncoder, we have to first specify the order in which we would like to encode our ordinal variable.

Code example:

```py

from sklearn.preprocessing import OrdinalEncoder

#Instantiate ordinal encoder

oe = OrdinalEncoder()

#Apply ordinalEncoder to income_status
oe.fit_transform(data[['income_status']])

```

**Pandas Map method**

The Pandas map method is a more manual approach to encoding ordinal variables where we individually assign numerical values to the categories in an ordinal variable.

Although it replicates the result of the OrdinalEncoder, it is not ideal for encoding ordinal variables with a high number of unique categories.

```py
data['income_status'].map({'low': 0,
                            'medium':1,
                            'high':2})
```
## Building a pipeline

Imagine we want to combine OneHotEncoder and OrdinalEncoder into a single-step column transformer. After we have separated our predictors from our target variable we can build this column transformer like this:

```py
#make a column encoder

column_encoder = make_column_encoded(
    (ohe, ['Gender','Blood_type', 'Colour']),
    (oe, ['income_status'])
)

#apply encoder to predictors

column_encoder.fit_transform(predictors)
```

After separating our data into train and validation sets, we can use Scikit-learn make_pipeline to build a machine learning pipeline with the preprocessing steps and the chosen algorithm for modeling.

Code example:

```py

from sklearn.pipeline import make_pipeline

# Instantiate pipeline with linear regression

lm = LinearRegression()
lm_pipeline = make_pipeline(column_encoder, lm)

#Fit pipeline to training set and make predictions on test set

lm_pipeline.fit(X_train, y_train)
lm_predictions = lm_pipeline.predict(X_test)

print('First 5 predictions:', list(lm_predictions[:5]))
```

For more information on make_pipeline from Scikit-learn, see the following documentation: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html

Source: 

https://towardsdatascience.com/guide-to-encoding-categorical-features-using-scikit-learn-for-machine-learning-5048997a5c79#:~:text=Feature%20encoding%20is%20the%20process,not%20data%20in%20text%20form.

https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63
