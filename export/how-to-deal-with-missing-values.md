# How to deal with missing values

Why do we need to fill in missing values?

Because most of the machine learning models that we want to use will provide an error if we pass NaN values into it.  The easiest way is to fill them up with 0 but this can reduce our model accuracy significantly.

Before we start deleting or imputing missing values, we need to understand the data in order to choose the best method to treat missing values.

As we saw in the exploratory data analysis, we can find missing values by using the info() function and looking at how many non-null values are in each column. Another way is using isnull() function.

### Methods to deal with missing values

**1. Deleting the column that has missing values**

It should only be used when there are many null values in the column. We can do this by using dataframe.dropna(axis=1) or by using drop() and specifying the column we want to drop. The problem with this method is that we may lose valuable information on that feature, as we have deleted it completely due to some null values. Since we are working with both training and validation sets, we are careful to drop the same columns in both DataFrames.


**2. Deleting rows that have missing values**

We can delete only the rows that have missing values by using the dropna() function again. This time we will use axis=0 as parameter because instead of looking for columns with missing values, we are looking for rows with missing values. 
We can add a 'threshold' to our function. That way dropping rows will meet the threshold conditions first. For example if we write the following function :

```py
df2=df.dropna (axis = 0, thresh=2) ------>  #It will keep the rows with at least 2 non-NA values.
```

This can be used in percentages too.

```py
df2=df.dropna (axis = 0, thresh= 0.65 * len(df)) ------>  #It will drop all rows with null values except the ones that have at least 65% of data filled (non null values).
```

Threshold can be used in rows, as well as columns using axis= 1.


**3. Filling missing values - Imputation**

In this method we fill null values with a certain value. Although it's simple, filling in the mean value generally performs quite well. While statisticians have experimented with more complex ways to determine imputed values (such as regression imputation, for instance), the complex strategies typically give no additional benefit once you plug the results into sophisticated machine learning models.
There are some ways to impute missing values:

- Mean imputation

If it is a numerical feature, we can replace the missing values with the mean, but make sure that the feature does not have extreme values so that the mean is still a central tendency.

- Median

If it is a numerical feature and the feature has extreme values that we have decided to keep, then the median would be a better central tendency because the outliers will pull the mean towards the outliers and it will not be a central tendency anymore. In that case, the median will replace missing values in a better way.

- Mode

The mode is considered the most frequent value, so if our feature is a categorical value, filling the missing data with the mode will be a good approach.

- New Value



- knn imputation



- regression imputation

In this case, the null values in one column are filled by fitting a regression model using other columns in the dataset.

You can use the fillna() function to fill the null values in the dataset.

Source: 

https://www.analyticsvidhya.com/blog/2021/05/dealing-with-missing-values-in-python-a-complete-guide/
