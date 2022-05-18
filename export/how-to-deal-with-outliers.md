
# How to deal with outliers

As we saw in the exploratory data analysis, in statistics, an outlier is a data point that differs significantly from other observations. An outlier may be due to variability in the measurement or it may indicate experimental error. 

Outliers are problematic for many statistical analyses because they can mislead the training process and give poor results. Unfortunately, there are no strict statistical rules for definitively identifying outliers. Finding outliers depends on subject-area knowledge and an understanding of the data collection process. There are many reasons why we can find outliers in our data.



**How to identify potential outliers:**

-Graphing your data to identify outliers.

Boxplots, histograms and scatterplots can highlight outliers
Boxplots use the interquartile method with fences to find outliers. We can also use it to find outliers in different groups of a column.
Histograms look for isolated bars
Scatterplots to detect outliers in a multivariate setting. A scatterplot with regression line shows how most of the points follow the fitted line for the model, except for some outliers.

-Using Z-scores to detect outliers

Z-scores can quantify outliers when our data has a normal distribution. Z-scores are the number of standard deviations above and below the mean that each value falls. For example, a Z-score of 2 indicates that an observation is two standard deviations above the average while a Z-score of -2 signifies it is two standard deviations below the mean. A Z-score of zero represents a value that equals the mean.
Z-scores can be misleading with small datasets. Sample sizes of 10 or fewer observations cannot have Z-scores that exceed a cutoff value of +/-3.

-Using the IQR to create boundaries

The IQR is the middle 50% of the dataset. It’s the range of values between the third quartile and the first quartile (Q3 – Q1). 

There are many ways to identify outliers. We must use our in-depth knowledge about all the variables when analyzing data. Part of this knowledge is knowing what values are typical, unusual, and impossible. With that understanding of the data, most of the times it is better to use visual methods.

Not all outliers are bad and some should not be deleted. In fact, outliers can be very informative about the subject-area and data collection process. It’s important to understand how outliers occur and whether they might happen again as a normal part of the process or study area.

**How to handle outliers:**

Should we remove outliers?

It depends on what  causes the outliers.

Causes for outliers:

-Data entry and measurement errors.

Let's imagine we have an array of ages: {30,56,21,50,35,83,62,22,45,233}. 233 is clearly an outlier because it is an impossible age as nobody lives 233 years. Examining the numbers more closely, we may conclude that the person in charge of data entry may have accidentally entered a double 3 so the real number would have been 23. If you determine that an outlier value is an error, correct the value when possible. That can involve fixing the typo or possibly remeasuring the item or person. If that’s not possible, you must delete the data point because you know it’s an incorrect value.

-Sampling problems

Inferential statistics use samples to draw conclusions about a specific population. Studies should carefully define a population, and then draw a random sample from it specifically. That’s the process by which a study can learn about a population.

Unfortunately, your study might accidentally obtain an item or person that is not from the target population. If you can establish that an item or person does not represent your target population, you can remove that data point. However, you must be able to attribute a specific cause or reason for why that sample item does not fit your target population.

-Natural variation

The process or population you’re studying might produce weird values naturally. There’s nothing wrong with these data points. They’re unusual, but they are a normal part of the data distribution. If our sample size is large enough, we can find unusual values. In a normal distribution, approximately 1 in 340 observations will be at least three standard deviations away from the mean. If the extreme value is a legitimate observation that is a natural part of the population we’re studying, you should leave it in the dataset.


How to handle them?

Sometimes it’s best to keep outliers in your data. They can capture valuable information that is part of your study area. Retaining these points can be hard, particularly when it reduces statistical significance! However, excluding extreme values solely due to their extremeness can distort the results by removing information about the variability inherent in the study area. You’re forcing the subject area to appear less variable than it is in reality.

If the outlier in question is:

A measurement error or data entry error, correct the error if possible. If you can’t fix it, remove that observation because you know it’s incorrect.
Not a part of the population you are studying (i.e., unusual properties or conditions), you can legitimately remove the outlier.
A natural part of the population you are studying, you should not remove it.

When you decide to remove outliers, document the excluded data points and explain your reasoning. You must be able to attribute a specific cause for removing outliers. Another approach is to perform the analysis with and without these observations and discuss the differences. Comparing results in this manner is particularly useful when you’re unsure about removing an outlier and when there is substantial disagreement within a group over this question.

What to do when we want to include outliers but we dont want them to mislead our results?

We can use bootstrapping techniques, models taht are robust and not sensitive to outliers.

Source:

https://www.kdnuggets.com/2017/01/3-methods-deal-outliers.html?msclkid=69e30c98cf9111ecb308f37782fbdee6

https://statisticsbyjim.com/basics/outliers/

https://statisticsbyjim.com/basics/remove-outliers/#:~:text=If%20the%20outlier%20in%20question%20is%3A%201%20A,you%20are%20studying%2C%20you%20should%20not%20remove%20it.?msclkid=69e228d8cf9111ec8475cfedf095ca97

https://medium.com/@mxcsyounes/hands-on-with-feature-engineering-techniques-dealing-with-outliers-fcc9f57cb63b
