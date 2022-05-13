
# HYPOTHESIS TESTING

### Density Curves

Having frequency histograms is a good way of looking at how is our data distributed. But at some point, we might want to see what percentage of our data falls into each of the variable's categories. In that case we can build a relative frequency histogram, which will shows us the percentages instead of the quantities. But as we have more and more data we might want to see smaller bins in our histogram so that we can see the distribution much clear. As we continue having more and more data, maybe we want to see even more thin bins which will get us into a point where we approach to an infiinite number of categories and the best way of looking at it would be to connect the top of the bars that you will actually get a curve. This is called the density curve.

![density_curve.jpg](../assets/density_curve.jpg)

**Probability from density curve:**

Let's see an example on how to calculate probabilities from density curves:

We have a set of women's heights that are normally distributed with a mean of 155 centimeters and a standard deviation of 15 centimeters. The height of a randomly selected women from this set will be denoted as W.

Find and interpret P(W > 170)

![density_probability_problem.jpg](../assets/density_probability_problem.jpg)

### Central Limit Theorem


When we draw samples of independent random variables (drawn from any single distribution with a finite variance), their sample mean tends toward the population mean and their distribution approaches a normal distribution as sample size increases, regardless of the distribution from which the random variables were drawn. Their variance will approach the population variance divided by the sample size.

The central limit theorem describes the shape of the distribution of sample means as a Gaussian or Normal Distribution.
The theorem states that as the size of the sample increases, the distribution of the mean across multiple samples will approximate a Gaussian distribution.

The central limit theorem does not state anything about a single sample mean, instead it states something about the shape or the distribution of sample means.

For example, let's say we have a fair and balanced 6-sided die. The result of rolling the die has a uniform distribution on [1,2,3,4,5,6]. The average result from a die roll is (1+2+3+4+5+6)/6 = 3.5

If we roll the die 10 times and average the values, then the resulting parameter will have a distribution that begins to look similar to a normal distribution, again centered at 3.5.

If we roll the die 100 times and average the values, then the resulting parameter will have a distribution that behaves even more similar to a normal distribution, again centered at 3.5, but now with decreased variance.

![central_limit_theorem.jpg](../assets/central_limit_theorem.jpg)

### Sampling Methods

Let's imagine we have a population of students. For that population we can calculate parameters, for example the age mean, or the grades standard deviation. These are all population parameters or truths about that population. Sometimes we might not know the population parameter, so the way to estimate that parameter is by taking a sample. 

The sampling method is the process of studying the population by gathering information and analyzing that data. It refers to how members from that population are selected for the study.

**Non-Representative Sampling:**

The non-representative sampling method is a technique in which the researcher selects the sample based on subjective judgment rather than the random selection. Not all the members of the population have a chance to participate in the study.


Convenience Sampling = Picks samples that are most convenient, like people that can be easily approached.

Consecutive Sampling = Picks a single person or a group of people for sampling. Then the researcher researches for a period of time to analyze the result and move to another group if needed.

Purposive Sampling = Picks samples for a specific purpose. An example is to focus on extreme cases. This can be useful but is limited because it doesn't allow you to make statements about the whole population.

Snowball Sampling = In this method, the samples have traits that are difficult to find. So, each identified member of a population is asked to find the other sampling units. Those sampling units also belong to the same targeted population.

**Representative Sampling:**

Simple Random Sampling = Pick samples (psuedo)randomly. Every item in the population has an equal and likely chance of being selected in the sample.

Systematic Sampling = Pick samples with a fixed interval. For example every 10th sample (0, 10, 20, etc.). It is calculated by dividing the total population size by the desired population size.

Stratified Sampling = The total population is divided into smaller groups formed based on a few characteristics in the population. Pick the same amount of samples from each of the different groups (strata) in the population.

Cluster Sampling = Divide the population into groups (clusters) and pick samples from those groups. The cluster or group of people are formed from the population set. The group has similar significatory characteristics. Also, they have an equal chance of being a part of the sample. This method uses simple random sampling for the cluster of population.

Let´s see some examples of how to get samples.


```python
# Convenience samples
convenience_samples = normal_dist[0:5]

# Purposive samples (Pick samples for a specific purpose)
# In this example we pick the 5 highest values in our distribution
purposive_samples = normal_dist.nlargest(n=5)

# Simple (pseudo)random sample
random_samples = normal_dist.sample(5)

# Systematic sample (Every 2000th value)
systematic_samples = normal_dist[normal_dist.index % 2000 == 0]

# Stratified Sampling
# We will get 1 person from every classroom in the dataset

df = pd.read_csv('dataset.csv')

strat_samples = []

for classroom in df['Classroom'].unique():
    samp = df[df['Classroom'] == classroom].sample(1)
    strat_samples.append(samp['Average_grade'].item())
    
print('Stratified samples:\n\n{}\n'.format(strat_samples))

# Cluster Sampling
# Make random clusters of ten people (Here with replacement)
c1 = normal_dist.sample(10)
c2 = normal_dist.sample(10)
c3 = normal_dist.sample(10)
c4 = normal_dist.sample(10)
c5 = normal_dist.sample(10)


# Take sample from every cluster (with replacement)
clusters = [c1,c2,c3,c4,c5]
cluster_samples = []
for c in clusters:
    clus_samp = c.sample(1)
    cluster_samples.extend(clus_samp)
print('Cluster samples:\n\n{}'.format(cluster_samples))    
```

**Imbalanced datasets**


An imbalanced dataset is a dataset where classes are distributed unequally. An imbalanced data can create problems in the classification task. If we are using accuracy as a performance metric, it can create a huge problem. Let’s say our model predicts if a bank transaction was a fraud transaction or a legal transaction. If the total legal transactions represented 99.83%, using an accuracy metric on the credit card dataset will give 99.83% accuracy, which is excellent. Would it be a good result? No.

For an imbalanced dataset, other performance metrics should be used, such as the Precision-Recall AUC score, F1 score, etc. Moreover, the model will be biased towards the majority class. Since most machine learning techniques are designed to work well with a balanced dataset, we must create balanced data out of an imbalanced dataset, but first we must split the dataset into training and testing because the change is only for the training purpose.

A common way to deal with imbalanced datasets is **resampling.** Here are two possible resampling techniques:

-Use all samples from our more frequently occuring event and then randomly sample our less frequently occuring event (with replacement) until we have a balanced dataset.

-Use all samples from our less frequently occuring event and then randomly sample our more frequently occuring event (with or without replacement) until we have a balanced dataset.



### BIAS, MSE, SE

**Bias** is the difference between the calculated value of a parameter and the true value of the population parameter being estimated.
  We can say that it is a measure of how far the sample mean deviates from the population mean. The sample mean is also called Expected value.

  For example, if we decide to survey homeowners on the value of their houses and only the wealthiest homeowners respond, then our home value estimate will be **biased** since it will be larger thant the true value for our population.

Let's see how would we calculate the population mean, expected value and bias using Python:


```python
# Take sample
df_sample = df.sample(100)

# Calculate Expected Value (EV), population mean and bias
ev = df_sample.mean()[0]
pop_mean = df.mean()[0]
bias = ev - pop_mean

print('Sample mean (Expected Value): ', ev)
print('Population mean: ', pop_mean)
print('Bias: ', bias)
```

**Mean Squared Error (MSE)** is a formula to measure how much estimators deviate from the true distribution. This can be very useful evaluating regression models.

**Standard Error (SE)** is a formula to measure how spread the distribution is from the sample mean.


```python
from math import sqrt
from scipy.stats import sem

Y = 100 # True Value
YH = 85 # Predicted Value

# MSE  
def MSE(Y, YH):
    return np.square(YH - Y).mean()

# RMSE 
def RMSE(Y, YH):
    return sqrt(np.square(YH - Y).mean())


print('MSE: ', MSE(Y, YH))
print('RMSE: ', RMSE(Y, YH))

#SE

norm_sample = normal_dist.sample(100)

print('Standard Error of normal sample: ', sem(norm_sample))
```

### Introduction to confidence levels and confidence intervals

To deal with uncertainty, we can use an interval estimate. It provides a range of values that best describe the population. 
To develop an interval estimate we first need to learn about confidence levels.

**Confidence Levels**



A confidence level is the probability that the interval estimate will include the population parameter (such as the mean).

A parameter is a numerical description of a characteristic of the population.

![hypothesis_testing_standard_normal_distribution.jpg](../assets/hypothesis_testing_standard_normal_distribution.jpg)

Sample means will follow the normal probability distribution for large sample sizes (n>=30)

To build an interval estimate with a 90% confidence level

Confidence level corresponds to a z-score from the standard normal table equal to 1.645

![hypothesis_testing_confidence_interval.jpg](../assets/hypothesis_testing_confidence_interval.jpg)

**Confidence Interval**


A confidence interval is a range of values used to estimate a population parameter and is associated with a specific confidence level.
The confidence interval needs to be described in the context of several samples.

Lets build a confidence interval around a sample mean using these equations:
    
x̄ ± z c 
    
Where:
    
x̄ = the sample mean

z = the z-score, which is the number of standard deviations based on the confidence level

c = the standard error of the mean

Select 10 samples and construct 90% confidence intervals around each of the sample means.
Theorethically, 9 of the 10 intervals will contain the true population mean, which remains unknown.

![hypothesis_testing_confidence_interval_example.jpg](../assets/hypothesis_testing_confidence_interval_example.jpg)

Do not misinterpret the definition of a confidence interval:

False: 'There is a 90% probability that the true population mean is within the interval'.

True: 'There is a 90% of probability that any given confidence interval from a random sample will contain the true population mean.'


To summarize, when we create a confidence interval, it's important to be able to interpret the meaning of the confidence level we used and the interval that was obtained. 
The confidence level refers to the long-term success rate of the method, which means, how often this type of interval will capture the parameter of interest.
A specific confidence interval gives a range of plausible values for the parameter of interest.

### Steps of formulating a hypothesis

Hypotheses are claims, and we can use statistics to prove or disprove them. Hypothesis testing structures the problems so that we can use statistical evidence to test these claims and check if the claim is valid or not.

1. Defining hypothesis

2. Assumption check

3. Set the significance levels

4. Selecting the proper test

5. Carrying out the hypothesis testing and calculate the test statistics and corres­­po­nding P-va­lue

6. Compare P-value with signif­icance levels and then decide to accept or reject null hypothesis

**1. Defining our null and alternative hypothesis**

First of all, we need to understand which scientific question we are looking for an answer to, and it should be formulated in the form of the Null Hypothesis (H₀) and the Alternative Hypothesis (H₁ or Hₐ). 

What is the null hypothesis? (Ho)

The null hypothesis is a statement about a population parameter, the statement assumed to be true before we collect the data. We test the likelihood of this statement being true in order to decide whether to accept or reject our alternative hypothesis. It can be thought of as the 'control' of the experiment and it usually has some equal sign (>=, <=, =)

What is the alternatuve hypothesis? (Ha)

A statement that directly contradicts the null hypothesis. This is what we want to prove to be true with our collected data. Can be thought of as the 'experiment'. It usually has the opposite sign of the null hypothesis.


**Fact:** Sample statistics is not what should be involved in our hypothesis. Our hypothesis are claims about the population that we want to study. Please remember that H₀ and Hₐ must be mutually exclusive, and Hₐ shouldn’t contain equality.

**2. Assumption Check**

To decide whether to use the parametric or nonparametric version of the test, we should verify if:

-Observations in each sample are independent and identically distributed (IID).

-Observations in each sample are normally distributed.

-Observations in each sample have the same variance.

The next thing we do is set up a threshold known as the significance level.

**3. Level of Significance**



As there is a 90% probability that any given confidence interval will contain the true population mean, there is a 10% chance that it wont.
This 10% is known as the level of significance and is represented by the shaded area.

While a researcher performs research, a hypothesis has to be set, which is known as the null hypothesis. This hypothesis is required to be tested via pre-defined statistical examinations. This process is termed as statistical hypothesis testing.
Significance means 'not by chance' or 'probably true'.

The level of significance is defined as the fixed probability of wrong elimination of null hypothesis when in fact, it is true. Level of significance α (alpha) is the probability of making a type 1 error (false positive). The probability for the confidence interval is a complement to the significance level. 
A (1-α) confidence interval has a significance level equal to α.

**4. Selecting the proper test**

A variety of statistical procedures exist. The appropriate one depends on the research question(s) we are asking and the type of data we collected. We need to analyze how many groups are being compared and whether the data are paired or not. To determine whether the data is matched, it is necessary to consider whether the data was collected from the same individuals.

**T-Test** : for 1 independent variable with 2 categories, and the target variable.

When we wish to know whether the means of two groups in a students classroom (male and female in a gender variable) differ, a t test is appropriate. In order to calculate a t test, we need to know the mean, standard deviation, and number of individuals in each of the two groups. An example of a t test research question is “Is there a significant difference between the writing scores of boys and girls in the classroom?” A sample answer might be: “Boys (M=5.67, SD=.45) and girls (M=5.76, SD=.50) score similarly in writing, t(35)=.54, p>.05.” [Note: The (35) is the degrees of freedom for a t test. It is the number of individuals minus the number of groups (always 2 groups with a t-test). In this example, there were 37 individuals and 2 groups so the degrees of freedom is 37-2=35.] Remember, a t test can only compare the means of two groups of an independent variable (for example gender) on a single dependent variable (for example writing score).

A t-distribution is flatter than a normal distribution. As the degrees of freedom increase, the shape of the t-distribution becomes similar to a normal distribution. With more than 30 degrees of freedom (sample size of 30 or more) the two distributions are practically identical.

Types of T-test:

-Two sample t-test: If two indepe­ndent groups have different mean

-Paired T-test: if one groups have different means at different times

-One Sample T-test: mean of a single group against a known mean

Assump­tions about data

1. indepe­ndent

2. normally distri­buted

3. have a similar amount of variance within each group being compared


**F-test (ANOVA Analysis)**

One-way Anova : One independent variable with more than 2 categories, and the target variable.

If we have one independent variable (with three or more categories) and one dependent variable, we do a one-way ANOVA. A sample research question is, “Do Doctors, teachers, and engineers differ on their opinion about a tax increase?” A sample answer is, “Teachers (M=3.56, SD=.56) are less likely to favor a tax increase than doctors (M=5.67, SD=.60) or engineers (M=5.34, SD=.45), F(2,120)=5.67, p<.05.” [Note: The (2,207) are the degrees of freedom for an ANOVA. The first number is the number of groups minus 1. Because we had three professions it is 2, 3-1=2. The second number is the total number of individuals minus the number of groups. Because we had 210 subject and 3 groups, it is 207 (210 - 3)]. 

Two-way Anova : More than 1 independent variable with two or more categories each, and the target variable.

A two-way ANOVA has three research questions, one for each of the two independent variables and one for the interaction of the two independent variables.

Sample Research Questions for a Two-Way ANOVA:
Do teachers, doctors and engineers differ on their opinion about a tax increase?
Do males and females differ on their opinion about a tax increase?
Is there an interaction between gender and profession regarding opinions about a tax increase?

A two-way ANOVA has three null hypotheses, three alternative hypotheses and three answers to the research question.

Assump­tions about data:

1. each data y is normally distri­buted

2. the variance of each treatment group is same

3. all observ­ations are indepe­ndent

**Chi-square**

We might count the incidents of something and compare what our actual data showed with what we would expect. Suppose we surveyed 45 people regarding whether they preferred rock, reggae, or jazz as their favorite music type. If there were no preference, we would expect that 15 would select rock, 15 would select reggae, and 15 would select jazz. Then our sample indicated that 30 like rock, 5 like reggae and 10 like jazz. We use a chi-square to compare what we observe (actual) with what we expect. A sample research question would be: “Is there a preference for the rock, reggae or jazz music?” A sample answer is “There was not equal preference for rock, reggae and jazz. 

Just as t-tests tell us how confident we can be about saying that there are differences between the means of two groups, the chi-square tells us how confident we can be about saying that our observed results differ from expected results.

Types of Chi-Square test:

-Test for independence: tests for the indepe­ndence of two catego­rical variables.

-Homoge­neity of Variance: test if more than two subgroups of a population share the same multiv­ariate distri­bution.

-Goodness of fit: whether a multin­omial model for the population distri­bution (P1,....Pm) fits our data.

Test for indepe­ndence and homoge­neity of variance share the same test statistics and degree of freedoms by different design of experiment.

Assump­tions:

1. one or two catego­rical variables

2. indepe­ndent observ­ations

3. outcomes mutually exclusive

4. large n and no more than 20% of expected counts < 5


**Regression**


Sometimes we wish to know if there is a relationship between two variables. A simple correlation measures the relationship between two variables.

In regression, one or more variables (predictors) are used to predict an outcome (criterion). 

Linear Regression:

What are the five linear regression assumptions and how can you check for them?

1. Linearity: the target (y) and the features (xi) have a linear relationship. To verify linearity we can plot the errors against the predicted y and look for the values to be symmetrically distributed around a horizontal line with constant variance.

2. Independence: the errors are not correlated with one another. To verify independence we can plot errors over time and look for non-random patterns (for time series data).

3. Normality: the errors are normally distributed. We can verify normality by ploting the errors with a histogram.

4. Homoskedasticity: the variance of the error term is constant across values of the target and features. To verufy it we can plot the errors against the predicted y.

5. No multicollinearity: Look for correlations above ~0.8 between features.

![simple_linear_regression_formula.jpg](../assets/simple_linear_regression_formula.jpg)

**5. What is the p-value?**

When testing an Hypothesis, the p-value is the likelihood that we would observe results at least as extreme as our result due purely to random chance if the null hypothesis were true. We use p-values to make conclusions in significance testing. More specifically, we compare the p-value to a significance level to make conclusions about our hypotheses.

If the p-value is lower than the significance level we chose, then we reject the null hypothesis, in favor of the alternative hypothesis.

If the p-value is greater than or equal to the significance level, then we fail to reject the null hypothesis but this doesn't mean we accept it.


What value is most often used  to determine statistical significance?

A value of alpha = 0.05 is most often used as the threshold for statistical significance.

In other words, a low p-value means that we have compelling evidence to reject the null hypothesis. If the p-value is lower than 5% we often reject H0 and accept Ha is true. We say that p < 0.05 is statistically significant, because there is less than 5% chance that we are wrong in rejecting the null hypothesis.

One way to calculate the p-value is through a T-test. We can use Scipy's ttest_ind function to calculate the t-test for the means of two independent samples of scores.

Let's see an example:

Maria designed an experiment where some individuals tasted lemonades from four different cups and attempted to identify which cup contained homemade lemonade. Each subject was given three cups that contained bittled lemonade and one cup that contained homemade lemonade (the order was randomized). She wanted to test if the subjects could do better than simply guessing when identifying the homemade lemonade.

Her null hypothesis Ho : p = 0.30

Her alternative hypothesis Ha : p > 0.30

*p is the likelihood of individuals to identify the homemade lemonade*

The experiment showed that 18 of the 50 individuals correctly identified the homemade lemonade. Maria calculated that the statistic p = 18/50 = 0.3 had an associated p-value of approximately 0.061.

Using a significance level of α = 0.05 what conclusions can we get?

* Since the p-value is greater than the significance level, we should reject the null hypothesis Ho.
* We don't have enough evidence to say that these individuals can do better than guessing when identifying the homemade lemonade.
The null hypothesis H_0: p=0.25 says their likelihood is no better than guessing, and we failed to reject the null hypothesis.

**6. Decision and Conclusion**

After performing hypothesis testing, we obtain a related p-value that shows the significance of the test.

If the p-value is smaller than the significance level, there is enough evidence to prove H₀ is not valid; you can reject H₀. Otherwise, you fail to reject H₀.

### Potential Errors in testing

What is Type I Error?

Type I error is the rejection of a true null hypothesis, or a 'false positive' classification.

What is Type II error?

Type II error is the non-rejection of a false null hypothesis, or a false negative classification.

We can make four different decisions with hypothesis testing:

Reject H0 and H0 is not true (no error)
Do not reject H0 and H0 is true (no error)
Reject H0 and H0 is true (Type 1 Error)
Do not reject H0 and H0 is not true (Type 2 error)
Type 1 error is also called Alpha error. Type 2 error is also called Beta error.

### What is A/B testing and why is it useful?

An A/B test is a controlled experiment where two variants (A and B) are tested against each other on the same response.

For example, a company could test two different email subject lines and then measure which one has the higher open rate. Once the superior variant has been determined (through statistical significance or some metric), all future customers will typically only receive the "winning" variant.

A/B testing is useful because it allows practitioners to rapidly test variations and learn about an audience preferences.

### Implications in Machine Learning

The central limit theorem has important implications in applied machine learning. The theorem does inform the solution to linear algorithms such as linear regression, but not exotic methods like artificial neural networks that are solved using numerical optimization methods.

**Covariance **

Covariance is a measure of how much two random variables vary together. If two variables are independent, their covariance is 0. However, a covariance of 0 does not imply that the variables are independent.


```python
#Code to get the covariance between two variables.

df[['Feature1', 'Feature2']].cov()

# Correlation between two normal distributions using Pearson's correlation

df[['Feature1', 'Feature2']].corr(method='pearson')
```

**Significance Tests**


-In order to make inferences about the skill of a model compared to the skill of another model, we must use tools such as statistical significance tests.

-These tools estimate the likelihood that the two samples of model skill scores were drawn from the same or a different unknown underlying distribution of model skill scores.

-If it looks like the samples were drawn from the same population, then no difference between the models skill is assumed, and any actual differences are due to statistical noise.

-The ability to make inference claims like this is due to the central limit theorem and our knowledge of the Gaussian distribution and how likely the two sample means are to be  a part of the same Gaussian distribution of sample means.

**Confidence intervals**

-Once we have trained a final model, we may wish to make  an inference about how skillful the model is expected to be in practice.

-The presentation of this uncertainty is called a confidence interval.

-We can develop multiple independent (or close to independent) evaluations of a model accuracy to result in a population of candidate skill estimates.

-The mean of this skill estimates will be an estimate (with error) of the true underlying estimate of the model skill on the problem.

-With knowledge that the sample mean will be a part of a Gaussian distribution from the central limit theorem, we can use knowledge of the Gaussian distribution to estimate the likelihood of the sample mean based on the sample size and calculate an interval of desired confidence around the skill of the model.

**Linear Regression in Machine Learning**



```python
# Linear regression from scratch
import random
# Create data from regression
xs = np.array(range(1,20))
ys = [0,8,10,8,15,20,26,29,38,35,40,60,50,61,70,75,80,88,96]

# Put data in dictionary
data = dict()
for i in list(xs):
    data.update({xs[i-1] : ys[i-1]})

# Slope
m = 0
# y intercept
b = 0
# Learning rate
lr = 0.0001
# Number of epochs
epochs = 100000

# Formula for linear line
def lin(x):
    return m * x + b

# Linear regression algorithm
for i in range(epochs):
    # Pick a random point and calculate vertical distance and horizontal distance
    rand_point = random.choice(list(data.items()))
    vert_dist = abs((m * rand_point[0] + b) - rand_point[1])
    hor_dist = rand_point[0]

    if (m * rand_point[0] + b) - rand_point[1] < 0:
        # Adjust line upwards
        m += lr * vert_dist * hor_dist
        b += lr * vert_dist   
    else:
        # Adjust line downwards
        m -= lr * vert_dist * hor_dist
        b -= lr * vert_dist
        
# Plot data points and regression line
plt.figure(figsize=(15,6))
plt.scatter(data.keys(), data.values())
plt.plot(xs, lin(xs))
plt.title('Linear Regression result')  
print('Slope: {}\nIntercept: {}'.format(m, b))
```

Source:

https://www.kaggle.com/code/carlolepelaars/statistics-tutorial/notebook
    
https://byjus.com/maths/level-of-significance/

https://byjus.com/maths/sampling-methods/

https://researchbasics.education.uconn.edu/anova_regression_and_chi-square/#

https://cheatography.com/mmmmy/cheat-sheets/hypothesis-testing-cheatsheet/



