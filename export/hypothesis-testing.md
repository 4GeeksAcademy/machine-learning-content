
# HYPOTHESIS TESTING

**Central Limit Theorem**

The central limit theorem describes the shape of the distribution of sample means as a Gaussian or Normal Distribution.

The theorem states that as the size of the sample increases, the distribution of the mean across multiple samples will approximate a Gaussian distribution.

It demonstrates that the distribution of errors from estimating the population mean fit a normal distribution.

The estimate of the Gaussian distribution will be more accurate as the size of the samples drawn from the population is increased. This means that if we use our knowledge of the Gaussian distribution in general to start making inferences about the means of samples drawn from a population, that these inferences will become more useful as we increase our sample size.

The central limit theorem does not state anything about a single sample mean, instead it states something about the shape  or the distribution of sample means.

**Interval estimate**

To deal with uncertainty, we can use an interval estimate.

It provides a range of values that best describe the population.

To develop an interval estimate we need to learn about confidence levels.

**Confidence Levels**

A confidence level is the probability that the interval estimate will include the population parameter (such as the mean).

A parameter is a numerical description of a characteristic of the population.


```python
#graph of normal distribution
```

Sample means will follow the normal probability distribution for large sample sizes (n>=30)

To build an interval estimate with a 90% confidence level

Confidence level corresponds to a z-score from the standard normal table equal to 1.645


```python
#graph by me normal dist
```

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


```python
#graph
```

Do not misinterpret the definition of a confidence interval:

False: 'There is a 90% probability that the true population mean is within the interval'
True: 'There is a 90% of probability that any given confidence interval from a random sample will contain the true population mean.'

**Level of Significance**

As there is a 90% probability that any given confidence interval will contain the true population mean, there is a 10% chance that it wont.
This 10% is known as the level of significance and is represented by the shaded area.

While a researcher performs research, a hypothesis has to be set, which is known as the null hypothesis. This hypothesis is required to be tested via pre-defined statistical examinations. This process is termed as statistical hypothesis testing.
Significance means 'not by chance' or 'probably true'.

The level of significance is defined as the fixed probability of wrong elimination of null hypothesis when in fact, it is true. Level of significance α (alpha) is the probability of making a type 1 error (false positive). The probability for the confidence interval is a complement to the significance level. 
A (1-α) confidence interval has a significance level equal to α.

**When σ is unknown**

So far our examples have assumed we know σ -the population standard deviation.

If σ is unknown we can substitute s (sample standard deviation) for σ.

**Confidence intervals for the mean with small samples**

So far we have discussed confidence intervals for the mean where n >= 30.

When σ is known, we are assuming the population is normally distributed and so we can follow the procedure for large sample sizes.
When σ is unknown (more often the case) we make adjustments.

**When σ is unknown-small samples**

-Substitute s, sample standard deviation, for σ

-Because of the small sample size, this substitution forces us to use the t-distribution probability distribution.

-Continuous probability distribution

-Bell-shaped and symmetrical around the mean

-Shape of curve depends on degrees of freedom (d.f.) which equals n-1

### Implications in Machine Learning

The central limit theorem has important implications in applied machine learning. The theorem does inform the solution to linear algorithms such as linear regression, but not exotic methods like artificial neural networks that are solved using numerical optimization methods.

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


```python
Source:
    
https://byjus.com/maths/level-of-significance/
    
!(descripcion de imagen)[./assets/imagen.png]
```
