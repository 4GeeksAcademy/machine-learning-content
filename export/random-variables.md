
# Random Variables

A random variable is a variable that can take on different values randomly.

On its own, a random variable is just a description of the states that are possible; it must be coupled with a probability distribution that specifies how likely each of these states are.

Random Variables may be discrete or continuous.

-A discrete random variable is one that has a finite or countably infinite number of states. Note that these states are not necessarily the integers. They can also just be named states that are not considered to have any numerical value.

-A continuous random variable is associated with a real value.

**Probability Distribution:** description of how likely a random variable or set of random variables is to take on each of its possible states. The way we describe probability distributions depends on whether the variables are discrete or continuous.


### Types of random variables

**Discrete random variable:**

-It has a finite number of states, not necesarily integers. They can also be a named states that are not considered to have any numerical value. Example: Coin toss (2 states), Throwing a dice (6 states), Drawing a card from a deck of cards (52 states), etc.

**Continuous random variable:**

-Must be associated with a real value. Example: Rainfall on a given day (in centimeters), Stock price of a company, Temperature of a given day.


```python
#graph
```

### Probability Distributions

**Bernoulli random variable**

It models a trial that results in success or failure. For example, heads or tails.

The *Bernoulli Distribution* is a distribution over a single binary random variable.

It is controlled by a single parameter, which gives the probability of the random variable being equal to 1.

It has the following properties:


```python
#graph
```

**Uniform random variable**

Models when there is complete ignorance that a result would have more probability than others. So all possible outcomes are equally likely.


```python
#graph
```

**Binomial Random Variable**

The *Binomial Distribution* with parameters n and p is the discrete probability distribution of the number of successes in a sequence of n independent experiments. It is frequently used to model the number of successes in a sample of size n drawn with replacement from a population of size N.

Binomial probability distributions help us to understand the likelihood of rare events and to set probable expected ranges.


```python
#graph
```


Experiment: n independent tosses of a coin within p(Heads) = p

Sample space: set of sequences of Heads and Tails, of length n

Random Varialbe X: number of Heads observed

Model of: number of successes in a given number of independent trials


```python
#big graph
```

**Geometric Random Variables**

Sample space: set of infinite sequences of Heads and Tails
Random variable X: number of tosses until the first Heads
Model of: waiting times, number of trials until a success


```python
#graph
```

**Poisson**

It is a discrete probability distribution that expresses the probability of a given number of events occuring in a fixed  interval of time or space.

Some examples that may follow a Poisson distribution are:

-Number of phone calls received by a call center per hour.
-Number of patients arriving in an emergency room bewteen 10pm and 11pm.



```python
#graph
```

**Normal**

Finally the most commonly used distribution over real numbers, also known as Gaussian distribution.

The two parameters are μ (mu = population mean) and σ (sigma = population standard deviation) control the normal distribution.

The parameter μ gives the coordinate of the central peak, which is also the mean of the distribution E[X] = μ.

The standard deviation of the distribution is given by σ and the variance by σ2.

Normal distributions are a sensible choice for many applications.

The central limit theorem shows that the sum of many independent random variables is approximately normally distributed. This means in practice, that many complicated systems can be modeled succesfully as normally.

Normal distribution encodes the maximum amount of uncertainty over the real numbers. We can thus think of the normal distribution as being the one that inserts the least amount of prior knowledge into a model.


```python
#graph
```

### EXPECTED VALUE

The expected value is exactly what it sounds like, what do you expect the value to be? 
The expected value is the mean of a random variable.
You can use this to work out the average score of a dice roll over 6 rolls.

Given the outcomes=(1, 2) and the probabilities=(1/8, 1/4) the expected value, E[x] is E[x] = 1(1/8) + 2(1/4) = 0.625.


```python
#graph
```

Interpretation: average in large number of independent repetitions.
    
Caution: If we have an infinite sum, it needs to be well-defined.

**Expectation of a Bernoulli random variable**

**Expectation of a Uniform random variable**

**Rules**

**Linearity of Expected Value**

E[F+G] = E[F] + E[G].

**Multiplication Rule of Expected Value**

E[F x G] does not equal E[F] * E[G]
