**Regression**


In regression, one or more variables (predictors) are used to predict an outcome (criterion). 

Linear Regression:

What are the five linear regression assumptions and how can you check for them?

1. Linearity: the target (y) and the features (xi) have a linear relationship. To verify linearity we can plot the errors against the predicted y and look for the values to be symmetrically distributed around a horizontal line with constant variance.

2. Independence: the errors are not correlated with one another. To verify independence we can plot errors over time and look for non-random patterns (for time series data).

3. Normality: the errors are normally distributed. We can verify normality by ploting the errors with a histogram.

4. Homoskedasticity: the variance of the error term is constant across values of the target and features. To verufy it we can plot the errors against the predicted y.

5. No multicollinearity: Look for correlations above ~0.8 between features.


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
