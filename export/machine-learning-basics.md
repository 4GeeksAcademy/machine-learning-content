# Machine Learning Basics

What is Machine Learning?

Machine learning is the field of science that studies algorithms that approximate functions increasingly well as they are given more observations. Machine learning algorithms are often used to learn and automate human processes, optimize outcomes, predict outcomes, model complex relationships and learn patterns in data. 

**Supervised vs Unsupervised learning**

Supervised learning...

Labeled data is data that has the information about target variable for each instance.
The most common uses of supervised learning are regression and classification.

Unsupervised learning...

The most common uses of unsupervised machine learning are clustering, dimensionality reduction, and association-rule mining.

**Online vs offline learning**

Online learning refers to updating models incrementally as they gain more information.

Offline learning refers to learning by batch processing data. If new data comes in, an entire new batch (including all the old and new data) must be fed into the algorithm to learn from the new data.

**Reinforcement learning**

Reinforcement learning describes a set of algorithms that learn from the outcome of each decision. For example, a robot could use reinforcement learning to learn that walking forward into a wall is bad, but turning away from a wall and walking is good. 

**What is the difference between bagging and boosting?**

Bagging and boosting are both ensemble methods, meaning they combine many weak predictors to create a strong predictor. One key difference is that bagging builds independent models in parallel, whereas boosting builds models sequentially, at each step emphasizing the observations that were missed in previous steps.

### How is data divided?

What is training data and what is it used for?

Training data is a set of examples that will be used to train the machine learning model.
For supervised machine learning, this training data must have a labeled. What you are trying to predict must be defined.
For unsupervised machine learning, the training data will contain only features and will use no labeled targets. What you are trying to predict is not defined.

What is a validation set and why use one?

A validation set is a set of data that used to evaluate a model's performance during training/model selection. After models are trained, they are evaluated on the validation set to select the best possible model.

It must never be used for training the model directly.

It must also not be used as the test data set because we have biased our model selection toward working well on this data, even tough the model was not directly trained on it.

What is a test set and why use one?

A test set is a set of data not used during training or validation. The model's performance is evaluated on the test set to predict how well it will generalize to new data.

### Training and validation techniques

**Train-test-split**

**Cross Validation**

Cross validation is a technique for more accurately training and validation models. It rotates what data is held out from model training to be used as the validation data.

Several models are trained and evaluated, with every piece of data being held out from one model. The average performance of all the models is then calculated.

It is a more reliable way to validate models but is more computationally costly. For example, 5-fold cross validation requires training and validating 5 models instead of 1.

**What is overfitting?**

Overfitting when a model makes much better predictions on known data (data included in the training set) than unknown data (data not included in the training set).

How can we combat overfitting?

A few ways of combatting overfitting are:

- simplify the model (often done by changing)

- select a different model

- use more training data

- gather better quality data to combat overfitting

How can we tell if our model is overfitting the data?

If our training error is low and our validation error is high, then our model is most likely overfitting our training data.

How can we tell if our model is underfitting the data?

If our training and validation error are both relatively equal and very high, then our model is most likely underfitting our training data.

**What are data pipelines?**

Any collection of ordered transformations on data

Source:
    
https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

https://www.kdnuggets.com/2020/09/understanding-bias-variance-trade-off-3-minutes.html

https://medium.com/@ranjitmaity95/7-tactics-to-combat-imbalanced-classes-in-machine-learning-datase-4266029e2861

