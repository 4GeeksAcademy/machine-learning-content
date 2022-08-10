# Getting started with Machine learning on AWS Cloud tools

Regardless of the problem you are working on, you normally have to go through the following steps:

We start with our business problem and spend some time converting it into a machine learning problem, which is not a simple process. Then we start collecting the data, and once we have all our data together, we visualize it, analyse it, and do some feature engineering to finally have clean data ready to train our model. We probably won't have our ideal model since the beginning, so with model evaluation we measure how the model is doing, if we like the perfomance or not, if it is accurate or not, and then we start to optimize it by tuning some hyperparameters. 

Once we are satisfied with our model we need to verify if it satisfies our initial business goal, otherwise we would have to work on feature augmentation or in collecting more data. Once our business goal is satisfied, we can deploy our model to make predictions in a production environment and it doesn't end there because we want to keep them up to date and current so we keep retraining them with more data. While in software you write rules to follow, in machine learning the model figures out the rule based on the data that it has been trained on. So in order to stay current you need to retrain your model on current data.

It is not simple but we have already learned how to do all this on our own. The good news about cloud computing is that we can implement some ML solutions without having to go through each of the previous steps.

Because it is currently the number one cloud computing provider, We chose AWS to learn some cloud computing skills. In the following image we can see the three layer AWS machine learning stack.

![aws_stack](../assets/aws_stack.jpg)

In the bottom of the stack we can find 'ML frameworks & infrastructure', which is what AWS would call the "hard" way of doing machine learning, by running some virtual machines where we are able to use GPUs if we need them and install some frameworks, for example Tensorflow, to start doing all the steps mentioned above.

There is an easier way, which is the 'ML services'. This is all about the service called SageMaker. 
SageMaker is a service that has the previous pipeline ready for you to use but you still need to know about the algorithms that you want to use, and you still need to code a little bit if you want to go a little bit deeper.

Now let's see the easiest way at the top of the stack image. In 'AI Models' the models are built already. We use for example a natural language processing service called 'Amazon Translate'.
AI services are a great way to try AI, specially if you don´t have any background, or if you are working in some rapid experimentation, they are a quick way to get into the business value, and if you find where the business value is and you need something more customized, then you can move down the stack to the next layer.

The great thing about this AI services APIS is that, as a developer, you can jumpstart to experiment instead of having to learn a lot of stuff before start using them, and then you can go deeper and customize them.

Three things that developer need to learn to get the most out of this services:

1. Understand your data, not only in AI services, but in all machine learning.
2. Understand your use case, test the service with your particular use case, not just the generic one.
3. Understand what success looks like. Machine learning is very powerful but it is not going to be 100% accurate.