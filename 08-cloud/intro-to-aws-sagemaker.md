# Getting started with Machine learning on AWS tools

Regardless of the problem you are working on, you normally have to go through the following steps:

We start with our business problem and spend some time converting it into a machine learning problem, which is not a simple process. Then we start collecting the data, and once we have all our data together, we visualize it, analyze it, and do some feature engineering to finally have clean data ready to train our model. We probably won't have our ideal model from the beginning, so with model evaluation, we measure how the model is doing, if we like the performance or not, if it is accurate or not, and then we start to optimize it by tuning some hyperparameters. 

Once we are satisfied with our model, we need to verify if it satisfies our initial business goal; otherwise, we would have to work on feature augmentation or collecting more data. Once our business goal is satisfied, we can deploy our model to make predictions in a production environment, and it doesn't end there because we want to keep them up to date and current, so we keep retraining them with more data. While in software you write rules to follow, in machine learning the model figures out the rule based on the data that it has been trained on. So, in order to stay current, you need to retrain your model on current data.

It is not simple but we have already learned how to do all this on our own. The good news about cloud computing is that we can implement some ML solutions without having to go through each of the previous steps.

Because it is currently the number one cloud computing provider, We chose AWS to learn some cloud computing skills. In the following image, we can see the three-layer AWS machine learning stack.

![AWS stack](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/aws_stack.jpg?raw=true)

At the bottom of the stack, we can find 'ML frameworks & infrastructure', which is what AWS would call the "hard" way of doing machine learning, by running some virtual machines where we are able to use GPUs if we need them and installing some frameworks, for example TensorFlow, to start doing all the steps mentioned above.

There is an easier way, which is the 'ML services'. This is all about the service called SageMaker. 
SageMaker is a service that has the previous pipeline ready for you to use, but you still need to know about the algorithms that you want to use, and you still need to code a little bit if you want to go a little bit deeper.

Now let's see the easiest way at the top of the stack image. In 'AI Models', the models are built already. We use, for example, a natural language processing service called 'Amazon Translate'.

AI services are a great way to try AI, specially if you don't have any background or if you are working on some rapid experimentation. They are a quick way to get into the business value, and if you find where the business value is and you need something more customized, then you can move down the stack to the next layer.

The great thing about these AI services APIs is that, as a developer, you can jumpstart to experiment instead of having to learn a lot of stuff before starting to use them, and then you can go deeper and customize them.

There are three things that developers need to learn to get the most out of these services:

1. Understand your data, not only in AI services, but in all machine learning.

2. Understand your use case, test the service with your particular use case, not just the generic one.

3. Understand what success looks like. Machine learning is very powerful, but it is not going to be 100% accurate.

## Amazon SageMaker

**What is Amazon SageMaker?**

Amazon SageMaker provides machine learning capabilities for data scientists and developers to prepare, build, train, and deploy high-quality ML models efficiently.

**SageMaker Workflow:**

1. Label data: Set up and manage labeling jobs for highly accurate training datasets within Amazon SageMaker, using active learning and human labeling.

2. Build: Connect to other AWS services and transform data in Amazon SageMaker notebooks.

3. Train: Use Amazon SageMaker's algorithms and frameworks, or bring your own, for distributed training.

4. Tune: Amazon SageMaker automatically tunes your model by adjusting multiple combinations of algorithm parameters.

5. Deploy: After training is completed, models can be deployed to Amazon SageMaker endpoints for real-time predictions.

6. Discover: Find, buy and deploy ready-to-use model packages, algorithms, and data products in AWS marketplace.

**SageMaker benefits**

- For data exploration and preprocessing, it provides fully managed instances running Jupyter notebooks that include example code for common model training and hosting exercises.

- When you are ready to train your data, simply indicate the type and quantity of instances you need and start the training with a single click.

- It provides machine learning algorithms highly optimized speed, accuracy and scaling to run on extremely large training datasets.

- SageMaker provides model artifacts and scoring images for deployment to Amazon EC2 or anywhere else.

![Amazon SageMaker](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/sagemaker.jpg?raw=true)

## SageMaker Studio

When you log in to Amazon Web Services and choose SageMaker, there are some important first steps to be taken:

First, we need to create a Studio domain. This is a one time process you do in your AWS console just for SageMaker. 

Once created, AWS manages the server for you, and you, as a consumer, will log in to a user profile. You can have as many user profiles as you want per Studio domain. What normally happens in organizations is that they give a Studio domain to a team of data scientists, and they create as many user profiles as they need. 

Every time you create a user profile, it is again the Jupyter server that AWS manages, but every time you create a new notebook, app or machine, they will be running on a new EC2 instance each, so you have to be cautious because all your running instances are what will be charged, so make sure to log out or, if you are not using an instance, shut it down so that you are not charged. The good part is that you can have as many machines running as you want and have them running in all different environments, so you can have an R machine, a Spark machine, a PyTorch machine, a TensorFlow machine, among others.

Reasons to pick SageMaker Studio:

- You have more compute power, you can have more machines to run on top of.

- You can add as much data to the server and it just grows.

- All the widgets that it has: Projects, Data Wrangler, Feature Store, Pipelines, Experiments and trials, Model registry, Endpoints, etc.

### Example code used in SageMaker Studio

In the left control panel of SageMaker you will find SageMaker Studio. Once you click on it, you will find a kind of JupyterLab environment, and you will be presented with a Launch screen.

**Background**

In this SageMaker example, you will learn the steps to build, train, tune and deploy a fraud detection model.

The following steps include the preparation of your SageMaker notebook, downloading data from the internet into SageMaker, transforming the data, using Gradient Boosting algorithm to create a model, evaluating its effectiveness and setting the model up to make predictions.

**Preparation**

Once you install Pandas, you need to specify some details:

- The S3 bucket and prefix that you want to use for training and modeling the data, because the data has to come from some storage. It is not recommended to have your data uploaded in a notebook because if you have a lot of data and you are constantly uploading it, you are constantly being charged for it, and it's also slower. S3 is definitely more friendly in terms of uploading and landing our datasets because it can store terabytes of data, and you can train your model on terabytes of data. Data can also be stored in the Elastic File System (EFS).

- The IAM role was used to give training and hosting access to your data. This is the role you have been assigned whenever we launched the SageMaker Studio. With `get_execution_role()` we fetch that role. It is important because it provides access to SageMaker to other AWS resources.

```py
!pip install --upgrade pandas
```

> It is important to mention that when you create a new notebook and its time to select the kernel, you can reuse the kernel from an existing session, so that it has all the same packages, to avoid reinstalling. Another way is separately building a base docker image and attach it to your Studio domain.

```py
import sagemaker # import the sagemaker python sdk, similar to other python packages but with different features
bucket = sagemaker.Session().default_bucket()
prefix = 'sagemaker/fraud-detection'

# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()

# Import the python libraries we will need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Ipython.display import Image  # To display images in the notebook
from Ipython.display import display  # To display outputs in the notebook
from time import gmtime, strftime  # For labeling SageMaker models
import sys
import math
import json
import zipfile
```

You don't strictly need the SageMaker Python SDK to use SageMaker. There are a couple other ways to invoke SageMaker APIs:

- You can use boto3.

- You can use the AWS CLI.

```py
# Make sure your Pandas version is set to 1.2.4 or later
# It is always a good idea to check versions
# As SageMaker is constantly adding features

pd.__version__
```

**Data**

In this example, the data is stored in S3 so we will download it from the public S3 bucket.

```py
!wget https://s3-us-west-2.amazonaws.com/sagemaker-e2e-solutions/fraud-detection/creditcardfraud.zip

# Unzipping the data
with zipfile.ZipFile('creditcardfraud.zip','r') as zip_ref:
    zip_ref.extractall('.')
```

Now let's take this into a Pandas dataframe and take a first look.

This data is optimized and already normalized, but the common scenarios with data are that we don't have any data, or that data is never clean. Normalization needs to be done in some cases, but it depends on the model you use. XGBoost tends to be very robust to non-normalized data, but if you are using K-means, for example, or any deep learning model, normalization should be done. 

```py
data = pd.read_csv('./creditcard.csv')
print(data.columns)
data[['Time','V1','V2','V27','V28','Amount','Class']].describe()
data.head(10)
```

This dataset has 28 columns Vi for i = 1...28 of anonymized features, along with columns for time, amount, and class. We already know that the Vi columns have been normalized to have a mean of 0 and a unit standard deviation as the result of PCA.

The class column corresponds to whether or not a transaction is fraudulent. You will see with the following code that the majority of data is non-fraudulent, with only 492 (.173%) against 284315 non-frauds (99.827%). 

```py
nonfrauds, frauds = data.groupby('Class').size()
print('Number of frauds: ', frauds)
print('Number of non-frauds: ', nonfrauds)
print('Percentage of fraudulent data: ', 100.*frauds/(frauds + nonfrauds))
```

Here, we are dealing with an imbalanced dataset, so the 'accuracy' metric would be a misleading metric as it will probably predict non-frauds in 100% of cases with a 99.9% accuracy. Other metrics, like recall or precision help us attack that problem. Another technique is oversampling the minority class data or undersampling the majority class data. You can read about SMOTE to understand both techniques.

Now let's separate our data into features and labels.

```py
feature_columns = data.columns[:-1]
label_column = data.columns[-1]

features = data[feature_columns].values.astype('float32')
labels = (data[label_column].values).astype('float32')
```

```py
# XGBoost needs the target variable to be the first one
model_data = data
model_data.head()
model_data = pd.concat([model_data['Class'], model_data.drop(['Class'], axis=1)], axis=1)
model_data.head()
```

Separating data into train and validation sets.

```py
train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729),
                                                [int(0.7 * len(model_data)), int(0.9 * len(model_data))])
train_data.to_csv('train.csv', header = False, index = False)
validation_data.to_csv('validation.csv', header = False, index = False)
```

We need the data before the training can begin, on S3. As we have downloaded the data from **S3**, we have to upload it back. We can create a Python script to preprocess the data and automatically upload it for us.

```py
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')) \
                                .upload_file('train.csv')  
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')) \
                                .upload_file('validation.csv')
s3_train_data = 's3://{}/{}/train/train.csv'.format(bucket, prefix)
s3_validation_data = 's3://{}/{}/validation/validation.csv'.format(bucket, prefix)
print('Uploaded training data location: {}'.format(s3_train_data))
print('Uploaded training data location: {}'.format(s3_validation_data))

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('Training artifacts will be uploaded to: {}'.format(output_location))
```

**Training**

The algorithm chosen for this example is XGBoost, so we have two options for training. The first one is to use the built-in XGBoost provided by AWS SageMaker, or we can use the open source package for XGBoost. In this particular example, we will use the built-in algorithm provided by AWS.

In SageMaker, behind the scenes is all container-based, so we need to specify the locations of the chosen algorithm containers. To specify the linear learner algorithm, we use a utility function to obtain its URL.

The complete list of built-in algorithms can be found here: https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html

So let's proceed with the training process. First, we need to specify the ECR container location for Amazon SageMaker's implementation of XGBoost.

```py
container = sagemaker.image_uris.retrieve(region=boto3.Session().region_name, framework='xgboost', version='latest')
```

Because we are training with the CSV file format, we'll create `s3_inputs` that our training function can use as a pointer to the files in S3, which also specifies that the content type is CSV.

```py
s3_input_train = sagemaker.inputs.TrainingInput(s3_data = 's3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data = 's3://{}/{}/validation'.format(bucket, prefix), content_type='csv')
```

When we are running a training job in SageMaker we are not using the compute power on the notebook. We are doing this training job on a machine in the cloud, and we need the flexibility to specify any machine that we want to use, based on the algorithm that we are using. For example, if we are using a deep learning model, we may want to use GPUs instead of CPUs, so we need that flexibility because we want to optimize on the cost as well. If we would have a GPU on our notebook, and we get distracted coding, we don't want to pay for that time because we are not really using the machine for what it was meant to be, so it becomes very important that in the training job we provide a certain type of machine, and when the training is done, it automatically terminates all the resources.

To sum up the idea, we need to do the training on a separate instance because, based on our job, our model and the size of our dataset, we need to specify the appropriate instance style.

In order to do that, we need to do that configuration by specifying training parameters to the estimator. This includes:

1. The XGBoost algorithm container.

2. The IAM role to use.

3. Training instance type and count.

4. S3 location for output data.

5. Algorithm hyperparameters.

And then a `.fit()` function specifying the s3 location for output data.

```py
sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(container,
                                    role, 
                                    instance_count = 1, # If you provide more than 1, it will be automatically done for you
                                    instance_type = 'ml.m4.xlarge',
                                    output_path = 's3://{}/{}/output'.format(bucket,prefix),
                                    sagemaker_session = sess)
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        num_round=100)

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

**Hosting**

Once the model is trained, we can use the estimator with `.deploy()`

```py
xgb_predictor = xgb.deploy(initial_instance_count = 1, # The word initial is because you can update it later to increase it
                            instance_type = 'ml.m4.xlarge')
```
 
**Evaluation**

Once deployed, you can evaluate it on the SageMaker notebook. In this case, we are only predicting whether it's a fraudulent transaction (1) or not (0), which produces a simple confusion matrix.

First, we need to determine how we pass data into and receive data from our endpoint. Our data is currently stored as NumPy arrays in the memory of our notebook instance. To send it in an HTTP POST request, we'll serialize it as a CSV string and then decode the resulting CSV.

> For inference with CSV format, SageMaker XGBoost requires that the data does not include the target variable.

```py
xgb_predictor.serializer = sagemaker.serializers.CSVSerializer()
```

Now, we'll use a function to:

- Loop over the test dataset.

- Split it into mini batches of rows.

- Convert those mini batches to CSV string payloads (we drop the target variable from the dataset first).

- Retrieve mini batch predictions by invoking the XGBoost endpoint.

- Collect predictions and convert the CSV output our model provides into a NumPy array.

```py
def predict(data, predictor, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = '.'.join([predictions, predictor.predict(array).decode('utf-8')])

    return np.fromstring(predictions[1:],sep=',')

predictions = predict(test_data.drop(['Class'], axis=1).to_numpy(), xgb_predictor)
```

Now let's check the confusion matrix to see how well we did.

```py
pd.crosstab(index=test_data.iloc[:,0], columns=np.round(predictions), rownames=['actual'], colnames=['predictions'])
```

> Due to randomized elements of the algorithm, your results may differ slightly.

**Hypertuning**

We can use Automatic Model Tuning (AMT) from SageMaker where you import the hyperparameters and then provide the range of those hyperparameters. In the following example, let's suppose we want to maximize the area under the curve (AUC) but we don't know which values of the eta, alpha, min_child_weight and max_depth hyperparameters to use to train the best model. To find the best values, we can specify a range of values, and SageMaker will look for the best combination of them to get the training job with the highest AUC.

```py
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
hyperparameter_ranges = {'eta': ContinuousParameter(0,1),
                        'min_child_weight': ContinuousParameter(0,10),
                        'alpha': ContinuousParameter(0,2),
                        'max_depth': IntegerParameter(1,10)}
```

```py
objective_metric_name = 'validation:auc'
```

```py
tuner = HyperparameterTuner(xgb,
                            objective_metric_name,
                            hyperparameter_ranges,
                            max_jobs = 9,
                            max_parallel_jobs = 3)
```

```py
tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

```py
boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName = tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']
```

```py
# returns the best training job name
tuner.best_training_job()
```

```py
# Deploy the best trained or user specified model to an Amazon SageMaker endpoint
tuner_predictor = tuner.deploy(initial_instance_count=1,
                                instance_type='ml.m4.xlarge')
```

```py
# Create a serializer
tuner_predictor.serializer = sagemaker.serializers.CSVSerializer()
```

**Optional Clean up**

When you are done with your notebook, the following code will remove the hosted endpoint you created and avoid any charges.

```py
# Store variables to be used in the next notebook
%store model_data
%store container
%store test_data

xgb_predictor.delete_endpoint(delete_endpoint_config=True)
```

Amazon SageMaker is part of the free tier of Amazon Web Services, so if you would like to dive deeper into this machine learning cloud tool, you can create an account in AWS and experiment with SageMaker. Please read the important information below.

> Accounts covered under the AWS Free Tier aren't restricted in what they can launch. As you ramp up with AWS, you might start using more than what's covered under the AWS Free Tier. Because these additional resources might incur charges, a payment method is required on the account. The AWS Free Tier doesn't cover all AWS services.
