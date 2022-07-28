# Deployment of a Machine learning model using Flask and Heroku 

Flask is a small and lightweight Python web framework that provides useful tools and features that make creating web applications easier using only a Python file. 

Heroku is a cloud platform that lets you build, deliver, monitor and scale apps. Heroku makes the processes of deploying, configuring, scaling, tuning, and managing apps as simple and straightforward as possible so that developers can focus on building great apps. It also includes a rich ecosystem of managed data services.

## Steps to create a web app using Flask in Python3

### **Step 1:** Activate environment and install Flask

```bash
pip install Flask
```

### **Step 2:** Create a basic application

In your directory, open a file named hello.py for editing. This hello.py file will serve as a minimal example of how to handle HTTP requests. Inside, you will import the Flask object, and create a function that returns an HTTP response. Write the following code inside hello.py:

```py
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'
```

Let's explain what the previous code just did. It first imports the Flask object from the flask package. You will then use it to create your Flask application instance with the name app. Pass the special variable __name__ which holds the name of the current Python module. It is used to tell the instance where it is located. You will need to do this because Flask sets up some paths in the background.

Once you create the app instance, you use it to handle incoming web requests and send responses to the user. @app.route is a decorator that converts a regular Python function into a Flask view function, which converts the function's return value into an HTTP response that will be displayed by an HTTP client, such as a web browser. Pass the value '/' to @app.route() to indicate that this function will respond to web requests for the URL /, which is the primary URL.

The hello() view function returns the string 'Hello, World!' in response.

Save and close the file.

To run your web application, you will first tell Flask where to find the application (the hello.py file in your case) with the `FLASK_APP` environment variable:

```bash
export FLASK_APP=hello
```

Then, execute it in development mode with the environment variable `FLASK_ENV`:

```bash
export FLASK_ENV=development
```

Finally, execute the app using `flask run`:

```py
flask run
```

Once it is running, the result should look similar to this:

```bash
Output
 * Serving Flask app "hello" (lazy loading)
 * Environment: development
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 813-894-335
 ```

 The above result has several information, such as:

-The name of the application you are running.

-The environment in which the application is running.

-Debug mode: on means that the Flask debugger is running. This is useful during development because it gives us detailed error messages when something goes wrong, which makes it easier to troubleshoot problems.

-The application runs locally on the URL http://127.0.0.1:5000/, 127.0.0.1 is the IP representing your computer's localhost and :5000 is the port number.

Now open a browser and type the URL http://127.0.0.1:5000; you will receive the string Hello, World! in response. This confirms that your application is running correctly.

### **Step 3:** Using HTML templates

The Bootstrap toolkit will help you give some style to your application without having to write your own HTML, CSS and JavaScrip code. The toolkit will allow you to focus on learning how Flask works, instead of learning HMTL.





### **Heroku step: Create a new web app on Heroku**

You should already have an account on Heroku, but if you don't. go ahead and create your account at 'https://www.heroku.com'.

- Click on 'Create a new app'

- On 'deploy' tab: link Heroku app to your Github account and select the repo to connect to.

- Scroll down and choose 'manual deploy'. After making sure you are on the branch you want to deploy (in this case: main), then clic on 'Deploy branch'. You will see all the required packages been installed like the following image:

*include image*

- When finished, it should look like the following screenshot:

*include image*

- Copy that link and paste it in your browser to test your app.




## Structure of your folder

*include an ideal structure image* 



> Don't forget that before starting your app, you should save your model. You can use the code below.

```py
import pickle

filename = 'titanic_model.pkl'

pickle.dump(classifier, open(filename,'wb'))
```



**Using Flask to make a web API for our machine learning model**

What you normally used to name app.py will have to be renamed for example to 'titanic.py', or 'build_features.py', etc because your app.py file will now be in charge of building the web app.

In your app.py file your code should look something like this:

```py
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('titanic_model.pkl','rb'))

@app.route('/') #http://www.google.com/
def home():
    return render_template('index.html')          

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)

    return render_template('index.html',prediction_text='Would you survive? {} (1=survived, 0=deceased)'.format(output))

if __name__=="__main__":
    app.run(port=5000, debug=True)
```


Source:

https://medium.com/towards-data-science/considerations-for-deploying-machine-learning-models-in-production-

https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3-es

https://medium.com/towards-data-science/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b

https://medium.com/towards-data-science/create-an-api-to-deploy-machine-learning-models-using-flask-and-heroku-67a011800c50

https://medium.com/towards-data-science/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

https://medium.com/towards-data-science/flask-and-heroku-for-online-machine-learning-deployment-425beb54a274
