# Deployment of a Machine learning model using Flask and Heroku 

Machine Learning models are powerful tools to make predictions based on available data. In order to make these models useful for society, they need to be deployed so that others can easily access them to make predictions. This can be done using Flask and Heroku.

Flask is a small and lightweight Python web framework that provides useful tools and features that make creating web applications easier using only a Python file. 

Heroku is a cloud platform that lets you build, deliver, monitor and scale apps. Heroku makes the processes of deploying, configuring, scaling, tuning, and managing apps as simple and straightforward as possible so that developers can focus on building great apps. It also includes a rich ecosystem of managed data services.

Let's imagine we have just finished creating our Titanic survival prediction model. Now what?

In order to predict with unknown data, we have to deploy it over the internet so that the outside world can use it.

For that, we will need to save the model so that we can predict the values later. We make use of *Pickle* in Python which is a powerful algorithm for serializing and deserializing a Python object structure, but there are other tools too. The following code saves the model using Pickle:

```py
# Serializing our model to a file called model.pkl
import pickle
filename = 'model.pkl'
pickle.dump(classifier, open(filename,'wb'))
```

## Steps to create a web app using Flask in Python3

For predicting survival in the Titanic from various attributes, we first need to collect the data (new attribute values) and then use the model we built to predict whether a passenger would survive or not in the Titanic. Therefore, in order to collect the data, we create an HTML form which would contain all the different options to select from each attribute. Here, I have created a simple form using only HTML. If you want to make the form more interactive, you can do so as well.

![Titanic prediction form](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/titanic_prediction_form.jpg?raw=true)

### Step 1: Activate environment and install Flask

In the command line, enter your project's directory. Once there, activate your environment and use `pip` to install Flask.

```bash
pip install Flask
```

### Step 2: Create a basic application

In your directory, open a file named `hello.py` for editing. This file will serve as a minimal example of how to handle HTTP requests. Inside, you will import the Flask object and create a function that returns an HTTP response. Write the following code inside `hello.py`:

```py
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'
```

Let's explain what the previous code just did. It first imports the Flask object from the flask package. You will then use it to create your Flask application instance with the name app. Pass the special variable `__name__` which holds the name of the current Python module. It is used to tell the instance where it is located. You will need to do this because Flask sets up some paths in the background.

Once you create the app instance, you use it to handle incoming web requests and send responses to the user. `@app.route` is a decorator that converts a regular Python function into a Flask view function, which converts the function's return value into an HTTP response that will be displayed by an HTTP client, such as a web browser. Pass the value `'/'` to `@app.route()` to indicate that this function will respond to web requests for the URL `/`, which is the primary URL.

The `hello()` view function returns the string 'Hello, World!' in response.

Save and close the file.

To run your web application, you will first tell Flask where to find the application (the `hello.py` file in your case) with the `FLASK_APP` environment variable:

```py
export FLASK_APP=hello
```

Then, execute it in development mode with the environment variable `FLASK_ENV`:

```py
export FLASK_ENV=development
```

Finally, execute the app using `flask run`:

```bash
flask run
```

Once it is running, the result should look similar to this:

```text
Output
 * Serving Flask app "hello" (lazy loading)
 * Environment: development
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 813-894-335
 ```

The above result has some information, such as:

- The name of the application you are running.

- The environment in which the application is running.

- **Debug mode: on** means that the Flask debugger is running. This is useful during development because it gives us detailed error messages when something goes wrong, which makes it easier to troubleshoot problems.

- The application runs locally on the URL `http://127.0.0.1:5000/`, *127.0.0.1* is the IP representing your computer's localhost and *:5000* is the port number.

Now open a browser and type the URL `http://127.0.0.1:5000`; you will receive the string `Hello, World!` in response. This confirms that your application is running correctly.

You now have a small Flask web application. You have run your application and displayed information in the web browser. Next, you will use the HTML files in your application.

### Step 3: Using HTML templates

Currently, your application only displays a simple message without HTML. Web applications primarily use HTML to display information to the visitor, so you will now work to incorporate a HTML file into your application, which can be displayed in the web browser.

Flask provides a `render_template()` helper function that allows the use of the Jinja template engine. This will make managing HTML much easier by writing your HTML code in `.html` files, in addition to using logic in your HTML code. You will use these HTML files, (templates), to create your web application.

In this step, you will create your main Flask application in a new file.

First, in your project directory, use your favorite text editor to create and edit your `app.py` file. Previously, you have been using the `app.py` to write the code for your final model. To avoid confusion, now you will use a `model.py` or a `titanic.py` for that, and the `app.py` will exclusively be to build your web app. This will host all the code you will use to create the application. 

In this new file, you will import the Flask object to create a Flask application instance, as you did before. You will also import the `render_template()` helper function that allows you to render HTML template files that exist in the templates folder you are about to create. The file will have a single view function that will be responsible for handling requests to the main `/` path. Add the following content:

```py
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
```

The `index()` view function returns the result of invoking `render_template()` with `index.html` as an argument; this instructs `render_template()` to look for a file named `index.html` in the `templates` folder. The folder and file do not exist yet, and you will receive an error if you run the application at this point. You are going to run it anyway, so that you are familiar with this commonly encountered exception. You will then resolve it by creating the necessary folder and file.

Save the file and close it.

Stop the development server on your other terminal running the hello application with `Ctrl + c`.

Before running the application, be sure to correctly specify the value for the FLASK_APP environment variable, since you are not using the hello application now.

```bash
export FLASK_APP=app
flask run
```

When you open the URL `http://127.0.0.1:5000` in your browser, the debugger page will be displayed, informing you that the `index.html` template was not found. The main line of code in the code responsible for this error will be highlighted. In this case, it is the line `return render_template('index.html')`.

If you click on this line, the debugger will reveal more code so that you have more context to help you resolve the problem.

You will probably see an error showing **template not found (index.html)**. 

Let's create folder templates. In your application, you will use templates to render HTML, which will display in the user's browser. This folder contains our HTML form file `index.html`. Start editing your `index.html` file by writing the following code:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ML app</title>
</head>
<body>
   <h1>Welcome to my Titanic Survival prediction app</h1>
</body>
</html>
```

Save the file and use your browser to navigate to `http://127.0.0.1:5000` again, or refresh the page. This time, the browser should display the text **Welcome to my Titanic Survival prediction app** in an **h1** tag.

In addition to the templates folder, Flask web applications also typically have a static folder to house files, such as the CSS files, JavaScript files, and images that the application uses.

You can create a `style.css` stylesheet file to add CSS to your application. First, create a directory called `static` inside your main project directory. Then create another directory called `css` inside `static` to host the `.css` files. The same can be done for js files and images for more complex apps.

Inside your `css` directory, create a `style.css` file and add the following rule:

```css
h1 {
    border: 2px #eee solid;
    color: brown;
    text-align: center;
    padding: 10px;
}
```

This code will add a border, change the color to brown, center the text and add a small padding to the **h1** tags.

Save and close the file.

In your `index.html` file you will add a link to your `style.css` file:

```html
<head>
    <meta charset="UTF-8">
    <link rel="stylesheet" href="{{ url_for('static', filename= 'css/style.css') }}">
    <title>Welcome to my Titanic Survival prediction app</title>
</head>
```

Here, you use the helper function `url_for()` to generate the appropriate file location. The first argument specifies that you are linking to a static file, and the second argument is the path to the file within the static directory.

Save and close the file.

After updating your application's index page, you will notice that the **Welcome to my Titanic Survival prediction app** text is now brown, centered, and framed within a border.

You can put the style you want to your `style.css` file. However, the Bootstrap tool kit can help you with this if you are not an expert. Now, if your application will have more than one page, then you can avoid unnecessary repetition of code with the help of a base template file, from which all your HTML files will inherit. If that is the case, you can write the following code in your `base.html` file:

```html
<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

    <title>{% block title %} {% endblock %}</title>
  </head>
  <body>
    <nav class="navbar navbar-expand-md navbar-light bg-light">
        <a class="navbar-brand" href="{{ url_for('index')}}">FlaskBlog</a>
        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav">
            <li class="nav-item active">
                <a class="nav-link" href="#">About</a>
            </li>
            </ul>
        </div>
    </nav>
    <div class="container">
        {% block content %} {% endblock %}
    </div>

    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
  </body>
</html>
```

Save and close the file once you have finished editing it.

Most of the code in the block above is standard HTML and code required for Bootstrap. The `<meta>` tags provide information for the web browser, the `<link>` tag links to Bootstrap CSS files, and the `<script>` tags are links to JavaScript code that enables some additional Bootstrap functionality.

However, the following highlighted parts are specific to the Jinja template engine:

- `{% block title %} {% endblock %}`: a block that serves as a placeholder for a title. You will later use it in other templates to give a custom title to each page of your application without having to rewrite the entire `<head>` section each time.

- `{{ url_for('index')}}`: a function invocation that will return the URL for the `index()` view function. This is different from the previous `url_for()` invocation you used to link to a static CSS file because it only requires one argument, which is the name of the view function, and links to the path associated with the function rather than to a static file.

- `{% block content %} {% endblock %}`: another block that will be replaced by content, depending on the secondary template (templates that inherit from `base.html`) that will override it.

Now that you have a `base.html` template, you can inherit that code into `index.html` by only adding the following code in your `index.html`:

```html
{% extends 'base.html' %}

{% block content %}
    <h1>{% block title %} Welcome to FlaskBlog {% endblock %}</h1>
{% endblock %}
```

You have used HTML templates and static files in Flask in a clean way. However, to make things simple for your first web app, we will keep only the `index.html` file.

Let's see how we should code a form requesting the attributes of our passengers.

> In order to predict the data correctly, the corresponding values of each label should match with the value of each input selected.

In the Titanic form that you saw at the beginning of this lesson, we were only requesting the numerical features for prediction, but in the case where we include categorical features that were previously label encoded, we need to put the same values in the HTML form. The following example shows how the form should be coded in case our 'Sex' feature would have been assigned 0 for Male and 1 for Female:

```html
<label for="Sex">Gender</label>
    <select id="relation" name="relation">
      <option value="0">Male</option>
      <option value="1">Female</option>
    </select>
```

You can find a couple of form examples in the following link:

https://www.geeksforgeeks.org/html-design-form/


### Step 4: Predicting the survival result

Let's run the application.

```bash
export FLASK_APP=app.py
run flask
```

When someone submits the form, the webpage should display the result if a passenger would survive or die in the Titanic. For this, we require the model file (`model.pkl`) we created before, in the same project folder. We add the following code to the `app.py` file:

```py
# Prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        
        if int(result)==1:
            prediction='Passenger survives'
        else:
            prediction='Passenger dies'
            
        return render_template("result.html",prediction=prediction) 
```

Here, after the form is submitted, the form values are stored in the variable `to_predict_list` in the form of a dictionary. We convert it into a list of the dictionary's values and pass it as an argument to the `ValuePredictor()` function. In this function, we load the `model.pkl` file, predict the new values, and return the result.

This result/prediction (whether the passenger survives or not) is then passed as an argument to the template engine with the HTML page to be displayed.

Create the following `result.html` file and add it to the templates folder.

```html
<!doctype html>
<html>
   <body>
       <h1>{{ prediction }}</h1>
   </body>
</html>
```

**An alternative code for the entire `app.py` file could be:**

```py
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')  # http://www.google.com/
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

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Would you survive? {} (1=survived, 0=deceased)'.format(output))

if __name__=="__main__":
    app.run(port=5000, debug=True)
```

Run the application again and it should predict the result after submitting the form. We have successfully created the Web application. Now it's time to use Heroku to deploy it.

## Deployment using Heroku

You should already have an account on Heroku, but if you don't, go ahead and create your account at https://www.heroku.com.

Let's make sure we also have the following before deploying to Heroku:

1. Gunicorn handles requests and takes care of complicated things. Download `gunicorn` to your virtual environment. You can use `pip` to download it.

```bash
pip install gunicorn
```

2. We have installed a lot of libraries and other important files like flask, gunicorn, sklearn etc. We need to tell Heroku that our project requires all these libraries to successfully run the application. This is done by creating a `requirements.txt` file.


3. `Procfile` is a text file in the root directory of your application to explicitly declare what command should be executed to start your app. This is an essential requirement for Heroku. This file tells Heroku we want to use the web process with the command gunicorn and the app name.

```bash
web: gunicorn app:app
```

Your current structure should be looking something like this:

![Current structure of repository](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-heroku-structure.jpg?raw=true)

4. Finally, use a `.gitignore` file to exclude unnecessary files that we don't want to deploy to Heroku.

We are ready! Push your project to Heroku! If you wish to do it directly on Heroku's website, you can do it as follows:

- Click on 'Create a new app'

- On 'deploy' tab: link Heroku app to your GitHub account and select the repo to connect to.

- Scroll down and choose 'manual deploy'. After making sure you are on the branch you want to deploy (in this case: `main`), click on 'Deploy branch'. You will see all the required packages have been installed, as shown in the following image:

![Deploying branch](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/deploying_branch.jpg?raw=true)

- When finished, it should look like the following screenshot:

![Deploy finished](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/deployed_to_heroku.jpg?raw=true)

- Copy that link and paste it in your browser to test your app.

**If you feel more comfortable with the command line, you will need to have git and Heroku CLI installed, and then follow this steps:**

> You can click on the following link to install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli 

```bash
heroku login
```

```bash
heroku create
```

```bash
git init
git add .
git commit -m 'initial commit'
```

```bash
git push heroku master
heroku open
```
        
Go ahead and test your web app!

Source:

https://www.heroku.com/

https://devcenter.heroku.com/articles/heroku-cli

https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3-es

https://medium.com/towards-data-science/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b

https://medium.com/towards-data-science/create-an-api-to-deploy-machine-learning-models-using-flask-and-heroku-67a011800c50

https://medium.com/towards-data-science/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

https://medium.com/towards-data-science/flask-and-heroku-for-online-machine-learning-deployment-425beb54a274
