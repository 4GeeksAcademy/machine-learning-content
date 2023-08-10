## Deployment in Render using Flask

After the model development phase, we will have a model that meets our expectations and satisfies our needs. For this model to be useful and fulfill the function for which it has been trained, we must make it available in an environment that allows us to use it. Here we propose a free environment called `Render`, but it can be transferred to other environments, free or paid.

### Render

Render is a cloud computing platform that facilitates the deployment, hosting and execution of applications, databases, scheduled tasks and other services. It is often described as an easy-to-use platform that combines the ease of platforms like Heroku with the power and flexibility of more traditional cloud providers like AWS.

Some key features and offerings of Render include:

1. **Web application deployment**: Render allows you to deploy web applications in various languages and frameworks, including Node.js, Ruby on Rails, Django and many others.
2. **Private services**: These are applications or jobs that are not exposed to the Internet but can be used by other applications in Render.
3. **Scheduled tasks**: Allows to execute periodic jobs, similar to cron jobs in Unix systems.
4. **Databases**: Render supports the deployment of databases such as PostgreSQL, and offers a persistent storage solution for data.
5. **Deployment from repositories**: You can connect your GitHub or GitLab repository and configure automatic deployments every time you push to your repository.

Render has earned a positive reputation for being an attractive option for developers and startups looking for a quick and easy way to deploy and scale applications without the administrative overhead of more traditional solutions.

#### Registration on the platform

In order to access Render you must have an account. To register you must access the following [link](https://dashboard.render.com/register). Once you have an account, you will have access to all the Render functionality:

![render-functionalities](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/render-functionalities.PNG?raw=true)

We can create database services, web deployment services, scheduled tasks...

### Integration in Render

In this lesson we will integrate the classification model we have developed in the [decision trees module](https://4geeks.com/es/syllabus/spain-ds-pt-1/read/exploring-decision-trees).

The `decision_tree_classifier_default_42.sav` model has been saved in a `Pickle` object so that it can be used, for example, to deploy it in a web service like this case.

#### Step 1: Create a Git repository

To integrate something into Render we must first have created a Git repository. The Git we are going to generate in this lesson can be found [here](https://github.com/4GeeksAcademy/flask-render-integration), which is derived from 4Geeks' Machine Learning Template.

#### Step 2: Create a basic application

We will now generate a simple application using the `Flask` library. In the `src` directory, we create a new file named `hello.py` which we will modify with the following code:

```py
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"
```

The created file will serve as a minimal example of how to handle HTTP requests. It imports the `Flask` object and creates a function that returns an HTTP response.

Right now the repository looks like this:

![flask-step1](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step1.png?raw=true)

#### Step 3: Run the application

To run the application locally we need the Python library `gunicorn`. We just need to install it, access with the console to the directory where the script is located and run `gunicorn app:app`.

![flask-step2](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step2.png?raw=true)

When finished, an address will be available through which we can access the web application:

![flask-step21](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step21.png?raw=true)

In this case, as we are developing it in a Codespace, the link is different from the one that would be generated locally, which would be `http://127.0.0.1:8000`.

Now we have implemented a very simple web application using Flask. In addition, we have been able to run it and display information in the web interface.

Now you have a small Flask web application. You have run your application and displayed information in the web browser. Next, we will add HTML files to customize the application.

#### Step 4: Implementing the application web interface

As we mentioned at the beginning of the lesson, we want to integrate the decision tree trained for the Iris dataset from the Machine Learning UCI repository. This dataset has 4 predictor variables: petal width (`petal width (cm)`), petal length (`petal length (cm)`), sepal width (`sepal width (cm)`) and sepal length (`sepal length (cm)`).

We will create an HTML that allows us to enter a value for each variable in order to carry out the prediction:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Iris - Model prediction</title>
</head>
<body>
    <h2>Introduce the values</h2>
    
    <form action="/" method="post">
        Petal width: <input type="number" step="any" name="val1" required><br><br>
        Petal length: <input type="number" step="any" name="val2" required><br><br>
        Sepal width: <input type="number" step="any" name="val3" required><br><br>
        Sepal length: <input type="number" step="any" name="val4" required><br><br>
        <input type="submit" value="Predict">
    </form>
    
    {% if prediction != None %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
```

This HTML contains a title and a form in which the values associated with each field must be entered. Then, by clicking on the `Predict` button, an element containing the prediction of the model will appear, depending on the values entered. In the HTML there are some sentences between braces that are pure Python code, a curious syntax used by Flask to enter values dynamically.

All the HTML templates that we generate must go in a `templates` folder that must be created at the same level as the `app.py`. We call this file `index.html` and store it in the folder.

In addition to creating the above template, we must update the code so that it is fed from the HTML, receives the fields and can return a prediction. Thus, the `app.py` file would be updated:

```py
from flask import Flask, request, render_template
from pickle import load
app = Flask(__name__)
model = load(open("/workspaces/flask-render-integration/models/decision_tree_classifier_default_42.sav","rb"))
class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        
        val1 = float(request.form['val1'])
        val2 = float(request.form['val2'])
        val3 = float(request.form['val3'])
        val4 = float(request.form['val4'])
        
        data = [[val1, val2, val3, val4]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None
    
    return render_template("index.html", prediction = pred_class)
```

We have created the `index` function, which replaces the old `hello_world` and is fed by the values entered in the HTML to trigger the prediction process. This is because when the `Predict` button is clicked, a POST request is sent to the script and the values entered in the HTML form are read to perform the prediction.

Ultimately, the method returns the rendered HTML, in this case with the value of the prediction based on the values.

Right now the repository looks like this:

![flask-step3](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step3.png?raw=true)

If we save the changes and run the application again (`gunicorn app:app`), after navigating to our local web application we will see the following:

![flask-step4](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step4.png?raw=true)

After filling in the values and clicking on `Predict`, the result is also displayed in the interface itself:

![flask-step5](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step5.png?raw=true)

Entering any value predicts a class. Moreover, the effectiveness of the model is as observed in the past module.

The web interface seems very simple and unattractive to users. The next step is to give it some styling.

#### Step 5: Styling the application web interface

An easy way to add styles is to use CSS. We can add a `<style>` block directly to the above HTML to enhance it visually. The `CSS` code we will include is as follows:

```css
body {
    font-family: Arial, sans-serif;
    margin: 40px;
    background-color: #f4f4f4;
}
form {
    background-color: #fff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
}
input[type="number"] {
    width: 100%;
    padding: 10px;
    margin: 10px 0;
    border-radius: 4px;
    border: 1px solid #ccc;
}
input[type="submit"] {
    background-color: #333;
    color: #fff;
    padding: 10px 15px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
input[type="submit"]:hover {
    background-color: #555;
}
h3 {
    margin-top: 20px;
    background-color: #fff;
    padding: 10px;
    border-radius: 4px;
}
```

The above code sets a light background for the entire page, and highlights the form and header with a white background and smoothly rounded edges. The input fields are more spacious and visual, with appropriate borders and padding, and the submit button features a color change when hovered over, providing visual feedback. In addition, more legible typography is used and elements are appropriately spaced with margins to prevent them from feeling cramped.

When inserted into the HTML, the code would look like this:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Iris - Model prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f4f4f4;
        }
        form {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
        }
        input[type="number"] {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            border: 1px solid #ccc;
        }
        input[type="submit"] {
            background-color: #333;
            color: #fff;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #555;
        }
        h3 {
            margin-top: 20px;
            background-color: #fff;
            padding: 10px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h2>Introduce the values</h2>
    
    <form action="/" method="post">
        Petal width: <input type="number" step="any" name="val1" required><br><br>
        Petal length: <input type="number" step="any" name="val2" required><br><br>
        Sepal width: <input type="number" step="any" name="val3" required><br><br>
        Sepal length: <input type="number" step="any" name="val4" required><br><br>
        <input type="submit" value="Predict">
    </form>
    
    {% if prediction != None %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
```

After re-running the application and accessing the web interface again, this is its new appearance:

![flask-step6](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step6.png?raw=true)

And again, when filling in the values and launching the prediction, this is how it is displayed on the front end:

![flask-step7](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step7.png?raw=true)

After developing the desired functionality and having a front end that meets our needs, we will integrate all this into Render.

#### Step 6: Create service in Render and deploy the application

The last step is to configure the service in Render and connect it to our Git repository. We must go to the Render Dashboard, select the `Web Services` section and choose the repository where we have uploaded all the code and the previous folders.

Once we select it, a form like the following one will appear:

![flask-step8](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step8.png?raw=true)

We will have to fill it with the following information:

- `Name`: The name we want our service to have. In this case we will introduce `4geeks-flask-integration`.
- `Branch`: The branch where our updated code is located, always in the latest version. We will have to leave the default value, `master`.
- `Root Directory`: In this case we have developed the code inside the `src` folder, which includes the Python script, the HTML template and the project libraries (file `requirements.txt`), so we should enter `src`.
- `Runtime`: The code is Python, so we will leave the default value, `Python 3`.
- `Build Command`: We will leave the default value, `pip install -r requirements.txt`.
- `Start Command`: We are already friendly with this command. We have used in the development gunicorn, so we will leave the default value, `gunicorn app:app`.

Finally, we will choose the free rate. The form, once filled in, should have the following information:

![flask-step9](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step9.png?raw=true)

In the next step we will see a console with the logs of the application deployment. The deployment is done step by step, first cloning the repository, building it (*build*), installing the dependencies, and, finally, executing the command to launch the web application.

![flask-step10](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step10.png?raw=true)

##### Resolve creation bug

Because the Render environment is different from our development environment (especially in the Python version, since 3.7 is used by default and in this bootcamp we use 3.10 and up), we may get an error in the build of the project. In this case its resolution is very simple:

![flask-step11](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step11.png?raw=true)

We have to access, in the same screen where the execution log is opened, to the `Environment` section and enter a new environment variable. In this case we have the `3.11.4` version of Python but you could enter any other (as long as it is from 3.7).

We re-launch the deployment and now it should work.

***

Once the deployment has been successful, this is the log that will be displayed:

![flask-step12](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step12.png?raw=true)

In fact, a section is available in which we can visualize the different deployments of our web application and the status of each one of them:

![flask-step13](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step13.png?raw=true)

#### Step 7: Using the service in Render

Once the deployment has been successful, we access the application from the link just below the name of the service, and we can now use the application and share it with our friends/colleagues/clients. The one we have created in this lesson is accessible at the following link: `https://fourgeeks-flask-integration.onrender.com/`.

![flask-step14](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step14.png?raw=true)