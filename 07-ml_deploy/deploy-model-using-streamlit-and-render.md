## Deployment in Render using Streamlit

In the previous module, we learned how to create a Machine Learning web application with `Flask`, an intuitive library that facilitates the generation of dynamic HTML templates to be able to use our models once trained.

Streamlit is another great alternative commonly used to generate applications simply by programming in Python, without the need to have previous knowledge of HTML or CSS.

We will create an application that we will integrate back into Render.

### Integration in Render

We will again use the classification model that we have developed in the [decision trees module](https://4geeks.com/lesson/exploring-decision-trees).

The `decision_tree_classifier_default_42.sav` model has been saved in a `Pickle` object so that it can be used, for example, to deploy it in a web service like this case.

#### Step 1: Create a repository in Git

To integrate something in Render we must first have to create a Git repository. The repository we are going to generate in this lesson is [here](https://github.com/4GeeksAcademy/streamlit-render-integration), which is derived from 4Geeks' Machine Learning Template.

#### Step 2: Create a basic application

Now we will generate a simple application using the `Flask` library. In the `src` directory, we create a new file named `app.py` that we will modify with the following code:

```py
import streamlit as st

st.title("Hello, World!")
```

Compared to the Flask syntax we saw in the previous module, it is much simpler and more intuitive. In the interface, we can expect an empty window with a title that shows "Hello, World!".

Right now, the repository looks like this:

![Streamlit step 1](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step1.png?raw=true)

#### Step 3: Run the application

To run the application locally, we can use the same library, since it provides a mechanism to run it in a very simple way. We must access with the console to the directory where the script is located and execute `streamlit run app.py`.

![Streamlit step 2](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step2.png?raw=true)

When finished, an address will be available through which we can access the web application:

![Streamlit step 2.1](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step21.png?raw=true)

In this case, as we are developing it in a Codespace, the link is different from the one that would be generated locally, which would be `http://172.16.5.4:8501`.

We can also appreciate that Streamlit provides a base style that is very different from Flask. This is the potential of this tool: with a simple and minimalist implementation, we can get aesthetic, attractive and usable applications.

Next, we will improve the interface in order to be able to use the model through it.

#### Step 4: Implementing the application web interface

As we mentioned at the beginning of the lesson, we want to integrate the decision tree trained for the Iris dataset from the Machine Learning UCI repository. This dataset has 4 predictor variables: petal width (`petal width (cm)`), petal length (`petal length (cm)`), sepal width (`sepal width (cm)`) and sepal length (`sepal length (cm)`).

The implementation of web interfaces in Streamlit is infinitely simpler, not needing to create a `templates` folder or create specific templates for each HTML page, everything is done directly in the same Python script `app.py`:

```py
from pickle import load
import streamlit as st

model = load(open("../models/decision_tree_classifier_default_42.sav", "rb"))
class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

st.title("Iris - Model prediction")

val1 = st.slider("Petal width", min_value = 0.0, max_value = 4.0, step = 0.1)
val2 = st.slider("Petal length", min_value = 0.0, max_value = 4.0, step = 0.1)
val3 = st.slider("Sepal width", min_value = 0.0, max_value = 4.0, step = 0.1)
val4 = st.slider("Sepal length", min_value = 0.0, max_value = 4.0, step = 0.1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)
```

With the above code, we managed to generate a form composed of 4 sliding elements (to make it easier for the user to enter the value) and a button that launches the prediction to the model and displays it on the screen.

Another advantage of Streamlit is that every time the script is saved, the web application is updated, and it is not necessary to restart the execution, as it was the case with gunicorn and Flask.

Once the changes are saved and the interface is updated, we will see the following:

![Streamlit step 3](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step3.png?raw=true)

After filling in the values and clicking on `Predict`, the result is also displayed in the interface itself:

![Streamlit step 4](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step4.png?raw=true)

Entering any value predicts a class. Also, the effectiveness of the model is as observed in the past module.

The web interface would be complete and ready for integration into Render, since it would not be necessary to update its style or layout as it was in Flask. However, the options offered by this library are many. [Guide and examples](https://blog.streamlit.io/designing-streamlit-apps-for-the-user-part-ii/).

#### Step 5: Create service in Render and deploy the application

The last step is to configure the service in Render and connect it with our Git repository. We must go to the Render Dashboard, select the `Web Services` section, and choose the repository where we have uploaded all the code and the previous folders.

Once we select it, a form like the following one will appear:

![Streamlit step 5](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step5.png?raw=true)

We will have to fill it with the following information:

- `Name`: The name we want our service to have. In this case, we will introduce `4geeks-streamlit-integration`.
- `Branch`: The branch where our updated code is located, always in the latest version. We should leave the default value, `main`.
- `Root Directory`: In this case we have developed the code inside the `src` folder, which includes the Python script and the project libraries (file `requirements.txt`), so we should enter `src`.
- `Runtime`: The code is Python, so we will leave the default value, `Python 3`.
- `Build Command`: We will leave the default value, `pip install -r requirements.txt`.
- `Start Command`: Although we could use gunicorn as we did with Flask, Streamlit also has a friendly interface for deploying solutions locally, so we modify the command and replace it with `streamlit run app.py`.

Finally, we will choose the free rate. The form, once filled out, should have the following information:

![Streamlit step 6](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step6.png?raw=true)

In the next step, as in the previous module and whenever you deploy a solution in Render, a console will appear to inform us of the deployment status:

![Streamlit step 7](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step7.png?raw=true)

#### Resolve creation bug

Because the Render environment is different from our development environment (especially in the Python version, since 3.7 is used by default and in this case we use 3.10 and up), we may get an error in the build of the project. In this case, its resolution is very simple:

![Streamlit step 8](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step8.png?raw=true)

We have to access, in the same screen where the execution log is opened, to the `Environment` section and enter a new environment variable. In this case we have the `3.11.4` version of Python but you could enter any other (as long as it is from 3.7).

We re-launch the deployment, and now it should work.

***

Once the deployment has been successful, this is the log that will be displayed:

![Streamlit step 9](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step9.png?raw=true)

#### Step 6: Using the service in Render

Once the deployment has been successful, we access the application from the link just below the name of the service, and we can now use the application and share it with our friends/colleagues/clients. The one we have created in this lesson is accessible at the following link: `https://fourgeeks-streamlit-integration.onrender.com/`.

![Streamlit step 10](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step10.png?raw=true)

> Note: As you have used the free plan, Render may throw the application away if it is not used. Depending on when you read this the application will be operational or not.
