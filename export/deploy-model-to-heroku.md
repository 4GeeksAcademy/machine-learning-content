# Steps of cloud and API deployment of a Machine learning model on Heroku

(under development)

## Structure of your folder

*structure image*

**Saving model to disk**

In your modeling file use the following code to save your model:

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

**Using a simple form with HTML and CSS to gather values and make predictions**

*put code of a simple form here*

**Create a new web app on Heroku**

You should already have an account on Heroku, but if you don't. go ahead and create your account at 'https://www.heroku.com'.

- Click on 'Create a new app'

- On 'deploy' tab: link Heroku app to your Github account and select the repo to connect to.

- Scroll down and choose 'manual deploy'. After making sure you are on the branch you want to deploy (in this case: main), then clic on 'Deploy branch'. You will see all the required packages been installed like the following image:

*include image*

- When finished, it should look like the following screenshot:

*include image*

- Copy that link and paste it in your browser to test your app.



Source:

https://medium.com/towards-data-science/considerations-for-deploying-machine-learning-models-in-production-

https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3-es
