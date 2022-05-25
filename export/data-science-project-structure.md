# Data Science Project Structure

...

Has it ever happened that you are asked to work on an existing project...

...



or after finishing a project, you want other developers to be able to work on yourr project withput conflicts...

...

### Gitpod

... info of why to use Gitpod

...what if i want to create my data science project from scratch?

...

Has it happened to you that every time you want to create a new project you copy the entire folder of a previous project with the same set of code and then start replacing variables, renaming folders and manually changing their code inputs, hoping to not forget anything on the way. This is a pretty tedious and repetitive task. Not to mention that it’s prone to errors. 

Also, as a machine learning engineer you must have heard that every project should have its own environment with specific tools to make sure that every person in your audience is able to install the specific version of each tool and be able to reproduce your code in their own computers. Is this part confusing for you? Maybe you do not completely understand yet how to create an environment, how to do it in the command line? If this is you, then you will love the tool the two tools we will show you in this lesson.

### Anaconda

The first tool we want to introduce in this lesson is Anaconda. Anaconda is an amazing collection of scientific Python packages, tools, resources, and IDEs. This package includes many important tools that a Data Scientist can use to harness the incredible force of Python. Anaconda individual edition is free and open source. It runs on Windows, macOS and Linux. Anaconda is great for deep models and neural networks. You can build models, deploy them, and integrate with leading technologies in the subject. Anaconda is optimized to run efficiently for machine learning tasks and will save you time when developing great algorithms. Anaconda contains Conda and Anaconda Navigator, an intuitive platform, where you can create, load, and switch between environments very easily. You will also be able to search and install open-source data science and machine learning packages, including pandas, numpy, matplotlib, seaborn, as well as Jupyter Notebooks and VS Code. You will be able to run the Anaconda Prompt terminal or the Anaconda Powershell terminal with your current environment. 

To get started using Anaconda, you can download it directly from the Anaconda website: https://www.anaconda.com/ . After downloading and installation, you will be able to find in your computer the Anaconda Prompt and the Anaconda Navigator.

Take some time to discover Anaconda. You can take a look at this task guides  in order to be able to create your project environment and install specific libraries for your project. Depending on your preference you can use the Anaconda Prompt or the Anaconda Navigator. 

**Using the Anaconda Prompt:**

https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


**Using the Anaconda Navigator:**

https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/#:~:text=the%20Remove%20button.-,Advanced%20environment%20management,access%20additional%2C%20advanced%20management%20features.

### Cookiecutter

Our second powerful tool is cookiecutter! This is an incredible way to create a project template for a type of analysis that you know you will need to repeat a number of times, while inputting the necessary data and/or parameters just once.

What is cookiecutter?

Projects can be python packages, web applications, machine learning apps with complex workflows or anything you can think of
Templates are what cookiecutter uses to create projects. What cookiecutter does is quite simple: it clones a directory and put it inside your new project. It then replaces all the names that are between {{ and }} (Jinja2 syntax) with names that it finds in the cookiecutter.json file. The best part is that it also has a specific template for data science and machine learning projects. (We’ll see an example of how to build a cookiecutter template)

As you already learned how to create a new environment, Cookiecutter must be part of your environment if you want to use it. If you use Anaconda, type conda list in your terminal and see if it shows up in the list of installed packages, otherwise let's get started using cookiecutter. 

You can install it with pip:

```py
pip install cookiecutter
```
\
or via Anaconda:

```py
conda config --add channels conda-forge
conda install cookiecutter
```
\
Now you can use cookiecutter to create new templates for projects and papers!

So now that you have installed it, what is the next step to have an organized data science project structure?

To start a new project, type:

```py
cookiecutter https://github.com/drivendata/cookiecutter-data-science
```

*For reference, we'll leave the documentation for cookiecutter-data-science in the following link:*

https://drivendata.github.io/cookiecutter-data-science/

### Create a cookiecutter template to kickstart Streamlit project

Streamlit is a Python library designed to build web applications. It’s very simple to use and provides a lot of functionalities that let you share experiments and results with your team and prototype machine learning apps.

Let's look at this Streamlit normal project structure from a machine learning engineer:

-*src* folder that contains :

    -the main script of the app (app.py ) 
    
    -utils module that contains two scripts: 

        -ui.py to put the layout functions

        -common.py to hold other utility functions for data processing or remote database connections (among other things)

-.gitignore file to prevent git from versioning unnecessary files (such as .env files, or .pyc files)

-Procfile and setup.sh : to handle the deployment on Heroku

-requirements.txt : to list the project dependencies

-.env file to store the environment variables of the project

-README.md to share details about the project

Now let's see how to implement the cookiecutter data science template on it:

Source: 

https://www.gitpod.io/docs/getting-started

https://medium.com/@lukaskf/why-you-should-consider-using-cloud-development-environments-a79c062a2798

https://www.anaconda.com/

https://towardsdatascience.com/an-overview-of-the-anaconda-distribution-9479ff1859e6

https://cookiecutter-data-science-vc.readthedocs.io/en/latest/getting_started/INSTALL.html

https://github.com/drivendata/cookiecutter-data-science/issues/17

https://drivendata.github.io/cookiecutter-data-science/

https://github.com/drivendata/cookiecutter-data-science
