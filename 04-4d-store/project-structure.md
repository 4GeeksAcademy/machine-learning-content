# Machine Learning Coding Project Structure

Almost no one starts starts a project from scratch in a professional team, it's a common practice to innitialize your project by using pre-defined files and structure (a template) that contains all the recomendations and initial setups needed to avoid past mistakes, using the most popular libraries and compatible with one of the major cloud platforms for later deployment.

A serious project template must contain some if not all of the following features:

1. Enviroment variable isolation to store sensible information.
2. Git ready.
3. A package manager for dependency management like PIP, Poetry, Pipenv, Conda, etc.
4. Readme file and well made documentation.
5. Compatibility with one of the major cloud platforms for continious deployment.
6. A `src` folder that will contain the programing code created by the developers.

Additionaly, in the case of maching learning or data-science we want our boilerplate to have a data folder for storing datasets in static file formats like CSV, JSON, etc.



## Project Structure

### The README file

**What is it?**

A README is a text file that introduces and explains a project. It contains information that is commonly required to understand what the project is about.

**Why should I make it?**

It's an easy way to answer questions that your audience will likely have regarding how to install and use your project and also how to collaborate with you.

**Who should make it?**

Anyone who is working on a programming project, especially if you want others to use it or contribute.

**When should I make it?**

Definitely before you show a project to other people or make it public. You might want to get into the habit of making it the first file you create in a new project.

**Where should I put it?**

In the top level directory of the project. This is where someone who is new to your project will start out. Code hosting services such as GitHub, Bitbucket, and GitLab will also look for your README and display it along with the list of files and directories in your project.

**How should I make it?**

While READMEs can be written in any text file format, the most common one that is used nowadays is Markdown. It allows you to add some lightweight formatting. You can learn more about it at the CommonMark website, which also has a helpful reference guide and an interactive tutorial.

Some other formats that you might see are plain text, reStructuredText (common in Python projects), and Textile.

You can use any text editor. There are plugins for many editors (e.g. Atom, Emacs, Sublime Text, Vim, and Visual Studio Code) that allow you to preview Markdown while you are editing it.

You can also use a dedicated Markdown editor like Typora or an online one like StackEdit or Dillinger. You can even use the editable template below.

### The requirements.txt file

**What are dependencies?**

Dependencies are external Python packages that your own project relies onto, in order to do the job is intended to. Let’s consider a Python project that makes use of pandas DataFrames. In this case, this project has a dependency on pandas package since it cannot work properly without pre-installing pandas. Every dependency may also have other dependencies. Therefore, dependency management can sometimes get quite tricky or challenging and needs to be handled properly. Our Python project, may have a dependency on a specific version of a third-party package, and we may also end up with dependency conflicts because a dependency needed a specific version of a package.

The most common way for handling dependencies and instructing package management tools about what specific versions we need in our own project is through a requirements text file.

The requirements.txt is a file listing all the dependencies for a specific Python project. It may also contain dependencies of dependencies, as discussed previously. The listed entries can be pinned or non-pinned. If a pin is used, then you can specify a specific package version (using ==), an upper or lower bound or even both.

```bash
matplotlib>=2.2
numpy>=1.15.0, <1.21.0
pandas
pytest==4.0.1
```

Finally, you could install these dependencies (normally in a virtual environment) through pip using the following command:

```bash
pip install -r requirements.txt
```

Once you install all the dependencies, you can see the precise version of each dependency installed in the virtual environment by running pip freeze. This command will list all the packages along with their specific pins (i.e. ==).

### The .gitignore file

You can configure Git to ignore files you don't want to check in to GitHub. You can create a .gitignore file in your repository's root directory to tell Git which files and directories to ignore when you make a commit. To share the ignore rules with other users who clone the repository, commit the .gitignore file in to your repository.

For more information about the .gitignore file read here: https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files

### The .env file

In our day to day work, we have all come across the need to hide specific things from our app before sharing them on a public version control system, like Git.

You could be dealing with your SQL database names and URLs, their passwords, some secret keys associated with AWS or Google Cloud Platform user groups, API secrets, the list goes on.

For this, we are going to learn the concept of Environment variables.

A preferred way to use environment variables in data science projects is using a .env file, which can be stored within your app. 
Make a new .env file and set all your variables within once, and save.

Example: 

```py
DB_USER = 'xxxxxxxxxxxxx'
DB_PASSWORD = 'xxxxxxxxxxxxxxx'
DB_PORT = 3306
DB_HOST = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
DB_NAME = 'xxxxxxxxxxxxx'
```

>Don’t forget to add this filename in your .gitignore file, in order to protect it from your all-knowing version control systems.

**How do I access them?**

If you have created the .env file we can access our environment variables stored there by using the dotenv package.

Run the following command inside your virtual environment to install the package:

```bash
pip install python-dotenv
```

Next, you can access the variables from your .env file  with:

```py
from dotenv import load_dotenv

load_dotenv()  # it reads the environment variables from .env file
```

Next, you can easily use the os.environ.get() function to get the variables back through their names.

### The DATA folder

**RAW** : The original, immutable data.

**INTERIM**: Intermediate data that has been transformed.

**PROCESSED**: The final, canonical data sets for modeling.

### The SRC folder

**The Utils.py file**: Utils is a collection of small Python functions and classes which make common patterns shorter and easier.

**The exploration notebook (explore.ipynb)**

This is a jupyter notebook meant for data exploration. If you are using VSCode, you need to install the Jupyter extension, in order to create notebook.ipynb files.

Notebooks are useful because they allow all sorts of data science tasks including data cleaning and transformation, numerical simulation, exploratory data analysis, data visualization, statistical modeling, machine learning, deep learning, and much more.

Jupyter Notebooks allows for cell by cell execution of code blocks which is advantageous because it allows for convenient testing of blocks of code, especially when we don't know the data yet and we need to try different methods to improve results.

**The Python script file (app.py)**

Notebooks are great for exploration, however we need to become engineers, and be able to build software, so once we have a better understanding of our data and have decided between different ways of cleaning it, we can create our data preprocessing pipeline in our app.py file. The same happens with the modeling process. After having experiment with different machine learning algorithms and parameters, we can code our final training model in our app.py file.

This will make it much easier for real time models, and for shoing results to stakeholders.

Source:

https://towardsdatascience.com/introducing-jupytext-9234fdff6c57

https://towardsdatascience.com/why-synchronize-your-jupyter-notebook-with-a-python-py-file-bbf35baf02ee

https://www.makeareadme.com/

https://towardsdatascience.com/requirements-vs-setuptools-python-ae3ee66e28af#:~:text=txt%20file-,The%20requirements.,be%20pinned%20or%20non%2Dpinned.

https://drivendata.github.io/cookiecutter-data-science/


