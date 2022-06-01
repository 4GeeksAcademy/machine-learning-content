# ML Project structure Activity

We have seen how to work with cookicutter on a common streamlit project. But Cookiecutter has a lot of templates for different project types, so as you can imagine, Data Science is not the exception.

Use the Cookiecutter Data Science open-source template to kickstart data science projects that follow the best standards in the industry.

To start a new data science new project, you can open the terminal in your project folder and type:

```py
cookiecutter https://github.com/drivendata/cookiecutter-data-science
```

That's it!! Now you can see an entire structure of very organized, readable and reusable structure. Take some time to explore how is it structured.

**Build from the environment up**

The first step in reproducing an analysis is always reproducing the computational environment it was run in. You need the same tools, the same libraries, and the same versions to make everything play nicely together.

One effective approach to this is use virtualenv. By listing all of your requirements in the repository (a requirements.txt file is included) you can easily track the packages needed to recreate the analysis. Here is a good workflow:

1. Run mkvirtualenv when creating a new project

2. pip install the packages that your analysis needs

3. Run pip freeze > requirements.txt to pin the exact package versions used to recreate the analysis

4. If you find you need to install another package, run pip freeze > requirements.txt again and commit the changes to version control.
 
Any person wanting to use your project will just need to write the following command in order to have your same dependencies:

```py
pip install -r requirements.txt
```
It is a good practice to execute the previous command every time you are about to start editing a project.

*For reference, we'll leave the documentation for cookiecutter-data-science in the following link:*

https://drivendata.github.io/cookiecutter-data-science/

### Getting ready!


Let's get ready for the next module where we will start exploring and cleaning a dataset. 

1. Create a folder named 'predicting-titanic-survival' for your template.

2. Install cookicutter and clone the cookiecutter data science template.

3. Make sure your project is also in your Github account (try using the same project name).

4. Look for the Titanic dataset (train and test datasets) in the following url: https://www.kaggle.com/competitions/titanic/data and make sure to put it in your 'raw data folder'.

5. In the .gitignore file write data/ to include the data folder so we avoid uploading the big datasets every time we commit to Github.
In your Readme file write the following: 'Titanic dataset can be found in the following url: https://www.kaggle.com/competitions/titanic/data '.

6. Share the link of your new Github project in Slack.
