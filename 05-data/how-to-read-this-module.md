## Exploratory data analysis

**Exploratory data analysis** (*EDA*) is the starting and fundamental approach to any data analysis as it aims to understand the main characteristics of a data set before performing more advanced analysis or further modeling.

EDA involves the following:

- **Data visualization**: Using plots such as histogram, boxplots, scatter plots and many others to visualize the distribution of the data, the relationships between variables and any anomalies or peculiarities in the data.
- **Anomaly identification**: Detecting and sometimes dealing with outliers or missing data that could affect further analysis.
- **Hypothesis formulation**: From the scan, analysts can begin to formulate hypotheses that will then be tested in more detailed analysis or modeling.

The main purpose of EDA is to see what the data can tell us beyond the formal modeling task. It serves to ensure that the subsequent modeling or analysis phase is done properly and that any conclusions are based on a correct understanding of the structure of the data.

### Descriptive analysis and EDA

Descriptive analysis and EDA, depending on where they are implemented, may be equivalent, but here we will distinguish them by their main differences:

1. **Descriptive analysis**: Focuses on describing the main characteristics of a data set using descriptive statistics, such as mean, median, range, and so on. Its main objective is to provide a clear and summarized description of the data.
2. **EDA**: Goes a step further, as it focuses on exploring patterns, relationships, anomalies, etc., in the data using more sophisticated graphs and statistics. Its main objective is to understand the structure of the data, relationships between variables and to formulate hypotheses or intuitions for further analysis or modeling.

In the real world, after information capture, one may start with descriptive analysis to get a basic sense of the data set and then proceed to EDA for further exploration. However, in many cases, the term EDA is used to encompass both processes, as the boundary between the two is somewhat fuzzy and depends on each workgroup, company, and so on.

### Machine Learning flow

The ideal Machine Learning flow should contain the following phases:

1. **Problem definition**: A need is identified to be solved using Machine Learning.
2. **Data set acquisition**: Once the problem to be solved has been defined, a data capture process is necessary to solve it. For this, sources such as databases, APIs and data from cameras, sensors, etc. can be used.
3. **Store the information**: The best way to store the information so that it can feed the Machine Learning process is to store it in a database. Flat files should be avoided as they are neither secure nor optimal. Consider including them in a database.
4. **Descriptive analysis**: Raw data stored in a database can be a great and very valuable source of information. Before starting to simplify and exploit them with EDA, we must know their fundamental statistical measures: means, modes, distributions, deviations, and so on. Knowing the distributions of the data is vital to be able to select a model accordingly.
5. **EDA**: This step is vital to ensure that we keep the variables that are strictly necessary and eliminate those that are not relevant or do not provide information. In addition, it allows us to know and analyze the relationships between variables and odd values.
6. **Modeling and optimization**: With the data ready, all that remains is to model the problem. With the conclusions of the two previous steps we must analyze which model best fits the data and train it. Optimizing a model after the first training is absolutely necessary unless it performs without errors.
7. **Deployment**: For the model to be able to bring value to our processes or to our customers, it must be consumable. Deployment is the implementation of the model in a controlled environment where it can be run and used to predict with real world data.