---
title: Matplotlib for Data Science
tags:
  - data visualization
  - matplotlib
  - python
  - data science
description: >-
  Learn the essentials of Matplotlib and master data visualization in Python.
  Discover how to create, customize, and analyze visual data step-by-step!
---

# Matplotlib for Data Science

Data visualization is one of the most powerful tools in Data Science. It allows us to explore, analyze, and communicate information effectively. In this article, we will learn how to use Matplotlib, one of the most popular Python libraries for creating graphs and visually representing data.

## Fundamentals of Matplotlib

### What is Matplotlib?

Matplotlib is a Python visualization library that allows you to create a wide variety of graphs, from simple to highly customized. Its main module is pyplot, which provides a MATLAB-like interface for generating visualizations with ease.

#### Installing Matplotlib

If you don't have the library installed yet, you can do so with the following command:

```bash
pip install matplotlib
```

After installation, you are ready to start plotting.

### Difference between pyplot and figure/axes

`Matplotlib` offers two main approaches to generating graphs:

- **pyplot:** An easy-to-use interface that allows you to quickly create graphs, ideal for beginners.

- **Figure/Axes (Object-Oriented Approach):** Provides greater control over graph customization.

Let's see how to use pyplot to create basic graphs. To start, we import the main library:

```python
import matplotlib.pyplot as plt
```

## Create a Basic Line Graph

Let's see how we can generate a line graph with Matplotlib:

```python
import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.linspace(0, 10, 100)  # Generate 100 points between 0 and 10
y = np.sin(x)  # Calculate the sine of each point

# Create the figure
plt.plot(x, y, label='Sine of x', color='b', linestyle='--')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Graph of a Sine Function')
plt.legend()
plt.show()
```
This code generates a graph of the sine function, showing labeled axes, a title, and a legend, as shown in the following image.

![image1](/assets/plot_sine.png)

## Subplots (Multiple Graphs in One Figure)

Often, we need to display multiple graphs in the same figure. For this, we use `subplot` or `subplots`.

### Using `plt.subplot()`

This method allows you to create subplots in a grid defined by rows and columns:

```python
plt.subplot(2, 1, 1)  # (rows, columns, index)
plt.plot(x, np.sin(x), 'r')
plt.title('Sine')

plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x), 'b')
plt.title('Cosine')

plt.tight_layout()  # Adjust the spaces between subplots
plt.show()
```

### Using `plt.subplots()`

This method provides greater flexibility and control over the subplots:

```python
fig, ax = plt.subplots(2, 1, figsize=(6, 6))  # 2 rows, 1 column
ax[0].plot(x, np.sin(x), 'r')
ax[0].set_title('Sine')
ax[1].plot(x, np.cos(x), 'b')
ax[1].set_title('Cosine')
plt.tight_layout()
plt.show()
```

![image2](/assets/subgraphic-plot.png)

## Object-Oriented Approach

The object-oriented approach of Matplotlib provides greater control over visualizations.

### Difference with pyplot

Instead of using `plt.plot()`, the `Figure` and `Axes` based approach allows for more precise customization of graphs.

Example:

```python
fig, ax = plt.subplots()
ax.plot(x, y, color='g', linestyle='-.', linewidth=2)
ax.set_title('Example with Object-Oriented Approach')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.grid(True)
plt.show()
```

![image3](/assets/plot-oo2.png)

## Working with Real Data

Matplotlib is very useful for visualizing data from files like CSV. We can combine it with pandas to load and plot real data.

### Load and visualize data from a CSV

```python
import pandas as pd

df = pd.read_csv('data.csv')  # Load a CSV file

plt.plot(df['Date'], df['Sales'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales over Time')
plt.xticks(rotation=45)
plt.show()
```

### Integration with Pandas

We can use `df.plot()` instead of `plt.plot()` for quick visualizations:

```python
df.plot(x='Date', y='Sales', kind='line')
plt.title('Graph generated with Pandas')
plt.show()
```

