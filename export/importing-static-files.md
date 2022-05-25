# Importing files into Python

### Importing flat files with Numpy


```python
import numpy as np
 
# Assign filename to variable: file
file = 'digits.csv'
 
# Load file as array: digits
digits = np.loadtxt(file, delimiter=',')
 
# Print datatype of digits
print(type(digits))
```

There are a number of arguments that np.loadtxt() takes that you'll find useful to change:

- delimiter changes the delimiter that loadtxt() is expecting
- skiprows allows you to specify how many rows (not indices) you wish to skip (for example if you don't want to include the header)
- usecols takes a list of the indices of the columns you wish to keep

If we have a file that includes a header consisting of strings and we try to import it as it is using np.load_txt(), python would throw us a ValueError saying that it could not convert string to float. We have two ways to solve this:

1. Set the data type argument dtype equal to str (for string).
2. skip the first row using the skiprows argument.

What happens if we have different datatypes in different columns?

The function np.loadtxt() will freak at this, but there is another function, np.genfromtxt(), which can handle such structures. By writing dtype=None to it, it will figure out what types each column should be. The parameter names=True indicates that there is a header.

There is also another function np.recfromcsv() that behaves similarly to np.genfromtxt(), except that its default dtype is None. 

### Importing flat files as dataframes with Pandas

**CSV FILES**

We can easily import files of mixed data types as DataFrames using the pandas functions read_csv() and read_table().


```python
import pandas as pd
 
# Assign the filename to a variable
file = 'titanic.csv'
 
# Read the file into a DataFrame variable
df = pd.read_csv(file)
 
# View the first rows of the DataFrame
print(df.head())
```

It is also possible to retrieve the corresponding numpy array using the attribute values.


```python
# Build a numpy array from the DataFrame: data_array
data_array = np.array(data.values)
```

Sometimes we will find purselves dealing with corrupted files thay may include comments, missing values, etc.
We can load those corrupted files with Pandas as follows:


```python
# Import file: data
data = pd.read_csv(file, sep='\t', comment='#', na_values='Nothing')
```

**EXCEL FILES**

At some point, we will also have to deal with Excel files. Given an Excel file imported into a variable, you can retrieve a list of the sheet names using the attribute sheet_names.


```python
import pandas as pd
 
# Assign spreadsheet to a file variable
file = 'battledeath.xlsx'
 
# Load spreadsheet: excel
excel_file = pd.ExcelFile(file)
 
# Print sheet names
print(excel_file.sheet_names)

```

We will learn how to import any given sheet of our loaded .xslx file as a DataFrame. We'll be able to do so by specifying either the sheet's name or its index.


```python
# Load the sheet '2004' into a DataFrame df1
df1 = excel_file.parse('2004')

# Print the head of the DataFrame df1
print(df1.head())

# Load a sheet into a DataFrame by index: df2
df2 = excel_file.parse(0)

# Print the head of the DataFrame df2
print(df2.head())
```

We have used the method parse(). However, we can add additional arguments like skiprows, names and parse_cols. These arguments skip rows, name the columns and designate which columns to parse, respectively. All these arguments can be assigned to lists containing the specific row numbers, strings and column numbers, as appropriate.


```python
# Parse the first column of the second sheet and rename the column: df2

df2 = excel_file.parse(1, parse_cols=[0], skiprows=[0], names=['City'])
```

**SAS FILES**

We will learn how to import a SAS file as a DataFrame using SAS7BDAT and pandas.


```python
import pandas as pd
import matplotlib.pyplot as plt
from sas7bdat import SAS7BDAT

# Save file to a DataFrame: df_sas
with SAS7BDAT('examplefile.sas7bdat') as file:
    df_sas = file.to_data_frame()
 
print(df_sas.head())
```

**STATA FILES**

How to import a Stata file as DataFrame using the pd.read_stata() function from pandas:


```python
import pandas as pd
import matplotlib.pyplot as plt
 
# Load Stata file into a pandas DataFrame: df
df = pd.read_stata('examplefile.dta')

# Plot histogram of one column of the DataFrame
pd.DataFrame.hist(df[['column1']])
```

**HDF5 FILES**


```python
import numpy as np
import h5py
 
file = 'examplefile.hdf5'
 
# Load file: data
data = h5py.File(file, 'r')
```

**MATLAB FILES**

In the case of matlab files we will use scipy.


```python
import scipy.io
 
mat = scipy.io.loadmat('examplefile.mat')
```
