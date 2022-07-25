# Exploring Time Series

## Time Series Manipulation using Pandas

In the following example we will only take in data from a uni-variate time series. That means we really are only considering the relationship between the y-axis value the x-axis time points. We’re not considering outside factors that may be effecting the time series.

A common mistake beginners make is they immediately start to apply ARIMA forecasting models to data that has many outside factors.


```python
# Creating a date range with hourly frequency

import pandas as pd
from datetime import datetime
import numpy as np
date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='H')
```


```python
date_rng
```




    DatetimeIndex(['2018-01-01 00:00:00', '2018-01-01 01:00:00',
                   '2018-01-01 02:00:00', '2018-01-01 03:00:00',
                   '2018-01-01 04:00:00', '2018-01-01 05:00:00',
                   '2018-01-01 06:00:00', '2018-01-01 07:00:00',
                   '2018-01-01 08:00:00', '2018-01-01 09:00:00',
                   ...
                   '2018-01-07 15:00:00', '2018-01-07 16:00:00',
                   '2018-01-07 17:00:00', '2018-01-07 18:00:00',
                   '2018-01-07 19:00:00', '2018-01-07 20:00:00',
                   '2018-01-07 21:00:00', '2018-01-07 22:00:00',
                   '2018-01-07 23:00:00', '2018-01-08 00:00:00'],
                  dtype='datetime64[ns]', length=169, freq='H')




```python
type(date_rng[0])
```




    pandas._libs.tslibs.timestamps.Timestamp



Now let´s create an example dataframe with the timestamp data we just created


```python
df = pd.DataFrame(date_rng, columns=['date'])
df['data'] = np.random.randint(0,100,size=(len(date_rng)))

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01 00:00:00</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-01 01:00:00</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-01 02:00:00</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-01 03:00:00</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-01 04:00:00</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>



If we want to do time series manipulation, we’ll need to have a date time index so that our data frame is indexed on the timestamp.


```python
#Convert the dataframe index to a datetime index 

df['datetime'] = pd.to_datetime(df['date'])
df = df.set_index('datetime')
df.drop(['date'], axis=1, inplace=True)
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01 00:00:00</th>
      <td>23</td>
    </tr>
    <tr>
      <th>2018-01-01 01:00:00</th>
      <td>73</td>
    </tr>
    <tr>
      <th>2018-01-01 02:00:00</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2018-01-01 03:00:00</th>
      <td>10</td>
    </tr>
    <tr>
      <th>2018-01-01 04:00:00</th>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filter data with only day 2.

df[df.index.day == 2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-02 00:00:00</th>
      <td>27</td>
    </tr>
    <tr>
      <th>2018-01-02 01:00:00</th>
      <td>80</td>
    </tr>
    <tr>
      <th>2018-01-02 02:00:00</th>
      <td>9</td>
    </tr>
    <tr>
      <th>2018-01-02 03:00:00</th>
      <td>67</td>
    </tr>
    <tr>
      <th>2018-01-02 04:00:00</th>
      <td>26</td>
    </tr>
    <tr>
      <th>2018-01-02 05:00:00</th>
      <td>13</td>
    </tr>
    <tr>
      <th>2018-01-02 06:00:00</th>
      <td>22</td>
    </tr>
    <tr>
      <th>2018-01-02 07:00:00</th>
      <td>68</td>
    </tr>
    <tr>
      <th>2018-01-02 08:00:00</th>
      <td>48</td>
    </tr>
    <tr>
      <th>2018-01-02 09:00:00</th>
      <td>75</td>
    </tr>
    <tr>
      <th>2018-01-02 10:00:00</th>
      <td>93</td>
    </tr>
    <tr>
      <th>2018-01-02 11:00:00</th>
      <td>80</td>
    </tr>
    <tr>
      <th>2018-01-02 12:00:00</th>
      <td>36</td>
    </tr>
    <tr>
      <th>2018-01-02 13:00:00</th>
      <td>67</td>
    </tr>
    <tr>
      <th>2018-01-02 14:00:00</th>
      <td>54</td>
    </tr>
    <tr>
      <th>2018-01-02 15:00:00</th>
      <td>66</td>
    </tr>
    <tr>
      <th>2018-01-02 16:00:00</th>
      <td>54</td>
    </tr>
    <tr>
      <th>2018-01-02 17:00:00</th>
      <td>9</td>
    </tr>
    <tr>
      <th>2018-01-02 18:00:00</th>
      <td>38</td>
    </tr>
    <tr>
      <th>2018-01-02 19:00:00</th>
      <td>36</td>
    </tr>
    <tr>
      <th>2018-01-02 20:00:00</th>
      <td>19</td>
    </tr>
    <tr>
      <th>2018-01-02 21:00:00</th>
      <td>19</td>
    </tr>
    <tr>
      <th>2018-01-02 22:00:00</th>
      <td>30</td>
    </tr>
    <tr>
      <th>2018-01-02 23:00:00</th>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filtering data between two dates

df['2018-01-04':'2018-01-06']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-04 00:00:00</th>
      <td>61</td>
    </tr>
    <tr>
      <th>2018-01-04 01:00:00</th>
      <td>62</td>
    </tr>
    <tr>
      <th>2018-01-04 02:00:00</th>
      <td>24</td>
    </tr>
    <tr>
      <th>2018-01-04 03:00:00</th>
      <td>76</td>
    </tr>
    <tr>
      <th>2018-01-04 04:00:00</th>
      <td>41</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-01-06 19:00:00</th>
      <td>44</td>
    </tr>
    <tr>
      <th>2018-01-06 20:00:00</th>
      <td>14</td>
    </tr>
    <tr>
      <th>2018-01-06 21:00:00</th>
      <td>45</td>
    </tr>
    <tr>
      <th>2018-01-06 22:00:00</th>
      <td>45</td>
    </tr>
    <tr>
      <th>2018-01-06 23:00:00</th>
      <td>63</td>
    </tr>
  </tbody>
</table>
<p>72 rows × 1 columns</p>
</div>



We could take the min, max, average, sum, etc., of the data at a daily frequency instead of an hourly frequency as per the example below where we compute the daily average of the data:


```python
df.resample('D').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>45.083333</td>
    </tr>
    <tr>
      <th>2018-01-02</th>
      <td>46.833333</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>48.333333</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>52.625000</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>47.791667</td>
    </tr>
    <tr>
      <th>2018-01-06</th>
      <td>37.333333</td>
    </tr>
    <tr>
      <th>2018-01-07</th>
      <td>45.666667</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>68.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['rolling_sum'] = df.rolling(3).sum()
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>rolling_sum</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01 00:00:00</th>
      <td>23</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-01-01 01:00:00</th>
      <td>73</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-01-01 02:00:00</th>
      <td>3</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>2018-01-01 03:00:00</th>
      <td>10</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>2018-01-01 04:00:00</th>
      <td>44</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>2018-01-01 05:00:00</th>
      <td>9</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>2018-01-01 06:00:00</th>
      <td>73</td>
      <td>126.0</td>
    </tr>
    <tr>
      <th>2018-01-01 07:00:00</th>
      <td>83</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>2018-01-01 08:00:00</th>
      <td>4</td>
      <td>160.0</td>
    </tr>
    <tr>
      <th>2018-01-01 09:00:00</th>
      <td>95</td>
      <td>182.0</td>
    </tr>
  </tbody>
</table>
</div>



It only starts having valid values when there are three periods over which to look back.

This is a good chance to see how we can do forward or backfilling of data when working with missing data values.


```python
df['rolling_sum_backfilled'] = df['rolling_sum'].fillna(method='backfill')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>rolling_sum</th>
      <th>rolling_sum_backfilled</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01 00:00:00</th>
      <td>23</td>
      <td>NaN</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>2018-01-01 01:00:00</th>
      <td>73</td>
      <td>NaN</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>2018-01-01 02:00:00</th>
      <td>3</td>
      <td>99.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>2018-01-01 03:00:00</th>
      <td>10</td>
      <td>86.0</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>2018-01-01 04:00:00</th>
      <td>44</td>
      <td>57.0</td>
      <td>57.0</td>
    </tr>
  </tbody>
</table>
</div>



It’s often useful to be able to fill your missing data with realistic values such as the average of a time period, but always remember that if you are working with a time series problem and want your data to be realistic, you should not do a backfill of your data.

When working with time series data, you may come across time values that are in Unix time. Unix time, also called Epoch time is the number of seconds that have elapsed since 00:00:00 Coordinated Universal Time (UTC), Thursday, 1 January 1970.

**How to convert epoch time to real time?**



```python
epoch_t = 1529272655
real_t = pd.to_datetime(epoch_t, unit='s')
real_t
```




    Timestamp('2018-06-17 21:57:35')




```python
# Now, let's convert it to Pacific time

real_t.tz_localize('UTC').tz_convert('US/Pacific')
```




    Timestamp('2018-06-17 14:57:35-0700', tz='US/Pacific')



Example code to handling missing values in time-series data:

```py
df['data'].isnull().sum()
df['data'] = df['data'].fillna(df['data'].bfill())

Source: 

https://towardsdatascience.com/how-to-forecast-time-series-with-multiple-seasonalities-23c77152347e

https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c

https://towardsdatascience.com/time-series-analysis-in-python-an-introduction-70d5a5b1d52a

https://github.com/WillKoehrsen/Data-Analysis/tree/master/additive_models

https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b

https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea
