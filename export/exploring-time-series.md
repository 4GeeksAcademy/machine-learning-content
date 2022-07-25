# Exploring Time Series

## Time Series Manipulation using Pandas


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



## Use case:

In the following example we will only take in data from a uni-variate time series. That means we really are only considering the relationship between the y-axis value the x-axis time points. We’re not considering outside factors that may be effecting the time series.

A common mistake beginners make is they immediately start to apply ARIMA forecasting models to data that has many outside factors.


```python
import pandas as pd
data = pd.read_csv('../assets/electric_production.csv', index_col=0)
data.head()
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
      <th>IPG2211A2N</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1939-01-01</th>
      <td>3.3335</td>
    </tr>
    <tr>
      <th>1939-02-01</th>
      <td>3.3590</td>
    </tr>
    <tr>
      <th>1939-03-01</th>
      <td>3.4353</td>
    </tr>
    <tr>
      <th>1939-04-01</th>
      <td>3.4607</td>
    </tr>
    <tr>
      <th>1939-05-01</th>
      <td>3.4607</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.index = pd.to_datetime(data.index)
```


```python
data.columns = ['Energy Production']
```


```python
import chart_studio.plotly
import cufflinks as cf
data.iplot(title="Energy Production Jan 1985--Jan 2018")
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    /workspace/machine-learning-content/06-12d-ml_algos/exploring-time-series.ipynb Cell 24 in <cell line: 3>()
          <a href='vscode-notebook-cell://4geeksacade-machinelear-qokegp1onqy.ws-us54.gitpod.io/workspace/machine-learning-content/06-12d-ml_algos/exploring-time-series.ipynb#ch0000047vscode-remote?line=0'>1</a> import chart_studio.plotly
          <a href='vscode-notebook-cell://4geeksacade-machinelear-qokegp1onqy.ws-us54.gitpod.io/workspace/machine-learning-content/06-12d-ml_algos/exploring-time-series.ipynb#ch0000047vscode-remote?line=1'>2</a> import cufflinks as cf
    ----> <a href='vscode-notebook-cell://4geeksacade-machinelear-qokegp1onqy.ws-us54.gitpod.io/workspace/machine-learning-content/06-12d-ml_algos/exploring-time-series.ipynb#ch0000047vscode-remote?line=2'>3</a> data.iplot(title="Energy Production Jan 1985--Jan 2018")


    File ~/.pyenv/versions/3.8.13/lib/python3.8/site-packages/cufflinks/plotlytools.py:1216, in _iplot(self, kind, data, layout, filename, sharing, title, xTitle, yTitle, zTitle, theme, colors, colorscale, fill, width, dash, mode, interpolation, symbol, size, barmode, sortbars, bargap, bargroupgap, bins, histnorm, histfunc, orientation, boxpoints, annotations, keys, bestfit, bestfit_colors, mean, mean_colors, categories, x, y, z, text, gridcolor, zerolinecolor, margin, labels, values, secondary_y, secondary_y_title, subplots, shape, error_x, error_y, error_type, locations, lon, lat, asFrame, asDates, asFigure, asImage, dimensions, asPlot, asUrl, online, **kwargs)
       1214 	return Figure(figure)
       1215 else:
    -> 1216 	return iplot(figure,validate=validate,sharing=sharing,filename=filename,
       1217 		 online=online,asImage=asImage,asUrl=asUrl,asPlot=asPlot,
       1218 		 dimensions=dimensions,display_image=kwargs.get('display_image',True))


    File ~/.pyenv/versions/3.8.13/lib/python3.8/site-packages/cufflinks/plotlytools.py:1470, in iplot(figure, validate, sharing, filename, online, asImage, asUrl, asPlot, dimensions, display_image, **kwargs)
       1468 	return offline.py_offline.iplot(figure, validate=validate, filename=filename, show_link=show_link, link_text=link_text, config=config)
       1469 else:		
    -> 1470 	return py.iplot(figure,validate=validate,sharing=sharing,
       1471 					filename=filename)


    File ~/.pyenv/versions/3.8.13/lib/python3.8/site-packages/chart_studio/plotly/plotly.py:135, in iplot(figure_or_data, **plot_options)
        133 if "auto_open" not in plot_options:
        134     plot_options["auto_open"] = False
    --> 135 url = plot(figure_or_data, **plot_options)
        137 if isinstance(figure_or_data, dict):
        138     layout = figure_or_data.get("layout", {})


    File ~/.pyenv/versions/3.8.13/lib/python3.8/site-packages/chart_studio/plotly/plotly.py:276, in plot(figure_or_data, validate, **plot_options)
        273 else:
        274     grid_filename = filename + "_grid"
    --> 276 grid_ops.upload(
        277     grid=grid,
        278     filename=grid_filename,
        279     world_readable=payload["world_readable"],
        280     auto_open=False,
        281 )
        283 _set_grid_column_references(figure, grid)
        284 payload["figure"] = figure


    File ~/.pyenv/versions/3.8.13/lib/python3.8/site-packages/chart_studio/plotly/plotly.py:1087, in grid_ops.upload(cls, grid, filename, world_readable, auto_open, meta)
       1084     if parent_path:
       1085         payload["parent_path"] = parent_path
    -> 1087 file_info = _create_or_overwrite_grid(payload)
       1089 cols = file_info["cols"]
       1090 fid = file_info["fid"]


    File ~/.pyenv/versions/3.8.13/lib/python3.8/site-packages/chart_studio/plotly/plotly.py:1528, in _create_or_overwrite_grid(data, max_retries)
       1526 if filename:
       1527     try:
    -> 1528         lookup_res = v2.files.lookup(filename)
       1529         if isinstance(lookup_res.content, bytes):
       1530             content = lookup_res.content.decode("utf-8")


    File ~/.pyenv/versions/3.8.13/lib/python3.8/site-packages/chart_studio/api/v2/files.py:85, in lookup(path, parent, user, exists)
         83 url = build_url(RESOURCE, route="lookup")
         84 params = make_params(path=path, parent=parent, user=user, exists=exists)
    ---> 85 return request("get", url, params=params)


    File ~/.pyenv/versions/3.8.13/lib/python3.8/site-packages/retrying.py:49, in retry.<locals>.wrap.<locals>.wrapped_f(*args, **kw)
         47 @six.wraps(f)
         48 def wrapped_f(*args, **kw):
    ---> 49     return Retrying(*dargs, **dkw).call(f, *args, **kw)


    File ~/.pyenv/versions/3.8.13/lib/python3.8/site-packages/retrying.py:220, in Retrying.call(self, fn, *args, **kwargs)
        218         jitter = random.random() * self._wait_jitter_max
        219         sleep = sleep + max(0, jitter)
    --> 220     time.sleep(sleep / 1000.0)
        222 attempt_number += 1


    KeyboardInterrupt: 



```python
from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data, model="multiplicative")
fig = result.plot()
plot_mpl(fig)
```


```python
from pyramid.arima import auto_arima
stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())
```


```python
train = data.loc['1985-01-01':'2016-12-01']
test = data.loc['2017-01-01':]
```


```python
stepwise_model.fit(train)
```


```python
future_forecast = stepwise_model.predict(n_periods=37)
```


```python
future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=[‘Prediction’])
pd.concat([test,future_forecast],axis=1).iplot()
```


```python
pd.concat([data,future_forecast],axis=1).iplot()
```

Source: 

https://towardsdatascience.com/how-to-forecast-time-series-with-multiple-seasonalities-23c77152347e

https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c

https://towardsdatascience.com/time-series-analysis-in-python-an-introduction-70d5a5b1d52a

https://github.com/WillKoehrsen/Data-Analysis/tree/master/additive_models

https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b

https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea
