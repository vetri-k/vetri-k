---
title: "House Price Prediction"
excerpt: "Predict house prices using advanced regression techniques <br/><img src='/images/HousingPrice/Ames_Iowa.png'>"
collection: portfolio
---

# Introduction

This project was inspired by interest in implementation of machine learning techniques. If there are opportunities for improvement, please let me know.
The Ames Housing dataset was compiled by Dean De Cock and was obtained from Kaggle. Goal of the project is to use Python, data manipulation skills, feature engineering skills and advanced regression techniques to predict sales price for each house.

Dataset consists of 79 explanatory variables (independent variables) and sale prices (dependent variable) describing every aspect of residential homes in Ames, Iowa and 2919 records. Sales price was provided for 1460 records which is the training set and sales price must be predicted for 1459 records which is the test set.
This project will cover data exploration, data analysis, feature engineering, model fitting, model prediction and model evaluation.

# Data fields and description

<br/><img src='/images/HousingPrice/data_des.jpg'

```Python
# Loading neccesary library packages

import numpy as np # array manipulation
import pandas as pd # dataframe manipulation
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
from datetime import datetime # data and time manipulation

from scipy import stats #?
from scipy.stats import skew, boxcox_normmax, norm #?
from scipy.special import boxcox1p #?

import matplotlib.gridspec as gridspec #?
from matplotlib.ticker import MaxNLocator #?

from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor

import warnings
# set the maximum number of rows and columns displayed
pd.options.display.max_columns = 50
pd.options.display.max_rows = 50
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight') # plotting style
```

# Exploratory data analysis

```Python
# Loading train and test datasets
train_filepath="../input/home-data-for-ml-course/train.csv"
train=pd.read_csv(train_filepath)

test_filepath="../input/home-data-for-ml-course/test.csv"
test=pd.read_csv(test_filepath)

# Get size of datasets
print("train dataset size is", train.shape)
print("test dataset size is", test.shape)
```
> train dataset size is (1460, 81)
> test dataset size is (1459, 80)

```Python
# Examine train dataset
train.head()
```

<br/><img src='/images/HousingPrice/out4.jpg'

```Python
# Examine test dataset
test.head()
```

<br/><img src='/images/HousingPrice/out5.jpg'

```Python
# Describe train dataset
train.describe()
```

<br/><img src='/images/HousingPrice/out6.jpg'

```Python
# data frame information: non-null values and data types
train.info()
```

<br/><img src='/images/HousingPrice/out7.jpg'

There are null values which needs to be addressed.

```Python
# Ploting Id column
sns.lineplot(data=train.Id)
sns.lineplot(data=test.Id)
```

<br/><img src='/images/HousingPrice/out10.jpg'

```Python
# Dropping unnecessary column "Id"
train.drop('Id', axis='columns', inplace=True) # If True, do operation inplace and return None.
test.drop('Id', axis='columns', inplace=True)

# Separating target variable and dropping it from the train data
y=train['SalePrice'].reset_index(drop=True)
train_features=train.drop(['SalePrice'], axis=1)
test_features=test

print("Size of target variable", y.shape)
print("Size of train data features", train_features.shape)
print("Size of test data features", test_features.shape)
```

> Size of target variable (1460,)
> Size of train data features (1460, 79)
> Size of test data features (1459, 79)

```Python
# Effect of lot area on house sale price
# Seaborn joint plot: bivariate plot with univariate marginal distributions
sns.jointplot(data=train, x='LotArea', y='SalePrice', height=10, alpha=0.4, color='red', xlim=(-10000,50000), ylim=(-10000,500000))
```
<br/><img src='/images/HousingPrice/out13.jpg'

Distribution is similar to normal distribution with outliers.

```Python
# Heat map to examine numerical correlation and understand linear relationship between features. Top part is dropped since it is symmetric with the part below.

correlation_train = train.corr(method='pearson') # Compute pairwise correlation of columns, excluding NA/null value
mask = np.triu(correlation_train.corr()) # show lower half of the heat map
sns.set(font_scale=1) # setting font size in plot
plt.figure(figsize=(20,20)) # setting figure size
sns.heatmap(correlation_train, # data
            annot=True, # include annotation in plot
            fmt='.1f', # annotation to 1 decimal point
            cmap='coolwarm', # color
            square=True, # If True, set the Axes aspect to "equal" so each cell will be square-shaped.
            mask=mask, # show lower half of the map
            linewidths=1, # line between each squares
            cbar=True # adding color bar on the right side
           )
```

# Insights


