---
title: "House Price Prediction"
classes: "wide"
excerpt: "Predict house prices using advanced regression techniques <br/><img src='/images/HousingPrice/Ames_Iowa.png'>"
collection: portfolio
---

# Introduction

This project was inspired by interest in implementation of machine learning techniques. If there are opportunities for improvement, please let me know.
The Ames Housing dataset was compiled by Dean De Cock and was obtained from Kaggle. Goal of the project is to use Python, data manipulation skills, feature engineering skills and advanced regression techniques to predict sales price for each house.

Dataset consists of 79 explanatory variables (independent variables) and sale prices (dependent variable) describing every aspect of residential homes in Ames, Iowa and 2919 records. Sales price was provided for 1460 records which is the training set and sales price must be predicted for 1459 records which is the test set.
This project will cover data exploration, data analysis, feature engineering, model fitting, model prediction and model evaluation.

# Data fields and description

<img src='/images/HousingPrice/data_des.jpg'>

```python
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

```python
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

```python
# Examine train dataset
train.head()
```

<img src='/images/HousingPrice/out4.jpg'>

```python
# Examine test dataset
test.head()
```

<img src='/images/HousingPrice/out5.jpg'>

```python
# Describe train dataset
train.describe()
```

<img src='/images/HousingPrice/out6.jpg'>

```python
# data frame information: non-null values and data types
train.info()
```

<img src='/images/HousingPrice/out7.jpg'>

There are null values which needs to be addressed.

```python
# Ploting Id column
sns.lineplot(data=train.Id)
sns.lineplot(data=test.Id)
```

<img src='/images/HousingPrice/out10.png'>

```python
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

```python
# Effect of lot area on house sale price
# Seaborn joint plot: bivariate plot with univariate marginal distributions
sns.jointplot(data=train, x='LotArea', y='SalePrice', height=10, alpha=0.4, color='red', xlim=(-10000,50000), ylim=(-10000,500000))
```

<img src='/images/HousingPrice/out13.png'>

Distribution is similar to normal distribution with outliers.

```python
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

<img src='/images/HousingPrice/out17.png'>

# Observations

- Strong correlation is observed between sale price and overallqual (overall material and finish) of the house, GrLivArea (ground living area).
- Sale price is also affected on different levels by
  - year built (YearBuilt)
  - masonry veneer area
  - basement area
  - first floor area
  - garage area
  - number of baths and rooms
- Dwelling type (MSSubClass) and overall condition (OverallCond) not that dependent on the sale price.
- There are correlation between number of rooms and area, garage size and area.

```python
# Merging train and test data before editing to reduce data manipulation work
features=pd.concat([train_features, test_features]).reset_index(drop=True) # resets the index to the default integer index
print(features.shape)
```

> (2919, 79)

```python
# Frequency counts in a column, value_counts(): Frequency counts
features['MSZoning'].value_counts(dropna=False) #Return a Series containing counts of unique values
```
> RL         2265
> RM          460
> FV          139
> RH           26
> C (all)      25
> NaN           4
> Name: MSZoning, dtype: int64

# Identify missing data

```python
# Detect missing values and Visualize

total_missing=features.isnull().sum() # checking for 'NaN' values and counting them
total_missing_nonzero=total_missing[total_missing!=0] # drop rows with zero, dfnew=df[df!=0]
percent=(total_missing/len(features))*(100)

plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.barplot(x=total_missing.index, y=percent)
```

<img src='/images/HousingPrice/out22.png'>

```python
# Identify if the missing data is categorical or numeric
pd.concat([total_missing, train.dtypes], axis=1)
```

<img src='/images/HousingPrice/out23.jpg'>

# Addressing missing data

```python
# List of columns with 'NaN' where NaN's equals none
none_cols = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
             'GarageType','GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

# List of columns with 'NaN' where NaN's equals 0
zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
    'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']

# List of columns with 'NaN' where NaN's (missing) replaced with mode
freq_cols = ['Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'Functional', 'KitchenQual',
    'SaleType']

# Filling the list of columns above with appropriate values:
for col in none_cols:
    features[col] = features[col].fillna('None')

for col in zero_cols:
    features[col] = features[col].fillna(0) # Replace NaN Values with Zeros in Pandas DataFrame

for col in freq_cols:
    features[col] = features[col].fillna(features[col].mode()[0])

total_missing=features.isnull().sum() # checking for 'NaN' values and counting them
total_missing
```

<img src='/images/HousingPrice/out26.jpg'>

```python
# Examine MSZoning categorical data based on MSSubClass
plt.figure(figsize=(15,10))
sns.boxplot(x=features['MSSubClass'], y=features['MSZoning'], whis=np.inf)
sns.stripplot(x=features['MSSubClass'], y=features['MSZoning'], color='0.3')
```

<img src='/images/HousingPrice/out27.png'>

```python
# Examine LotFrontage categorical data based on Neighborhood
plt.figure(figsize=(15,10))
sns.boxplot(x=features['Neighborhood'], y=features['LotFrontage'], whis=np.inf)
sns.stripplot(x=features['Neighborhood'], y=features['LotFrontage'], color='0.3') # scatterplot where one variable is categorical
plt.xticks(rotation=90)
plt.show()
```

<img src='/images/HousingPrice/out28.png'>

```python
# Filling 'MSZoning' (categorical data) according to MSSubClass mode.
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))
# call .groupby() and pass the name of the column that needs to group on, which is "MSSubClass".
# Then, use ["MSZoning"] to specify the columns on which actual aggregation needs to be performed.

# Filling 'LotFrontage' (numerical data) according to Neighborhood median.
features['LotFrontage'] = features.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

# Changing numerical features to category.
features['MSSubClass'] = features['MSSubClass'].astype(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)
```

# Feature engineering

```python
features.nunique()
# features['Condition1']
```

<img src='/images/HousingPrice/out30.jpg'>

```python
# Transforming rare values (less than 10) into one group.

others = ['Condition1', 'Condition2', 'RoofMatl', 'Exterior1st', 'Exterior2nd','Heating', 'Electrical', 'Functional', 'SaleType']

for col in others:
    mask = features[col].isin(features[col].value_counts()[features[col].value_counts() < 10].index)
    features[col][mask] = 'Other'
```

# Categorical data

```python
# Displaying categorical data

def srt_box(y, df):
    
    fig, axes = plt.subplots(14, 3, figsize=(25, 80)) # Create a figure and a set of subplots
    axes = axes.flatten()
# simultaneously iterate through each column of data and through each of our axes making a plot for each step along the way.
# using the axes.flatten() method, don’t have to go through the hastle of nested for loops to deal with a variable number of rows and columns in our figure.
    
    for i, j in zip(df.select_dtypes(include=['object']).columns, axes):
        sortd = df.groupby([i])[y].median().sort_values(ascending=False) # Group within each column is sorted based on median to see influence of sale price
        sns.boxplot(x=i,
                    y=y,
                    data=df,
                    palette='plasma',
                    order=sortd.index, # within each plot, box plots are ordered based on sortd
                    ax=j) # plots arranged based on axes 
        j.tick_params(labelrotation=45) # tick label rotation
        j.yaxis.set_major_locator(MaxNLocator(nbins=18)) # Tick locators define the position of the ticks

        plt.tight_layout() # Automatically adjust subplot parameters to give specified padding

# sale prices vs catergorical data
srt_box('SalePrice', train)
```

<img src='/images/HousingPrice/out33.png>'

# Categorical data observations

MSZoning: Floating village houses has the highest median value. Residental low density houses has outliers. Residental high and medium are similar and commercial is the least.

LandContour: Hillside houses are expensive than the rest.

Neighborhood: Expensive houese are in Northridge Heights, Northridge and Timberland. Outliers are observed in Gilbert, Northwest Ames, North Ames, Edwards and Old Town. Low price houses are observed in Briardale, Iowa DOT and Rail Road and Meadow Village.

MasVnrType: stone masonry veneer type are higher than others.

CentralAir: Higher sale price are observed on houses with central air conditioning .

GarageType: Houses with Builtin garage are expensive. Houses with carports are cheaper.

```python
# Changing some of categorical variables to numeric to quantify them

neigh_map = {
    'MeadowV': 1,
    'IDOTRR': 1,
    'BrDale': 1,
    'BrkSide': 2,
    'OldTown': 2,
    'Edwards': 2,
    'Sawyer': 3,
    'Blueste': 3,
    'SWISU': 3,
    'NPkVill': 3,
    'NAmes': 3,
    'Mitchel': 4,
    'SawyerW': 5,
    'NWAmes': 5,
    'Gilbert': 5,
    'Blmngtn': 5,
    'CollgCr': 5,
    'ClearCr': 6,
    'Crawfor': 6,
    'Veenker': 7,
    'Somerst': 7,
    'Timber': 8,
    'StoneBr': 9,
    'NridgHt': 10,
    'NoRidge': 10
}
features['Neighborhood'] = features['Neighborhood'].map(neigh_map).astype('int')
# Map values of Series according to input correspondence.

ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['ExterQual'] = features['ExterQual'].map(ext_map).astype('int')
features['ExterCond'] = features['ExterCond'].map(ext_map).astype('int')

bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['BsmtQual'] = features['BsmtQual'].map(bsm_map).astype('int')
features['BsmtCond'] = features['BsmtCond'].map(bsm_map).astype('int')

bsmf_map = {
    'None': 0,
    'Unf': 1,
    'LwQ': 2,
    'Rec': 3,
    'BLQ': 4,
    'ALQ': 5,
    'GLQ': 6
}
features['BsmtFinType1'] = features['BsmtFinType1'].map(bsmf_map).astype('int')
features['BsmtFinType2'] = features['BsmtFinType2'].map(bsmf_map).astype('int')

heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
features['HeatingQC'] = features['HeatingQC'].map(heat_map).astype('int')
features['KitchenQual'] = features['KitchenQual'].map(heat_map).astype('int')
features['FireplaceQu'] = features['FireplaceQu'].map(bsm_map).astype('int')
features['GarageCond'] = features['GarageCond'].map(bsm_map).astype('int')
features['GarageQual'] = features['GarageQual'].map(bsm_map).astype('int')
```

# Numerical data

```python
# Scatter plots to see how numerical data effect sale prices is scatter plots. Plotting polynomial regression lines to see general trend and to spot outliers.

# Plotting numerical features with polynomial order to detect outliers.

def srt_reg(y, df):
    fig, axes = plt.subplots(12, 3, figsize=(25, 80))
    axes = axes.flatten()

    for i, j in zip(df.select_dtypes(include=['number']).columns, axes):

        sns.regplot(x=i,
                    y=y,
                    data=df,
                    ax=j,
                    order=3,
                    ci=None,
                    color='#e74c3c',
                    line_kws={'color': 'black'},
                    scatter_kws={'alpha':0.4})
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=10))

        plt.tight_layout()
srt_reg('SalePrice', train)

<img src='/images/HousingPrice/out35.png'>

# Numerical data observations

OverallQual: Sale price of the house increases with overall quality.

OverallCondition: Houses are around 5/10 condition and negatively skewed.

YearBuilt: New houses are expensive compared to old ones.

GrLivArea and GarageArea: Sale price increases with increase in ground floor area and garage area.

MoSold and YrSold: Month and year of sale does not have a effect on house sale price.

# Outliers

```python
# Dropping outliers based on plots

features = features.join(y)
features = features.drop(features[(features['OverallQual'] < 5) & (features['SalePrice'] > 200000)].index)
# .index, the index (row labels) of the DataFrame. Finds the index where both conditions are satisfied and those indexs are dropped from features
features = features.drop(features[(features['GrLivArea'] > 4000) & (features['SalePrice'] < 200000)].index)
features = features.drop(features[(features['GarageArea'] > 1200) & (features['SalePrice'] < 200000)].index)
features = features.drop(features[(features['TotalBsmtSF'] > 3000) & (features['SalePrice'] > 320000)].index)
features = features.drop(features[(features['1stFlrSF'] < 3000) & (features['SalePrice'] > 600000)].index)
features = features.drop(features[(features['1stFlrSF'] > 3000) & (features['SalePrice'] < 200000)].index)

y = features['SalePrice']
y.dropna(inplace=True)
features.drop(columns='SalePrice', inplace=True)
```

# Creating new features
```python
# Combining related features based on observations
features['TotalSF'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] + features['1stFlrSF'] + features['2ndFlrSF'])
features['TotalBathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) + features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))
features['TotalPorchSF'] = (features['OpenPorchSF'] + features['3SsnPorch'] + features['EnclosedPorch'] + features['ScreenPorch'] + features['WoodDeckSF'])
features['YearBlRm'] = (features['YearBuilt'] + features['YearRemodAdd'])

# Combining quality features
features['TotalExtQual'] = (features['ExterQual'] + features['ExterCond'])
features['TotalBsmQual'] = (features['BsmtQual'] + features['BsmtCond'] + features['BsmtFinType1'] + features['BsmtFinType2'])
features['TotalGrgQual'] = (features['GarageQual'] + features['GarageCond'])
features['TotalQual'] = features['OverallQual'] + features['TotalExtQual'] + features['TotalBsmQual'] + features['TotalGrgQual'] + features['KitchenQual'] + features['HeatingQC']

# Creating new features based on quality features
features['QualGr'] = features['TotalQual'] * features['GrLivArea']
features['QualBsm'] = features['TotalBsmQual'] * (features['BsmtFinSF1'] +features['BsmtFinSF2'])
features['QualPorch'] = features['TotalExtQual'] * features['TotalPorchSF']
features['QualExt'] = features['TotalExtQual'] * features['MasVnrArea']
features['QualGrg'] = features['TotalGrgQual'] * features['GarageArea']
features['QlLivArea'] = (features['GrLivArea'] - features['LowQualFinSF']) * (features['TotalQual'])
features['QualSFNg'] = features['QualGr'] * features['Neighborhood']

# Observing the effects of new features on sale price.
def srt_reg(feature):
    merged = features.join(y)
    fig, axes = plt.subplots(5, 3, figsize=(25, 40))
    axes = axes.flatten()

    new_features = ['TotalSF', 'TotalBathrooms', 'TotalPorchSF', 'YearBlRm',
        'TotalExtQual', 'TotalBsmQual', 'TotalGrgQual', 'TotalQual', 'QualGr',
        'QualBsm', 'QualPorch', 'QualExt', 'QualGrg', 'QlLivArea', 'QualSFNg']

    for i, j in zip(new_features, axes):

        sns.regplot(x=i,
                    y=feature, # looks for the feature in the "merged" data
                    data=merged,
                    ax=j,
                    order=3,
                    ci=None,
                    color='#e74c3c',
                    line_kws={'color': 'black'},
                    scatter_kws={'alpha':0.4})
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=10))

        plt.tight_layout()

# checking new features
srt_reg('SalePrice')
```

<img src='/images/HousingPrice/out40.png'>

```python
# Creating additional features which are binary based on numeric features 

features['HasPool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['Has2ndFloor'] = features['2ndFlrSF'].apply(lambda x: 1if x > 0 else 0)
features['HasGarage'] = features['QualGrg'].apply(lambda x: 1 if x > 0 else 0)
features['HasBsmt'] = features['QualBsm'].apply(lambda x: 1 if x > 0 else 0)
features['HasFireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
features['HasPorch'] = features['QualPorch'].apply(lambda x: 1 if x > 0 else 0)
```

# Data transformation

```python
# Certain continious variable features does not follow normal distribution
# Box-Cox transforms data closely to normal distribution

# Numerical features which are skewed
skewed = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'LowQualFinSF', 'MiscVal']

# Finding skewness of the numerical features.
skew_features = np.abs(features[skewed].apply(lambda x: skew(x)).sort_values(ascending=False))

# Filtering skewed features.
high_skew = skew_features[skew_features > 0.3]

# Taking indexes of high skew.
skew_index = high_skew.index

# Applying boxcox transformation to fix skewness.
# Box cox transformation is defined as a way to transform non-normal dependent variables in our data to a normal shape to reduce noise and improve prediction.

for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))   

# Features to drop
to_drop = ['Utilities','PoolQC','YrSold','MoSold','ExterQual','BsmtQual','GarageQual','KitchenQual','HeatingQC',]

# Dropping features
features.drop(columns=to_drop, inplace=True)

# Getting dummy variables for categorical data.
features = pd.get_dummies(data=features) # Convert categorical variable into dummy/indicator variables
```

# Data check

```python
# Checking data before modelling 
print(f'Number of missing values: {features.isna().sum().sum()}')
print(features.shape)
features.head()
```

> Number of missing values: 0
> (2908, 226)

<img src='/images/HousingPrice/out44.jpg'>

```python
# Separating train and test data
train = features.iloc[:len(y), :]
test = features.iloc[len(train):, :]

# Check how transformed data correlates with sale price
correlations = train.join(y).corrwith(train.join(y)['SalePrice']).iloc[:-1].to_frame() # Join columns of another DataFrame
correlations['Abs Corr'] = correlations[0].abs()
sorted_correlations = correlations.sort_values('Abs Corr', ascending=False)['Abs Corr']
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(sorted_correlations.to_frame()[sorted_correlations>=.5], cmap='coolwarm', annot=True, vmin=-1, vmax=1, ax=ax);
```

<img src='/images/HousingPrice/out46.png'>

```python
def plot_dist3(df, feature, title):
    
    # Creating a customized chart. and giving in figsize and everything.
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    
    # creating a grid of 3 cols and 3 rows.
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # Customizing the histogram grid.
    ax1 = fig.add_subplot(grid[0, :2])
    
    # Set the title.
    ax1.set_title('Histogram')
    
    # plot the histogram.
    sns.distplot(df.loc[:, feature],
                 hist=True,
                 kde=True,
                 fit=norm,
                 ax=ax1,
                 color='#e74c3c')
    ax1.legend(labels=['Normal', 'Actual'])

    # customizing the QQ_plot.
    ax2 = fig.add_subplot(grid[1, :2])
    
    # Set the title.
    ax2.set_title('Probability Plot')
    
    # Plotting the QQ_Plot.
    stats.probplot(df.loc[:, feature].fillna(np.mean(df.loc[:, feature])),plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor('#e74c3c')
    ax2.get_lines()[0].set_markersize(12.0)

    # Customizing the Box Plot:
    ax3 = fig.add_subplot(grid[:, 2])
    
    # Set title.
    ax3.set_title('Box Plot')
    
    # Plotting the box plot.
    sns.boxplot(df.loc[:, feature], orient="v", ax=ax3, color='#e74c3c')
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=24))

    plt.suptitle(f'{title}', fontsize=24)

# check target value distribution. applying log transformation
# Checking target variable

plot_dist3(train.join(y), 'SalePrice', 'Sale Price Before Log Transformation')
```

<img src='/images/HousingPrice/out48.png'>

```python
# Setting model data.

X = train
X_test = test
yl = np.log1p(y) # log transformation of sale price

plot_dist3(train.join(yl), 'SalePrice', 'Sale Price After Log Transformation')
```

<img src='/images/HousingPrice/out50.png'>

# Modeling

# Cross validation

```python
# Cross validation setup
from sklearn.model_selection import KFold, cross_val_score
kfolds = KFold(n_splits=10, shuffle=True, random_state=42) # Training data is split into 10, data is shuffled before splitting, randomness in each fold
```

# Defining model
```python
# Lasso regression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
lasso_alphas = [5e-5, 1e-4, 5e-4, 1e-3]
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=lasso_alphas, random_state=42, cv=kfolds))
# RobustScaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range)
# The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters

# Ridge regression
from sklearn.linear_model import RidgeCV
ridge_alphas = [13.5, 14, 14.5, 15, 15.5]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kfolds))

# Gradient Boost Regression
from sklearn.ensemble import GradientBoostingRegressor
gradb = GradientBoostingRegressor(n_estimators=6000, learning_rate=0.01,
                                  max_depth=4, max_features='sqrt',
                                  min_samples_leaf=15, min_samples_split=10,
                                  loss='huber', random_state=42)
# xgboost Regression
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=6000,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006, random_state=42)
                       
# Stacking
stackcv = StackingCVRegressor(regressors=(lasso, ridge, gradb), meta_regressor=xgboost, use_features_in_secondary=True)
```

# Model performance
```python
from sklearn.model_selection import cross_validate
def model_check(X, y, estimators, kfolds):
    ''' Function to test multiple estimators '''
    model_table = pd.DataFrame()
    row_index = 0
    for est, label in zip(estimators, labels):
        model_table.loc[row_index, 'Model Name'] = label
        cv_results = cross_validate(est, X, y, cv=kfolds, scoring='neg_root_mean_squared_error', return_train_score=True)
        # cross_validate evaluates metric by cross-validation and record fit/score times
        # 'neg_mean_squared_error’ is a scoring parameter 
        model_table.loc[row_index, 'Train RMSE'] = -cv_results['train_score'].mean()
        model_table.loc[row_index, 'Test RMSE'] = -cv_results['test_score'].mean()
        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()
        row_index += 1
    return model_table

# Setting list of estimators and labels for them:

estimators = [ridge, lasso, gradb, xgboost]
labels = ['Ridge', 'Lasso', 'GradientBoostingRegressor', 'XGBoost']

# Executing cross validation.

models = model_check(X, yl, estimators, kfolds)
display(models.round(decimals=3))
```

<img src='/images/HousingPrice/out66.jpg'>

# Model fitting to training dataset

```python
lasso_fit = lasso.fit(X, yl)
ridge_fit = ridge.fit(X, yl)
gradb_fit = gradb.fit(X, yl)
stackcv_fit = stackcv.fit(X.values, yl.values)
```

# Blending models
```python
y_train = np.expm1(yl)
y_pred = np.expm1((0.3 * lasso_fit.predict(X)) +
          (0.3 * ridge_fit.predict(X)) +
          (0.1 * gradb_fit.predict(X)) +
          (0.3 * stackcv_fit.predict(X.values)))

from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
rmsle = np.sqrt(mean_squared_log_error(y_train, y_pred))
print(rmsle)
```

> 0.0694000779271092

# Prediction
```python
# Inversing and flooring log scaled sale price predictions
prediction['SalePrice'] = np.floor(np.expm1((0.3 * lasso_fit.predict(X_test)) +
                                            (0.3 * ridge_fit.predict(X_test)) +
                                            (0.1 * gradb_fit.predict(X_test)) +
                                            (0.3 * stackcv_fit.predict(X_test.values))
                                           )
                                  )
prediction = prediction[['Id', 'SalePrice']]
prediction.head()
```

<img src='/images/HousingPrice/out70.jpg'>

```python
# Saving prediction file
prediction.to_csv('prediction.csv', index=False)
```
