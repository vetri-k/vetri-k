---
title: "Indiana Vehicle Crash"
excerpt: "Predict vehicle crash injury using classification techniques <br/><img src='/images/INVehCrash/car-crash.jpg'>"
collection: portfolio
---

# Introduction

This project provides a case study of traffic accidents classification and severity prediction in the State of Indiana. Indiana vehicle crash information and data provided by State of Indiana can be used to classify accidents according to the severity and predictive models can be built. Identifying accident severity in vehicle crash will help in making the after accident protocols faster and efficient. This will also help in implementing road safety policies based on the criteria identified using the data set. 

The Automated Reporting Information Exchange System (ARIES) is the State of Indiana’s crash repository. Crash data is generated through first responder crash reports and collected within ARIES.

```python
# Loading neccesary library packages

import pandas as pd # dataframe manipulation
import numpy as np # array manipulation

import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from datetime import datetime # data and time manipulation
plt.style.use('fivethirtyeight') # plotting style
```

```python
# reading dataset
df_filepath="../input/aries-crash-data-2019/aries_crash_data_2019.csv"
# dtypes = {'INDEXING_NUMBER': 'float64', }
df=pd.read_csv(df_filepath)
```

```python
# Checking dataset size
print(df.shape)
df.head()
```

<img src='/images/INVehCrash/out3.jpg'>

Total number of accidents recorded in the state of Indiana for 2019 is 625702

# Exploratory data analysis

```python
# Visualizing collisions based on county
COUNTYDESCR_count=(pd.value_counts(df.COUNTYDESCR)*100)/len(df.COUNTYDESCR)
plt.figure(figsize=(20,10))
sns.barplot(x=COUNTYDESCR_count.index, y=COUNTYDESCR_count)
plt.ylabel("Vehicle Crash (%)")
plt.xticks(rotation=90)
plt.show()
```

<img src='/images/INVehCrash/out6.png'>

```python
# Function for plotting multiple bar charts
def barchart_multi(dataframe):
    fig, axes=plt.subplots(7, 3, figsize=(25,55))
    axes=axes.flatten()
    
    for i, j in zip(dataframe.columns, axes):
        dataframe_count=(pd.value_counts(dataframe[i])*100)/len(dataframe[i])
        sns.barplot(x=dataframe_count.index, y=dataframe_count, palette='plasma', ax=j)
        j.tick_params(labelrotation=90)            
    
    plt.tight_layout()
```

```python
# Plotting interested features to understand the dataset
df_bc=df[['GENDERCDE', 'AGE_GRP', 'POSINVEHDESCR', 'INJSTATUSDESCR', 'INJNATUREDESCR', 'INJLOCCDESCR', 
          'COLLISION_DAY', 'COLLISION_MONTH', 'COLLISION_TIME_AM_PM', 'MOTORVEHINVOLVEDNMB', 'LIGHTCONDDESCR', 
         'WEATHERDESCR', 'SURFACETYPECDE_CONDDESCR', 'SURFACETYPEDESCR', 'PRIMARYFACTORDESCR', 'MANNERCOLLDESCR',
          'UNITTYPEDESCR', 'AXELSTXT', 'SPEEDLIMITTXT', 'ROADTYPEDESCR', 'PRECOLLACTDESCR']]
barchart_multi(df_bc)
```

<img src='/images/INVehCrash/out8.png'>

# Observations
* Comparatively more male drivers were involved in accidents.
* Age group 15-24 years are involved in more accidents compared to other age groups.
* Driver's are most injured. This might be because of one person traveling in car.
* Significant number of injured people had complaint of pain.
* Majority of head injury was observed.
* Higher accidents were observed on Friday.
* Collision month data might be missing since no accident records were observed for December month and significantly less records for Novemeber.
* Higher accidents occur at PM.
* Comparatively higher accidents occur during daylight, clear weather and dry road condition. This suggests accidents may happen due to driver being complacent.
* Rear end collision is higher compared to other manner of accidents.
* Passenger car / 2 axle vehicles are involved in more crash.
* Higher percentage of crash happens on two lanes (two way), where speed limit is 30 mph and while going straight.
* Primary reason for crash is following too closely.

```python
# Features narrow downed for modeling based on priliminary analysis
df2=df[['GENDERCDE', 'AGE_GRP', 'SAFETYEQUUSEDDESCR', 'INJSTATUSCDE', 'INJSTATUSDESCR', 'INJNATUREDESCR', 'INJLOCCDESCR', 'COUNTYDESCR', 
'CITYDESCR', 'COLLDTE', 'COLLISION_DAY', 'COLLISION_MONTH', 'COLLISION_YEAR', 'COLLISION_TIME_AM_PM', 'MOTORVEHINVOLVEDNMB', 
'TRAILERSINVOLVEDNMB', 'LIGHTCONDDESCR', 'WEATHERDESCR', 'SURFACETYPECDE_CONDDESCR', 'SURFACETYPEDESCR', 'PRIMARYFACTORDESCR', 
'MANNERCOLLDESCR', 'TRAFFICCNTRLDESCR', 'UNITTYPEDESCR', 'VEHYEARTXT', 'VEHMAKETXT', 'VEHMODELTXT', 'OCCUPSNMB', 'AXELSTXT', 
'SPEEDLIMITTXT', 'ROADTYPEDESCR', 'PRECOLLACTDESCR']]
```

# Addressing missing values

```python
print(df2.shape)
df2.isnull().sum()
```

<img src='/images/INVehCrash/out11-1.png'>
<img src='/images/INVehCrash/out11-2.png'>

```python
# "INJSTATUSCDE" is the attribute to predict. Dropping rows where "INJSTATUSCDE" records are not found. 
df2=df2.dropna(subset=['INJSTATUSCDE'])
print(df2.shape)
df2.isnull().sum()
```

<img src='/images/INVehCrash/out12.png'>

```python
# Visualizing missing values
missing=df2.isnull().sum()
percent=(missing/len(df2))*100
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
plt.ylabel("Missing %")
sns.barplot(x=missing.index, y=percent)
plt.show()
```

<img src='/images/INVehCrash/out13.png'>

# Next steps (currently working on)
Feature engineering
Attributes selection
Machine learning model
Results and discussion
