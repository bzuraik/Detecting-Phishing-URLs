# Cyber-Security

As part of a project, I was tasked with enabling ads on Book-My-Show website while ensuring user privacy and safety. To accomplish this, I needed to analyze whether specific URLs contained phishing attacks that could harm visitors to the site.

## Dataset Details:

The dataset included 11,000 samples, each corresponding to a different URL. Each sample had 32 features that described the URL with values of -1, 0, and 1. These features were used to classify each URL as phishing (1), suspicious (0), or legitimate (1).

## Exploratory Data Analysis:

To begin, we performed exploratory data analysis on the dataset. We used histograms and heatmaps to visualize the data and determined the number of samples in the dataset as well as the unique elements in each feature. We also checked if there were any null values in the features.

```python
import warnings
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, cv
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.model_selection import GridSearchCV, StratifiedKFold
```




```python
## Print first few rows of this data.
print('shape of the df', df_data.shape)
df_data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>having_IPhaving_IP_Address</th>
      <th>URLURL_Length</th>
      <th>Shortining_Service</th>
      <th>having_At_Symbol</th>
      <th>double_slash_redirecting</th>
      <th>Prefix_Suffix</th>
      <th>having_Sub_Domain</th>
      <th>SSLfinal_State</th>
      <th>Domain_registeration_length</th>
      <th>...</th>
      <th>popUpWidnow</th>
      <th>Iframe</th>
      <th>age_of_domain</th>
      <th>DNSRecord</th>
      <th>web_traffic</th>
      <th>Page_Rank</th>
      <th>Google_Index</th>
      <th>Links_pointing_to_page</th>
      <th>Statistical_report</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

#Exploratory Data Analysis
## Each sample has 32 features ranging from -1,0,1. Explore the data using histogram, heatmaps.

## Determine the number of samples present in the data, unique elements in all the features.

## Check if there is any null value in any features.

----------------------------------------------------------------------------------------





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41</td>
      <td>880</td>
      <td>129.0</td>
      <td>322</td>
      <td>126</td>
      <td>8.3252</td>
      <td>NEAR BAY</td>
      <td>452600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21</td>
      <td>7099</td>
      <td>1106.0</td>
      <td>2401</td>
      <td>1138</td>
      <td>8.3014</td>
      <td>NEAR BAY</td>
      <td>358500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52</td>
      <td>1467</td>
      <td>190.0</td>
      <td>496</td>
      <td>177</td>
      <td>7.2574</td>
      <td>NEAR BAY</td>
      <td>352100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52</td>
      <td>1274</td>
      <td>235.0</td>
      <td>558</td>
      <td>219</td>
      <td>5.6431</td>
      <td>NEAR BAY</td>
      <td>341300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52</td>
      <td>1627</td>
      <td>280.0</td>
      <td>565</td>
      <td>259</td>
      <td>3.8462</td>
      <td>NEAR BAY</td>
      <td>342200</td>
    </tr>
  </tbody>
</table>
</div>

```python
#identify the type of data in each column
df_data.info()
```

<div class="output_wrapper">
<div class="output">


<div class="output_area">




<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class 'pandas.core.frame.DataFrame'&gt;
RangeIndex: 11055 entries, 0 to 11054
Data columns (total 32 columns):
 #   Column                       Non-Null Count  Dtype 
---  ------                       --------------  ----- 
 0   index                        11055 non-null  object
 1   having_IPhaving_IP_Address   11055 non-null  int64 
 2   URLURL_Length                11055 non-null  int64 
 3   Shortining_Service           11055 non-null  int64 
 4   having_At_Symbol             11055 non-null  int64 
 5   double_slash_redirecting     11055 non-null  int64 
 6   Prefix_Suffix                11055 non-null  int64 
 7   having_Sub_Domain            11055 non-null  int64 
 8   SSLfinal_State               11055 non-null  int64 
 9   Domain_registeration_length  11055 non-null  int64 
 10  Favicon                      11055 non-null  int64 
 11  port                         11055 non-null  int64 
 12  HTTPS_token                  11055 non-null  int64 
 13  Request_URL                  11055 non-null  int64 
 14  URL_of_Anchor                11055 non-null  int64 
 15  Links_in_tags                11055 non-null  int64 
 16  SFH                          11055 non-null  int64 
 17  Submitting_to_email          11055 non-null  int64 
 18  Abnormal_URL                 11055 non-null  int64 
 19  Redirect                     11055 non-null  int64 
 20  on_mouseover                 11055 non-null  int64 
 21  RightClick                   11055 non-null  int64 
 22  popUpWidnow                  11055 non-null  int64 
 23  Iframe                       11055 non-null  int64 
 24  age_of_domain                11055 non-null  int64 
 25  DNSRecord                    11055 non-null  int64 
 26  web_traffic                  11055 non-null  int64 
 27  Page_Rank                    11055 non-null  int64 
 28  Google_Index                 11055 non-null  int64 
 29  Links_pointing_to_page       11055 non-null  int64 
 30  Statistical_report           11055 non-null  int64 
 31  Result                       11055 non-null  int64 
dtypes: int64(31), object(1)
memory usage: 2.7+ MB
</pre>
</div>
</div>

</div>
</div>

```python
df_data.nunique()
```
```python
having_IPhaving_IP_Address     2
URLURL_Length                  3
Shortining_Service             2
having_At_Symbol               2
double_slash_redirecting       2
Prefix_Suffix                  2
having_Sub_Domain              3
SSLfinal_State                 3
Domain_registeration_length    2
Favicon                        2
port                           2
HTTPS_token                    2
Request_URL                    2
URL_of_Anchor                  3
Links_in_tags                  3
SFH                            3
Submitting_to_email            2
Abnormal_URL                   2
Redirect                       2
on_mouseover                   2
RightClick                     2
popUpWidnow                    2
Iframe                         2
age_of_domain                  2
DNSRecord                      2
web_traffic                    3
Page_Rank                      2
Google_Index                   2
Links_pointing_to_page         3
Statistical_report             2
Result                         2
dtype: int64
```

```python
#check for NULL value in the dataset
df_data.isnull().sum().sum()
```

```python
0
```

```python
# Duplicate check
df1 = df_data.T
print(df1.duplicated().sum()) # there is no duplicate column values
```

```python
0
```

### Plot histogram and heat map for data exploration

```python
df_data.hist(bins=50, figsize=(20,15))
plt.show()
```

![SC1](https://user-images.githubusercontent.com/95400232/220487194-ac55c90c-81b2-4c18-a37f-6cb0e6e973b2.png)

```python
pd.value_counts(df_data['Result']).plot.bar()
```

![__results___15_1](https://user-images.githubusercontent.com/95400232/220487356-37d1a98b-6426-4e8c-8c53-bd99301a8b23.png)

```python
#correlation map
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(df_data.corr(),annot=True, linewidths=.5, fmt='.1f',ax=ax)
```

![__results___16_1](https://user-images.githubusercontent.com/95400232/220487498-08f5c53e-a33b-4196-b57d-724c9d673612.png)

## Correlation of Features and Feature Selection:

### Next, we looked for any correlated features in the data and removed any features that were highly correlated.

```python
# threshold greater than 0.75
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.75)]
print(to_drop)
```
```python
['double_slash_redirecting', 'port', 'HTTPS_token', 'Submitting_to_email', 'popUpWidnow']
```
```python
#drop the columns which are highly correlated 
df_data.drop(to_drop,axis=1,inplace=True)
```

```python
X=df_data.drop(columns='Result')
X
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>having_IPhaving_IP_Address</th>
      <th>URLURL_Length</th>
      <th>Shortining_Service</th>
      <th>having_At_Symbol</th>
      <th>Prefix_Suffix</th>
      <th>having_Sub_Domain</th>
      <th>SSLfinal_State</th>
      <th>Domain_registeration_length</th>
      <th>Favicon</th>
      <th>Request_URL</th>
      <th>...</th>
      <th>on_mouseover</th>
      <th>RightClick</th>
      <th>Iframe</th>
      <th>age_of_domain</th>
      <th>DNSRecord</th>
      <th>web_traffic</th>
      <th>Page_Rank</th>
      <th>Google_Index</th>
      <th>Links_pointing_to_page</th>
      <th>Statistical_report</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11050</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11051</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11052</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11053</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11054</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>


```python
Y=df_data['Result']
Y= pd.DataFrame(Y)
Y.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>

## Building a Classification Model:

### In the second week, I built a robust classification system to identify phishing URLs. We used a binary classifier to detect malicious or phishing URLs and plotted the ROC curve to illustrate the diagnostic ability of this binary classifier. We validated the accuracy of the data using the K-Fold cross-validation technique. The final output was a model that gave maximum accuracy on the validation dataset with selected attributes.

```python
#model build for different binary classification and show confusion matrix

def build_model(model_name,train_X, train_Y, test_X, test_Y):
    if model_name == 'LogisticRegression':
        model=LogisticRegression()
    elif model_name =='KNeighborsClassifier':
        model = KNeighborsClassifier(n_neighbors=4)
    elif model_name == 'XGBClassifier':
        model = XGBClassifier(objective='binary:logistic',eval_metric='auc')
    else:
        print('not a valid model name')
    
    model=model.fit(train_X,train_Y)
    
    pred_prob=model.predict_proba(test_X)
    
    fpr, tpr, thresh = roc_curve(test_Y, pred_prob[:,1], pos_label=1)
    
    model_predict= model.predict(test_X)
    acc=accuracy_score(model_predict,test_Y)
    print("Accuracy: ",acc)
    
    # Classification report 
    print("Classification Report: ")
    print(classification_report(model_predict,test_Y))
    #print("Confusion Matrix for", model_name)
    con =  confusion_matrix(model_predict,test_Y)
    sns.heatmap(con,annot=True, fmt ='.2f')
    plt.suptitle('Confusion Matrix for '+model_name, x=0.44, y=1.0, ha='center', fontsize=25)
    plt.xlabel('Predict Values', fontsize =25)
    plt.ylabel('Test Values', fontsize =25)
    plt.show()
    return model, acc, fpr, tpr, thresh
```

```python
#Model 1 - LogisticRegression
lg_model,acc1, fpr1, tpr1, thresh1 = build_model('LogisticRegression',train_X, train_Y, test_X, test_Y.values.ravel())
```

```python
Accuracy:  0.9092553512209828
Classification Report: 
              precision    recall  f1-score   support

           0       0.90      0.90      0.90      1477
           1       0.92      0.92      0.92      1840

    accuracy                           0.91      3317
   macro avg       0.91      0.91      0.91      3317
weighted avg       0.91      0.91      0.91      3317
```

![__results___28_2](https://user-images.githubusercontent.com/95400232/220488605-7e8b5926-a829-4f1d-afd8-fdf103d1df3a.png)

```python
# Model 2 - KNeighborsClassifier
knn_model,acc2, fpr2, tpr2, thresh2 = build_model('KNeighborsClassifier',train_X, train_Y, test_X, test_Y.values.ravel())
```
```python
Accuracy:  0.9439252336448598
Classification Report: 
              precision    recall  f1-score   support

           0       0.96      0.92      0.94      1554
           1       0.93      0.97      0.95      1763

    accuracy                           0.94      3317
   macro avg       0.95      0.94      0.94      3317
weighted avg       0.94      0.94      0.94      3317
```
![__results___30_2](https://user-images.githubusercontent.com/95400232/220488722-de4f9f7c-6581-4300-bc11-836d93c13efb.png)

```python
# Model 3 - XGBClassifier
xgb_model, acc3, fpr3, tpr3, thresh3 = build_model('XGBClassifier',train_X, train_Y, test_X, test_Y.values.ravel())
```

```python
Accuracy:  0.9662345492915285
Classification Report: 
              precision    recall  f1-score   support

           0       0.96      0.96      0.96      1482
           1       0.97      0.97      0.97      1835

    accuracy                           0.97      3317
   macro avg       0.97      0.97      0.97      3317
weighted avg       0.97      0.97      0.97      3317
```
![__results___32_1](https://user-images.githubusercontent.com/95400232/220488815-67db8313-e1f0-41ba-a585-31778bfcae0a.png)

```python
# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(test_Y))]
p_fpr, p_tpr, _ = roc_curve(test_Y, random_probs, pos_label=1)
```

```python
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
plt.plot(fpr3, tpr3, linestyle='--',color='red', label='XGBClassifier')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show()
```
![__results___35_0](https://user-images.githubusercontent.com/95400232/220488894-c0c5c5f5-e08a-4c35-a795-464f6309c059.png)
#### ROC plot shows XGBClassifier True Positive rate is higher than the other models.

##Data Acuracy Validation
Using GridSearchCV with StratifiedKFold cross-validation technique to validate the accuracy of data and find best parameter of different binary classifier models.

###LogisticRegression Model
```python
import warnings
warnings.filterwarnings("ignore")

# Create the parameter grid based on the results of random search
param_grid = {
    'solver':['liblinear','newton-cg'],
    'C': [0.01,0.1,1,10,100],
    'penalty': ["l1","l2"]
}
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = LogisticRegression() , param_grid = param_grid,
cv = StratifiedKFold(4), n_jobs = -1, verbose = 1, scoring = 'accuracy' )

grid_search.fit(train_X,train_Y.values.ravel())
```
#### Fitting 4 folds for each of 20 candidates, totalling 80 fits


```python
print('Best Parameter:')
print('F1 Score:', grid_search.best_score_)
print('Best Hyperparameters:', grid_search.best_params_)
print('Model object with best parameters:')
print(grid_search.best_estimator_)
```
Best Parameter:
F1 Score: 0.9119919888624345
Best Hyperparameters: {'C': 10, 'penalty': 'l1', 'solver': 'liblinear'}
Model object with best parameters:
LogisticRegression(C=10, penalty='l1', solver='liblinear')

### KNeighborsClassifier Model evaluation using GridSearchCV
```python
grid_params = {
    'n_neighbors':[3,4,11,19],
    'weights':['uniform','distance'],
    'metric':['euclidean','manhattan']
}
gs= GridSearchCV(
KNeighborsClassifier(),
grid_params,
verbose=1,
cv=3,
n_jobs=-1
)
gs_results = gs.fit(train_X,train_Y.values.ravel())
```
#### Fitting 3 folds for each of 16 candidates, totalling 48 fits

```python
print('Best Parameter:')
print('F1 Score:', gs_results.best_score_)
print('Best Hyperparameters:', gs_results.best_params_)
print('Model object with best parameters:')
print(gs_results.best_estimator_)
```
Best Parameter:
F1 Score: 0.9568364237886406
Best Hyperparameters: {'metric': 'manhattan', 'n_neighbors': 11, 'weights': 'distance'}
Model object with best parameters:
KNeighborsClassifier(metric='manhattan', n_neighbors=11, weights='distance')

### XGBClassifier with kfold cross validation

```python
xgb_cv = XGBClassifier(n_estimators=100,objective='binary:logistic',eval_metric='auc')
scores = cross_val_score(xgb_cv, train_X, train_Y.values.ravel(), cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
```
Scores: [0.96382429 0.97157623 0.9496124  0.97286822 0.9496124  0.96899225
 0.97028424 0.97157623 0.95989651 0.96377749]
Mean: 0.9642020250642652
Standard Deviation: 0.00829683486401786

### The final output consists of the model, which will give maximum accuracy on the validation dataset with selected attributes.

```python
results=pd.DataFrame({'Model':['LogisticRegression','KNN','XGBoost'],
                     'Accuracy Score':[acc1,acc2,acc3]})
result_df=results.sort_values(by='Accuracy Score', ascending=False)
result_df=result_df.set_index('Model')
result_df
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy Score</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>XGBoost</th>
      <td>0.966235</td>
    </tr>
    <tr>
      <th>KNN</th>
      <td>0.943925</td>
    </tr>
    <tr>
      <th>LogisticRegression</th>
      <td>0.909255</td>
    </tr>
  </tbody>
</table>
