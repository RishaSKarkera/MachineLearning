# Symptoms Checker using Machine Learning
Identify possible conditions and treatment related to your symptoms. Help you understand what your medical symptoms could mean.

This dataset is downloaded from Kaggle and the link is mentioned below: https://www.kaggle.com/kerneler/starter-symptom-checker-aaf68256-4

The steps included in this analysis are:

1.Data Collection
2.Data Analysis
3.Data Visualization
4.Data Cleaning
5.Algorithm selection
6.Prediction
7.Saving the Model

### Importing Libraries
```
import pandas as pd
import seaborn as sb
from matplotlib.pyplot import scatter as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
```

## Step 1: Data Collection

### Collecting the data into dataframe from the local machine
```
data = pd.read_csv('symptoms.csv')
```
## Step 2: Data Analysis

In this dataset, the target attribute is 'Prognosis', which shows the disease based on the observations of different symptom. Hence, Its a classification problem as records need to be classified.

### Shows the number of rows and columns in the dataset (rows,columns).
```
data.shape
(4920, 133)
```
### Check first 5(default) values in dataset
```
data.head()
```

![](Images/head 1.png)

![](Images/head 2.png)
### Check last 5(default) values in dataset
```
data.tail()
```
![](Images/tail 1.png)

![](Images/tail 2.png)
### Information about dataset
```
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4920 entries, 0 to 4919
Columns: 133 entries, itching to prognosis
dtypes: int64(132), object(1)
memory usage: 5.0+ MB
```
### Statistical information about dataset
```
data.describe()
```
![](Images/describe 1.png)

![](Images/describe 2.png)
### To get unique elements from a columns
```
data['prognosis'].unique()
array(['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
       'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
       'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
       'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
       'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
       'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins',
       'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
       'Osteoarthristis', 'Arthritis',
       '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
       'Urinary tract infection', 'Psoriasis', 'Impetigo'], dtype=object)
```
Gives all the diseases specified in the prognosis column
### Replace 'object' categorical data with numerical by creating a dictionary for the values to be replaced
```
classes={'Fungal infection':0,'Allergy':1,
         'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,'Peptic ulcer diseae':5, 'AIDS':6, 'Diabetes ':7,
       'Gastroenteritis':8, 'Bronchial Asthma':9, 'Hypertension ':10, 'Migraine':11,
       'Cervical spondylosis':12, 'Paralysis (brain hemorrhage)':13, 'Jaundice':14,
       'Malaria':15, 'Chicken pox':16, 'Dengue':17, 'Typhoid':18, 'hepatitis A':19,
       'Hepatitis B':20, 'Hepatitis C':21, 'Hepatitis D':22, 'Hepatitis E':23,
       'Alcoholic hepatitis':24, 'Tuberculosis':25, 'Common Cold':26, 'Pneumonia':27,
       'Dimorphic hemmorhoids(piles)':28, 'Heart attack':29, 'Varicose veins':30,
       'Hypothyroidism':31, 'Hyperthyroidism':32, 'Hypoglycemia':33,
       'Osteoarthristis':34, 'Arthritis':35,
       '(vertigo) Paroymsal  Positional Vertigo':36, 'Acne':37,
       'Urinary tract infection':38, 'Psoriasis':39, 'Impetigo':40}

data.replace({'prognosis':classes},inplace=True)
```
## Step 3: Data Visualization
### To plot bar graph for prognosis
```
sb.countplot(data['prognosis'])
```
![](Images/countplot.png)
It shows the count of the records related to different diseases
There are 41 diseases in this data set and all have equal number of data 
### To coount prognosis instances
```
data['prognosis'].value_counts()
39    120
33    120
25    120
21    120
17    120
13    120
9     120
5     120
1     120
40    120
36    120
32    120
28    120
24    120
20    120
16    120
12    120
8     120
4     120
29    120
37    120
35    120
2     120
31    120
27    120
23    120
19    120
15    120
11    120
7     120
3     120
38    120
34    120
30    120
26    120
22    120
18    120
14    120
10    120
6     120
0     120
Name: prognosis, dtype: int64
```
120 records of patients having these diseases are present in this data set
### To show the correlation between 2 attributes
```
data.corr()
```
![](Images/corr 1.png)

![](Images/corr 2.png)
### Heatmap
```
plt.subplots(figsize=(20,15))
sb.set(font_scale=0.8)
x=sb.heatmap(data.corr(),annot=True,cmap='coolwarm')  #now it shows correlation between the attributes
plt.show()
```
