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


```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/RishaSKarkera/MachineLearning/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
