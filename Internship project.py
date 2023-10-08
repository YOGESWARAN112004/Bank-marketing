#!/usr/bin/env python
# coding: utf-8

# # Importing required libraries 

# In[1]:


import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# In[2]:


warnings.filterwarnings(action="ignore")


# ## Importing dataset

# In[3]:


df = pd.read_csv('bank-full.csv', delimiter=';')


# In[4]:


df.head(10)


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[7]:


object_columns = df.select_dtypes(include=['object'])

# Loop through each selected column
for column_name in object_columns:
    unique_values = df[column_name].unique()
    print(f"Column: {column_name}")
    print(unique_values)
    print("-" * 30)


# In[8]:


sns.kdeplot(df['age'], shade=True)

# Add labels and a title
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Age Distribution Density Plot')



# In[9]:


# Calculate basic descriptive statistics for age
age_mean = df['age'].mean()
age_median = df['age'].median()
age_std = df['age'].std()
age_min = df['age'].min()
age_max = df['age'].max()

# Print the descriptive statistics
print(f"Mean Age: {age_mean}")
print(f"Median Age: {age_median}")
print(f"Standard Deviation: {age_std}")
print(f"Minimum Age: {age_min}")
print(f"Maximum Age: {age_max}")


# In[10]:


plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', hue='y', kde=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution by Outcome')
plt.show()


# In[11]:


from scipy.stats import chi2_contingency

# List of categorical columns (excluding 'y' which is the target)
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Set your significance level
alpha = 0.05

# Iterate through each categorical column and perform the chi-squared test
for column in categorical_columns:
    contingency_table = pd.crosstab(df[column], df['y'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    if p < alpha:
        print(f"Reject the null hypothesis: There is a significant association between '{column}' and 'y'.")
    else:
        print(f"Fail to reject the null hypothesis: '{column}' and 'y' are independent.")


# In[12]:


from scipy.stats import f_oneway

# List of numerical columns
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Set your significance level
alpha = 0.05

# Iterate through each numerical column and perform the one-way ANOVA test
for column in numerical_columns:
    groups = [group for _, group in df.groupby('y')[column]]
    f_statistic, p_value = f_oneway(*groups)
    
    if p_value < alpha:
        print(f"Reject the null hypothesis: There are significant differences in '{column}' across 'y' categories.")
    else:
        print(f"Fail to reject the null hypothesis: '{column}' does not significantly differ across 'y' categories.")


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
plt.figure(figsize=(8, 4))
sns.kdeplot(data=df, x='campaign', shade=True)
plt.title('Density Plot for Campaign')
plt.xlabel('Campaign')
plt.ylabel('Density')
plt.show()


# In[14]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 5, 1)
sns.countplot(data=df, x='job')
plt.title('Countplot for job')

plt.subplot(1, 5, 2)
sns.countplot(data=df, x='marital')
plt.title('Countplot for marital')

plt.subplot(1, 5, 3)
sns.countplot(data=df, x='education')
plt.title('Countplot for education')

plt.subplot(1, 5, 4)
sns.countplot(data=df, x='default')
plt.title('Countplot for default')

plt.subplot(1, 5, 5)
sns.countplot(data=df, x='poutcome')
plt.title('Countplot for poutcome')

plt.tight_layout()
plt.show()


# In[15]:


numeric_columns = df.select_dtypes(include='number')  # Select numeric columns
plt.figure(figsize=(12, 6))
sns.boxplot(data=numeric_columns)
plt.xticks(rotation=90)
plt.show()


# ## Removing outliers 

# In[16]:


def find_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)


# In[17]:


columns_to_check = ['age', 'balance', 'day','duration']

# Find outliers for each column
outliers_by_column = {}
for col in columns_to_check:
    outliers_by_column[col] = find_outliers_iqr(df[col])

# Combine outliers from different columns using logical OR
all_outliers = pd.DataFrame(outliers_by_column).any(axis=1)

# Identify and print outliers
outliers = df[all_outliers]
print("Outliers in the specified columns:")
print(outliers)

# Remove the outliers from the DataFrame
df = df[~all_outliers]


# In[18]:


numeric_columns = df.select_dtypes(include='number')  # Select numeric columns
plt.figure(figsize=(12, 6))
sns.boxplot(data=numeric_columns)
plt.xticks(rotation=90)
plt.show()


# In[19]:


df.shape


# ## Converting columns to binary

# In[20]:


binary_columns = ['default', 'housing', 'loan', 'y']

# Convert binary categorical columns to binary numerical columns (1 and 0)
for col in binary_columns:
    df[col] = df[col].map({'yes': 1, 'no': 0})
df.columns


# In[21]:


non_numeric_columns = df.select_dtypes(exclude=['number']).columns

# Drop the non-numeric columns from the DataFrame
df_numeric = df.drop(columns=non_numeric_columns)

# Create a correlation matrix for the numeric columns
corr_matrix = df_numeric.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)

# Customize the heatmap
plt.title('Correlation Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


# ## Converting categorical columns

# In[22]:


df.drop('pdays', axis=1, inplace=True)
df.drop('previous', axis=1, inplace=True)

# Specify the categorical columns other than binary
cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

# Perform one-hot encoding for each non-binary categorical column
df = pd.get_dummies(df, columns=cat_columns, drop_first=True)




# In[23]:


boolean_columns = df.select_dtypes(include='bool').columns

 # Map 'False' to 0 and 'True' to 1 in the boolean columns
df[boolean_columns] = df[boolean_columns].astype(int)


# In[24]:


df.head(10)


# In[25]:


X = df.drop(columns=['y'])  # Features (exclude the 'y' column)
y = df['y']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a logistic regression model
logistic_reg_model = LogisticRegression(random_state=42)

# Train the model on the training data
logistic_reg_model.fit(X_train, y_train)

# Make predictions on the test data
y_prob = logistic_reg_model.predict_proba(X_test)[:, 1]

# Adjust the threshold to improve precision for class 1 (e.g., to 0.7)
threshold = 0.7
y_pred_adjusted = (y_prob >= threshold).astype(int)

# Evaluate the adjusted predictions
adjusted_precision = precision_score(y_test, y_pred_adjusted)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred_adjusted)
confusion = confusion_matrix(y_test, y_pred_adjusted)
classification_report_output = classification_report(y_test, y_pred_adjusted)

# Print the results
print(f'Adjusted Precision for class 1: {adjusted_precision:.2f}')
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(confusion)
print('Classification Report:')
print(classification_report_output)


# In[26]:


models = {
    'LDA': LinearDiscriminantAnalysis(),
    'k-NN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Dtc ': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),

}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, Y_pred)
    print(f"{model_name}: Accuracy = {accuracy}")

    # Define a filename based on the model name and accuracy
    filename = f"{model_name}_accuracy_{round(accuracy, 2)}.joblib"

    # Save the model to the specified filename
    joblib.dump(model, filename)

    print(f"Model saved as {filename}")
    print(classification_report(y_test, Y_pred, digits=3))


# In[27]:


from sklearn.feature_selection import SelectKBest, f_classif

# Initialize SelectKBest with f_classif scoring function
select_k_best = SelectKBest(score_func=f_classif, k=5)  # Replace 5 with the desired number of features

# Fit SelectKBest to your data
select_k_best.fit(X_train, y_train)

# Get the scores of features for SelectKBest
feature_scores_k_best = select_k_best.scores_

# Get the selected features for SelectKBest
selected_features_indices = select_k_best.get_support()
selected_features_k_best = X.columns[selected_features_indices]

print("SelectKBest Selected Features:")
print(selected_features_k_best)


# ## what is the distribution 0f customer age?

# Mean Age: 40.93621021432837
# 
# Median Age: 39.0
# 
# Standard Deviation: 10.618762040975431
# 
# Minimum Age: 18
# 
# Maximum Age: 95

# ![image.png](attachment:image.png)

# ## What is the relationship between customer age and subscription?

# ![image.png](attachment:image.png)

# ## Are there any other factors that are correlated with subscription?

# sa per the statistical reports every columns are some how correlated with subscription column

# ## What is the accuracy of the logistic regression model?

# The accuracy of  the logistic regression model is 0.93

# ## What are the most important features for the logistic regression model?

# => job
# 
# => duration
# 
# => contact
# 
# => month
# 
# => poutcome
# 
# =>age

# ## What is the precision of the logistic regression model?

# precison for 0 is 94
# 
# precison for 1 is 75
# 

# ##  What is the recall of the logistic regression model?

# recall value for 0 is 0.99
# 
# recall value for 1 is 0.20
# 

# ## What is the f1-score of the logistic regression model?

# f1-score for 0 is 0.96
# 
# f1-score for 1 is 0.31

# ## How can you improve the performance of the logistic regression model?

# To make our model work better, we need to pay attention to the information we give it. This means picking the important details and leaving out the ones that don't matter much. We also need to make sure all the numbers are in the same range and easy to understand.
# 
# So, we start by choosing the right details from our data. Some details are important, like how much money someone has, while others might not help much. We keep the good ones and throw away the less useful ones.
# 
# Next, we make sure all the numbers are in a similar range. Imagine comparing a car's speed in miles per hour to its weight in pounds – it doesn't work well. We make the numbers easier to work with by putting them in the same range.
# 
# This way, our model can understand the data better and make better predictions. It's like giving the model the right tools to do its job well
# 

# ## What are the limitations of the logistic regression model?

# ### Straight-Line Thinking:
# It likes straight lines. If the relationship between stuff you're looking at isn't a straight line, it might not do a great job.
# 
# ### Not for Super Complicated Stuff:
# For really complex problems where things get all twisty-turny, logistic regression can struggle. It's like using a ruler for a curve – doesn't work well.
# 
# ### Too Many Features: 
# If you have lots and lots of things to look at, logistic regression can get confused. It's like trying to keep track of too many toys – it gets overwhelmed.
# 
# ### Friends Shouldn't Be Too Close: 
# It doesn't like when your features are too similar to each other (highly correlated). Think of it as not liking twins – it wants things to be more unique.
# 
# ### Outliers Throw It Off:
# If you have weird data points (outliers), logistic regression can be easily influenced by them. It's like one loud voice in a quiet room – it pays too much attention.
# 
# ### Lopsided Data:
# When one group is way bigger than the other, logistic regression can favor the big group and not do well with the smaller one. It's like cheering for the team with the most players.
# 
# ### Missing Data Needs Fixing: 
# If some data is missing, you have to fill in the blanks. Logistic regression doesn't like gaps in the story.
# 
# ### Only Good for Two Choices:
# It's great for yes-or-no questions but may need help for more than two choices.
# 
# ### Probabilities Aren't Always Perfect:
# Sometimes, the probabilities it gives you aren't super accurate. You might need to adjust them.
# 
# ### Numbers Only:
# It prefers numbers. If you have words or categories, you have to turn them into numbers first.
# 
# ### It's Not for Every Puzzle:
# For super tricky problems, logistic regression might not be your best puzzle piece. Sometimes, you need fancier tools.
# 
# ### Outliers Again:
# Those weird data points? They can throw it off balance. Like a wobbly bike wheel, it might not roll smoothl

# In[ ]:




