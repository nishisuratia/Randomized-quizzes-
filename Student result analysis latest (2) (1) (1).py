#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('Resultdata.csv')
df


# In[ ]:





# In[3]:


df = pd.DataFrame(df,  columns = ['syear','student_id','StudentName','standard_id','standard_title','subject_id','subject_title','exam_id','exam_title','obtain_marks','total_marks'])
df


# In[5]:


df['percent'] = (df['obtain_marks'] /
                      df['total_marks'])  * 100
df


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


import requests


# In[ ]:





# In[9]:


df['syear'].value_counts()


# In[10]:


df['StudentName'].value_counts()


# In[11]:


df['standard_id'].value_counts()


# In[12]:


df['subject_title'].value_counts()


# In[13]:


df['exam_title'].value_counts()


# In[14]:


df.nunique() #number of unique values in each column


# In[15]:


sns.set(rc={"figure.figsize":(30, 5)})


# In[16]:


ax = sns.scatterplot(x="subject_title", y="obtain_marks", data=df)


# In[17]:


ax = sns.scatterplot(x="subject_title", y="percent", data=df)


# In[18]:


grade=df.groupby("standard_title").aggregate({'obtain_marks':'mean'})
grade.reset_index(inplace=True)
grade


# In[ ]:





# In[19]:


grade1=df.groupby("subject_title").aggregate({'obtain_marks':'mean'})
grade1.reset_index(inplace=True)
grade1


# In[20]:


sns.barplot(data=grade,x='standard_title',y='obtain_marks');
plt.show(); #on an average marks are obtained highest in std1


# In[ ]:





# In[21]:


df = df.sort_values('syear')
print(df)


# In[22]:


df1=df[:16341] #2017
df1 #splitting data into resptive years 


# In[23]:


df2=df[16341:34449]#2018
df2


# In[24]:


df4=df[51738:59634] #2020
df4


# In[25]:


df5=df[59634:]#2021
df5


# In[26]:


df3=df[34449:51738] #2019
df3


# In[27]:


sns.barplot(data=df,x='syear',y='percent');
plt.show();


# In[28]:


sns.set(rc={"figure.figsize":(30, 5)})


# In[29]:


ax=sns.boxplot(x="subject_title", y="percent", data=df1);
ax.set_title(" subject vs percent in 2017")


# In[30]:


ax=sns.boxplot(x="subject_title", y="percent", data=df2);
ax.set_title(" subject vs percent in 2018")


# In[31]:


ax=sns.boxplot(x="subject_title", y="percent", data=df3);
ax.set_title(" subject vs percent in 2019")


# In[32]:


ax=sns.boxplot(x="subject_title", y="percent", data=df4);
ax.set_title(" subject vs percent in 2020")


# In[33]:


ax=sns.boxplot(x="subject_title", y="percent", data=df5);
ax.set_title(" subject vs percent in 2021")


# In[34]:


ax=sns.boxplot(x="exam_title", y="percent", data=df1);
ax.set_title(" type of exam vs percent in 2017")


# In[35]:


ax=sns.boxplot(x="exam_title", y="percent", data=df2);
ax.set_title(" type of exam vs percent in 2018")


# In[36]:


ax=sns.boxplot(x="exam_title", y="percent", data=df3);
ax.set_title(" type of exam vs percent in 2019")


# In[37]:


ax=sns.boxplot(x="exam_title", y="percent", data=df4);
ax.set_title(" type of exam vs percent in 2020")# more exams in 2020 and 2021


# In[38]:


ax=sns.boxplot(x="exam_title", y="percent", data=df5);
ax.set_title(" type of exam vs percent in 2021")


# In[39]:


ax=sns.stripplot(x="standard_title", y="percent", data=df1);
ax.set_title(" standard vs percent in 2017") #std 1-8 data is present in the year 2017 with student scoring in higher percentile


# In[40]:


ax=sns.stripplot(x="standard_title", y="percent", data=df2);
ax.set_title(" standard vs percent in 2018")


# In[41]:


ax=sns.stripplot(x="standard_title", y="percent", data=df3);
ax.set_title(" standard vs percent in 2017")


# In[42]:


ax=sns.stripplot(x="standard_title", y="percent", data=df4);
ax.set_title(" standard vs percent in 2020")


# In[43]:


ax=sns.stripplot(x="standard_title", y="percent", data=df5);
ax.set_title(" standard vs percent in 2021")


# In[44]:


ax=sns.stripplot(x="standard_title", y="percent", data=df);
ax.set_title(" standard vs percent overall for 5 years")


# In[45]:


dff = df.sort_values('subject_title')#sorting according to standard
print(dff)


# In[46]:


df['subject_title'].value_counts()


# In[47]:


dff.head()


# In[48]:


dff.tail()


# In[49]:


dff1=dff[:71] #subject -activity
dff1


# In[50]:


dff2=dff[71:215]#subject- computer
dff2


# In[51]:


dff3=dff[215:430]#subject- drawing
dff3


# In[52]:


dff4=dff[430:12788]#subject -english
dff4


# In[53]:


dff5=dff[12788:15133]#subject -environment
dff5


# In[54]:


dff6=dff[15133:26835]#subject gujarati
dff6


# In[55]:


dff7=dff[26835:36498]#subject -hindi
dff7


# In[56]:


dff8=dff[36498:38000]#subject -mari aas paas
dff8


# In[57]:


dff9=dff[38000:50281]#subject -mathematics
dff9


# In[58]:


dff10=dff[50281:50352]#subject -music
dff10


# In[59]:


dff11=dff[50352:57101]#subject -sanskrit
dff11


# In[60]:


dff12=dff[57101:64690]#subject -science and technology
dff12


# In[61]:


dff13=dff[64690:]#subject -social science
dff13


# In[62]:


#1 activity 
#2 computer
#3 drawing
#4 english
#5 environment 
#6 gujarati
#7 hindi 
#8 mari as pas
#9 maths
#10 music
#11 sanskrit
#12 science and technology
#13 social science


# In[63]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff1);# standard analysis
ax.set_title(" standard vs percent for activity")#only std1 and std2 have activity as a subject


# In[64]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff2);
ax.set_title(" standard vs percent for computer")#few students in computer for std-3-7


# In[65]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff3);
ax.set_title(" standard vs percent for drawing")#std 1-7 have drawing


# In[66]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff4);
ax.set_title(" standard vs percent for english")# all std have english


# In[67]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff5);
ax.set_title(" standard vs percent for environment")#std 2-5 have environment subject


# In[68]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff6);
ax.set_title(" standard vs percent for gujarati")# all std have gujarati


# In[69]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff7);
ax.set_title(" standard vs percent for hindi") #std 1-9 have hindi


# In[70]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff9);
ax.set_title(" standard vs percent for mathematics")#all std 1-10 have mathematics


# In[71]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff10);
ax.set_title(" standard vs percent for music")#few students in music and in std1-2 only


# In[72]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff11);
ax.set_title(" standard vs percent for sanskrit")#std 6-10 have sanskrit


# In[73]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff12);
ax.set_title(" standard vs percent for science and technology")#std 6-10


# In[74]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff13);
ax.set_title(" standard vs percent for social science")#std 6-10


# In[75]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff2);# exam analysis
ax.set_title(" exam vs percent for computer")# computer only has prelim exams


# In[76]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff1);
ax.set_title(" exam vs percent for activity")#activity only have prelims


# In[77]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff3);
ax.set_title(" exam vs percent for drawing")#drawing only have prelims


# In[78]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff4);
ax.set_title(" exam vs percent for english")#english has all sorts of exams


# In[79]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff5);
ax.set_title(" exam vs percent for environment")


# In[80]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff6);
ax.set_title(" exam vs percent for gujarati")


# In[81]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff9);
ax.set_title(" exam vs percent for mathematics")#below mentioned exams for math


# In[82]:


ax=sns.stripplot(x="subject_title", y="percent", data=dff);
ax.set_title(" subject vs percent")#


# In[83]:


frames=[dff9,dff12]


# In[84]:


result=pd.concat(frames)


# In[85]:


result


# In[86]:


value = result.sort_values('standard_title')#sorting according to standard
print(value)


# In[87]:


value['standard_title'].value_counts()


# In[88]:


one=value[326:3577]# dataset of science and maths with std 6-10
one


# In[89]:


two=value[7735:]
two


# In[90]:


frame=[one,two]


# In[91]:


final=pd.concat(frame)
final


# In[92]:


final.rename(columns={"standard_title": "standard"}, inplace=True)
final


# In[93]:


final['standard'] = final['standard'].replace(['GM-STD-10'],'10')
final['standard'] = final['standard'].replace(['GM-STD-9'],'9')
final['standard'] = final['standard'].replace(['GM-STD-8'],'8')
final['standard'] = final['standard'].replace(['GM-STD-7'],'7')
final['standard'] = final['standard'].replace(['GM-STD-6'],'6')
final['subject_title'] = final['subject_title'].replace(['Science & Technology'],'Science')
final


# In[ ]:





# In[94]:


final.head()


# In[ ]:





# In[95]:


final.info()


# In[96]:


categorical=final.select_dtypes(include=['object']).columns.tolist()
categorical


# In[97]:


final = final.astype({'standard':'int'})
final


# In[98]:


final.info()


# In[ ]:





# In[99]:


part2=pd.read_csv("PAL_Questions_v1.csv")
part2


# In[ ]:





# # Standard 6 science questions

# In[ ]:





# In[100]:


part3=part2[8771:]
part3


# In[101]:


part4=final.sort_values(by=['subject_title','standard'])
part4


# # Stanndard 10 Science results

# In[102]:


part5=part4[10525:12042]
part5


# In[103]:


df_merged = part3.merge(part5, on='standard', how='outer')
df_merged


# In[104]:


df_merged.info()


# In[105]:


ax=sns.stripplot(x="Cognitive Difficulty", y="percent", data=df_merged);
ax.set_title(" difficulty vs percent ")


# In[106]:


ax=sns.stripplot(x="Difficulty Levels", y="percent", data=df_merged);
ax.set_title(" difficulty vs percent ")


# In[107]:


df_merged['percent'].value_counts()


# In[108]:


sns.histplot(df_merged['percent'])


# In[109]:


sns.histplot(df_merged['Cognitive Difficulty'])


# In[110]:


df_merged['Cognitive Difficulty'].value_counts()


# In[111]:


sns.histplot(df_merged["Bloom's Taxonomy"])


# In[112]:


df_merged["Bloom's Taxonomy"].value_counts()


# In[113]:


df_merged.isnull().sum()


# In[114]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


# In[ ]:





# In[115]:


features=df_merged.drop("Cognitive Difficulty",axis='columns')
target=df_merged["Cognitive Difficulty"]


# In[116]:


label_encoder = LabelEncoder()
encoded_target = label_encoder.fit_transform(target)


# In[ ]:





# In[117]:


splitter=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in splitter.split(features,encoded_target):
  X_train,X_test=features.iloc[train_index],features.iloc[test_index]
  y_train,y_test=encoded_target[train_index],encoded_target[test_index]


# In[118]:


categorical_column = X_train.select_dtypes(include=['object']).columns.tolist()
X_train_encoded = X_train.copy()  # Make a copy to avoid modifying the original data
for col in categorical_column:
    X_train_encoded[col] = label_encoder.fit_transform(X_train[col])

X_test_encoded = X_test.copy()  # Make a copy to avoid modifying the original data
for col in categorical_column:
    X_test_encoded[col] = label_encoder.fit_transform(X_test[col])


# In[ ]:





# In[119]:


X_train_encoded.shape


# In[120]:


X_test.shape


# In[121]:


y_train.shape


# In[122]:


y_test.shape


# In[123]:


model=RandomForestClassifier(n_estimators=30, max_depth=10)
model.fit(X_train_encoded,y_train)


# In[124]:


predictions=model.predict(X_test_encoded)


# In[125]:


accuracy=accuracy_score(y_test,predictions)
print(f'Accuracy:{accuracy}')


# In[126]:


import pandas as pd
import requests

# Sample student ID
student_id = 540

# Assuming df_merged is defined elsewhere in your code
df_merged['percent_category'] = pd.cut(df_merged['percent'], bins=[0, 33, 66, 100], labels=['low', 'medium', 'high'])

# Extract desired category based on student ID
desired_category = df_merged.loc[df_merged['student_id'] == student_id, 'percent_category'].values[0]

# Define desired bloom taxonomy based on desired category
if desired_category == 'low':
    desired_bloom = ['Knowledge', 'Comprehension']
elif desired_category == 'medium':
    desired_bloom = ['Application', 'Analysis']
else:
    desired_bloom = ['Synthesis', 'Evaluation']

# Filter questions based on desired category and bloom taxonomy
selected_questions = df_merged.loc[(df_merged['percent_category'] == desired_category) & 
                                   (df_merged["Bloom's Taxonomy"].isin(desired_bloom)), ['id','title']].sample(n=5)

# Print selected questions
print("Selected Questions:")
for i, (_, question) in enumerate(selected_questions.iterrows(), start=1):
    question_id = question['id']
    question_text = question['title']
    print(f"Question {i} (ID: {question_id}): {question_text}")

# API endpoint
api_endpoint = "https://erp.triz.co.in/api/pal_questions"

# Prepare data associated with the selected student ID
selected_student_data = df_merged[df_merged['student_id'] == student_id].to_dict(orient='records')[0]

# Send data to API
response = requests.post(api_endpoint, json=selected_student_data)

# Check response status code
if response.status_code == 200:
    # Parse JSON response
    api_response = response.json()
    
    # Extract questions from the API response
    questions = api_response.get('title', [])
    
    # Display questions
    print("\nAPI Response:")
    print("Selected Questions from API:")
    for i, question in enumerate(questions, start=1):
        print(f"Question {i}: {question}")
else:
    print(f"Failed to retrieve questions. Status code: {response.status_code}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




