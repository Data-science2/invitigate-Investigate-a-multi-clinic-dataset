#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigate the reason of no-show appointment
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# 
# This dataset collects information from 100k medical appointments in Brazil and is focused on the question of
# whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.
# ● ‘ScheduledDay’ tells us on what day the patient set up their appointment. ● ‘Neighborhood’ indicates the location of the hospital. ● 
# ‘Scholarship’ indicates whether or not the patient is enrolled in Brasilian welfare program Bolsa Família.
# ● Be careful about the encoding of the last column: it says ‘No’ if the patient showed up to their appointment, and ‘Yes’ if they did not show up.
# 
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# In[ ]:





# In[ ]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
pd.options.display.max_rows = 9999
df.head()


# In[ ]:


#explore the data
df.shape


# the data contain 110527 appointment and 14 different columns

# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# this is the type pf data

# it appears that about32% people recived sms which is low and we must re-check the sms campaign,
# also the mean age is about 37 , probaply we have one or more row having error because of age which is(-1), we will check it
# also, the percentage of alcoholic, handcapped , chronic diseases and scolar shipped patient is low

# In[ ]:


#identify the error
error=df.query("Age==-1")
error


# this row is having error in entering age we may remove it in cleaning data to prevent error

# In[ ]:


#ispecting missing data 
df.info()


# there is no missing data

# In[ ]:


#find if tere is a duplicate data
df.duplicated().sum()


# there is no duplicate rows(data) found

# In[ ]:


#find the no of unique patient ID
df['PatientId'].nunique()


# we have only 62299 unique patient id,
#   110527-62299= 48228 that  mean we have 48228 duplicate patient id who had different appointment dates

# In[ ]:


df['Age'].max()


# As we can see the maximum age is 115 it may be a reason of not attednding

# In[ ]:


df.query('Age==115')


# will there is only 5 patient IDs with this age 2 of them has showed, 5 patient is not a signicant number, also appear that two has a unique IDs number and both of them has showed up at the end

# In[ ]:


df.sort_values(['Age'], ascending=[False])


# will it appear that the age Doesnot affect attendance except for young people because of parental care, we still need further investigation

# In[ ]:


#find the most repeated data
df.mode().head(1)


# it appears from mode that most of patient is female and the most appointment and scheduled date is in july 2016, however most people donot recieve a sms, also Most of people is children come with their parents(high parental care) (age =0)

# **cleaning data**
# 
# From the data description and questions to answer, I've determined that some of the dataset columns are not necessary for the analysis process and will therefore be removed.
# PatientId,Sms_received,AppointmentID 
# This will help to process the Data Analysis Faster.
# also we will remove age error and some misspelling.
# also we will remove duplicated id with duplicated data show
# 

# In[ ]:


#Removing -1 age
df.drop(index=99832,inplace=True)


# In[ ]:


df.describe()


# here we have our age cleaned

# In[ ]:


#correction of column names
df.rename(columns={'Hipertension':'Hypertenstion'},inplace=True)
df.rename(columns={'No-show':'no_show'},inplace=True)
df.head()


# In[ ]:


#remove duplicated id and duplicated no-show status 
df.drop_duplicates(['PatientId','no_show'],inplace=True)
df.shape


# In[ ]:


df.drop(['PatientId','AppointmentID','AppointmentDay','ScheduledDay'],axis=1,inplace=True)
df.head()
#we dropped appointment date and scheduled date because the time frame is small and not significant


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. **Compute statistics** and **create visualizations** 
# 
# 
# 
# 
# **###General look###**

# In[ ]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
df.hist(figsize=(25,7));


# ### Dividing the patient into two groups

# In[ ]:


#Dividing the patient into two groups(show and no show)
show=df.no_show=='No'
noshow=df.no_show=='Yes'
df[show].count(),df[noshow].count()


# In[ ]:


df[show]


# In[ ]:


df[noshow]


# it appears that the number of patient that attended the clinic after appointment is 54154 while who don't is 17663

# In[ ]:


df[show].mean(),df[noshow].mean()


# this show that the attended patient who recieved SMS is about 30 % while who don't is about 45%
# so we need to re-check the message sent to the patient

# In[ ]:


df[show].describe(),df[noshow].describe()


# from the previous data it shows that the average age is not signifcance to attendance or not this and other data will be discussed in details

# In[ ]:


df[show].mode(),df[noshow].mode()


# this data show that regarding to attendance the most neighbourhood shown is jardim camburai the same for non attendance
# also most patient arenot cronically ill or handcapped
# also, Most of the didnot recieve a sms

# In[ ]:


#let's divide people according to ther gender
male=df.Gender=='M'
female=df.Gender=='F'
df[male].count(),df[female].count()


# the number of males is 25350 while the number of female is 46466

# # INVISTIGATION and Questions On the Affecting Factor For The Attendance rate

# In[ ]:


df[male].mean(),df[female].mean()


# as you can see the numbers are very close so it's not a signifant one 
# let's see the percentantage of attendance and absence in the gragh

# In[ ]:


#does the gender affect attendance 
#we showed in previous that the female is more than male in appointment and gender
#let us see in graph to assure the percentage of attendance
plt.figure(figsize=[15,6])
plt.style.use('ggplot')
df['Gender'][show].value_counts(normalize=True).plot(kind='pie',label='show')
plt.legend();
plt.title('compare between attendance of gender')
plt.xlabel('Gender')
plt.ylabel('Patients Number');

#the female is more attendance because they made more appointment , we can confirm by calculating the absence 
#if it's the same percentage so the gender is not affecting attendance


# In[ ]:


#let us see the percentage of absent
plt.figure(figsize=[15,6])
df['Gender'][noshow].value_counts(normalize=True).plot(kind='pie',label='noshow')
plt.legend();
plt.title('compare between absence of gender')
plt.xlabel('Gender')
plt.ylabel('Patients Number');

#as expected gender has no influence on presence or absence


# does the scolarships affect the attendance?

# In[ ]:


df.Scholarship[show].mean()


# In[ ]:


df.Scholarship[noshow].mean()


# as it shown the difeernce in both values of prescence is insignificance
# we will show it gragh

# In[ ]:


plt.figure(figsize=[20,10])
df.Scholarship[show].value_counts().plot(kind='bar',color='green',label='show',fontsize=20)
df.Scholarship[noshow].value_counts().plot(kind='bar',color='red',label='noshow',fontsize=20)
plt.legend();


# Does the handcap affect the attendance?

# In[ ]:


df.Handcap[show].mean()


# In[ ]:


df.Handcap[noshow].mean()


# the difference in mean between both numbers is very vey low which show the the handcap is not affecting the attendance

# In[ ]:



plt.figure(figsize=[60,20])
df.Age[show].value_counts().plot(kind='bar',color='green',label='show',fontsize=30)

plt.legend();


# we can see from here there s no effect of age in the show of patient except for children till 2 years(show high parentral care) and for old people show low show 
# we need to see the percentage compared to no show for patients

# In[ ]:


plt.figure(figsize=[60,20])
df.Age[show].value_counts().plot(kind='bar',color='green',label='show',fontsize=30)
df.Age[noshow].value_counts().plot(kind='bar',color='red',label='noshow',fontsize=30)
plt.legend();


# here we can see the percentage of show to no show is the same with the exception of children who has higher percntage of attendance
# and the percentage of the old people is relatively lower which can allow us to provide services to help to bring the old people for the appointment and save their lives 

# In[ ]:


plt.figure(figsize=[15,8])
df.Neighbourhood[show].value_counts().plot(kind='bar',color='green',label='show')
df.Neighbourhood[noshow].value_counts().plot(kind='bar',color='red',label='noshow')
plt.legend();
plt.title('compare according to neig.iloc[1-71815]hbourhood')
plt.xlabel('Neighbourhood')
plt.ylabel('Patients Number');


# In[ ]:


df[show].groupby('Gender').mean()['Age'],df[noshow].groupby('Gender').mean()['Age']


# it shown that the mean age for both gender is almost the same let's see in gragh

# In[ ]:


plt.figure(figsize=[15,7])
df[show].groupby('Gender').mean()['Age'].plot(kind='bar',color='green',label='show'),df[noshow].groupby('Gender').mean()['Age'].plot(kind='bar',color='red',label='noshow')
plt.legend()
plt.ylabel('mean age');


# the mean of ages is approximatly same so there is no correlation between gender ages and attendance

# In[ ]:


plt.figure(figsize=[15,7])
df[show].groupby(['Hypertenstion','Diabetes']).mean()['Age'].plot(kind='bar',color='green',label='show'),
df[noshow].groupby(['Hypertenstion','Diabetes']).mean()['Age'].plot(kind='bar',color='red',label='noshow')
plt.legend();
plt.title('Compare Acc. to mean ages and chronic diseases attendance',fontsize=25,loc = 'left')
plt.xticks(fontsize=20,rotation=360)
plt.annotate('Healthy patient',xy=(0,30),fontsize=14,xytext=(0,45),arrowprops=dict(facecolor='yellow'),color='green')
plt.annotate('Diabetic',xy=(1,53),fontsize=14,xytext=(1,65),arrowprops=dict(facecolor='yellow'),color='green')
plt.annotate('Hypertensive',xy=(2,59),fontsize=14,xytext=(2,66),arrowprops=dict(facecolor='yellow'),color='green')
plt.annotate('Both Diesases',xy=(3,62),fontsize=14,xytext=(2.7,69),arrowprops=dict(facecolor='yellow'),color='green')
plt.ylim([0,72])
plt.xlim([-1,4])
plt.axhline(y=31,xmin=-1,xmax=0.15,linewidth=2,marker=">",ms=7,linestyle= 'dashed',color='blue')
plt.axhline(y=53,xmin=-1,xmax=0.35,linewidth=2,marker=">",ms=7,linestyle= 'dashed',color='blue')
plt.axhline(y=60,xmin=-1,xmax=0.55,linewidth=2,marker=">",ms=7,linestyle= 'dashed',color='blue')
plt.axhline(y=63,xmin=-1,xmax=0.75,linewidth=2,marker=">",ms=7,linestyle= 'dashed',color='blue')
plt.xlabel('chronic diseases')
plt.ylabel('mean age');


# In[ ]:


df[show].groupby(['Hypertenstion','Diabetes']).mean()['Age']
df[noshow].groupby(['Hypertenstion','Diabetes']).mean()['Age']


# (0,0) mean the patient is healthy
# (0,1) mean the patient is Diabetic
# (1,0)mean the patient is hypertensive
# (1,1) mean the patient has both diseases
# we see that there s no correlation between disease and attendendance

# In[ ]:


df[show].groupby(['Hypertenstion','Diabetes']).mean()['Age'],df[noshow].groupby(['Hypertenstion','Diabetes']).mean()


# this Data confirm the shown in presentation, so chronic diseases is not affecting attendance

# In[ ]:


#we will remake sure that the age doesnot affect attendance by histogram
#does age affect attendance
plt.figure(figsize=[20,5])
df['Age'][show].hist(alpha=.5,bins=15,color='green',label='show')
df['Age'][noshow].hist(alpha=.5,bins=15,color='red',label='noshow')
plt.legend();
plt.title('compare different ages')
plt.xlabel('Age')
plt.ylabel('Patients Number');


# it show that young people (children) attend more which show the high parental care as it appeared in showing the mode value

# In[ ]:


#does receiveing sms affect the attendance
#we showed preveously that they are inversiouly propotional and we need to revise the texting team and the message sent but let us make sure by plotting
plt.figure(figsize=[20,5])
df['SMS_received'][show].hist(alpha=.5,bins=15,color='green',label='show')
df['SMS_received'][noshow].hist(alpha=.5,bins=15,color='red',label='noshow')
plt.legend();
plt.title('compare according to receiving SMS')
plt.xlabel('SMS')
plt.ylabel('Patients Number');


# as it appears more than 75 percent of patient who didnot recieve message has attended while about 50 percent of patient who recieved didnot recieve
# so there is an inverse propotion relation 
# and we need to Re-check the sent text 

# In[ ]:


#effect of neighbourhood
plt.figure(figsize=[15,8])
df.Neighbourhood[show].value_counts().plot(kind='bar',color='green',label='show')
df.Neighbourhood[noshow].value_counts().plot(kind='bar',color='red',label='noshow')
plt.legend();
plt.title('compare according to neighbourhood')
plt.xlabel('Neighbourhood')
plt.ylabel('Patients Number');


# jardam camburai has the highest attendance and show in number of patients

# In[ ]:


#mean age differ from one place to another?
plt.figure(figsize=[20,5])
df[show].groupby('Neighbourhood').Age.mean().plot(kind='bar',color='green',label='show')
df[noshow].groupby('Neighbourhood').Age.mean().plot(kind='bar',color='red',label='noshow')
plt.legend();
plt.title('compare according to neighbourhood age')
plt.xlabel('Neighbourhood')
plt.ylabel('Mean age'); 


# there is different ages shown in different neighboorhood which relatively affect attendance

# In[ ]:


#Does SMS goes to every neighboorhood on the same way and its effect?
plt.figure(figsize=[20,5])
df[show].groupby('Neighbourhood').SMS_received.mean().plot(kind='bar',color='green',label='show')
df[noshow].groupby('Neighbourhood').SMS_received.mean().plot(kind='bar',color='red',label='noshow')
plt.legend();
plt.title('compare according to neighbourhood SMS recieving')
plt.xlabel('Neighbourhood')
plt.ylabel('Patient Number'); 


# as shown some neighborhood who recieved message Made response while others is not 

# <a id='conclusions'></a>
# ## Conclusions
# 
# neighboorhood has a great effect in number of patient and attendance percentage,
# The number of patient in neighboorhood is affected by ages and sometimes sms,
# in general we see the more the patient grow the less they attend the resirvation as from 0  to 3 years there is great attendance which show the great parent care and the old people show slightly low attendance may be because they can't and we can facilitate their coming or we can home appointment for severly ill patient and help to save their lives,
# but overall the age hasn't a great effect in attendance except for children and old people alittle bit,
# female has higher resirvation and attendance rate but the percentage with male is almost the same 
# limitation:No clear correlation between attendance and gender also chronic diseases doesnot affect it, handcapped patients and enrollment in welfare program also has little affect in enrollment
# 
# 

# Rescources that I used are many but most from: www.stackoverflow.com
# ,www.geeksforgeeks.org,
# www.stackoverflow.com,
# www.towardsdatascience.com,
# www.tutorialspoint.com,
# www.pandas.pydata.org 

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

