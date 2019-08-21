# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans


r = 'D:/project FOR Money Dashboard/raw data/hash/20190101_20190625_MDB_Transaction_Data_Ford_FULL_20190625.csv'
#read datafile for 2012 or 2013
#r = 'D:/project FOR Money Dashboard/raw data/hash/20120101_20121231_MDB_Transaction_Data_Ford_FULL_20190626.csv'
#r = 'D:/project FOR Money Dashboard/raw data/hash/20130101_20131231_MDB_Transaction_Data_Ford_FULL_20190626.csv'
file = pd.read_csv(r,sep = "\"\|\"",nrows = 2000000)

#sort out the formate of file
file = file.rename(columns=lambda x: x.replace('"','')) 
file['Hashbyte'] = file['Hashbyte'].str.replace('\"','')
#add user reference into hashbyte
file['Detection'] = file['User Reference'].astype(str)+file['Hashbyte']
#add the replication number in to the table as"count"
counted = file.groupby(['Detection']).size().reset_index().rename(columns={0: 'count'})
file = pd.merge(file,counted, on = ['Detection'])
#create "age" field
file['Age'] = file['Year of Birth'].apply(lambda x: 2019-x)
# filling the missing values
#fill the "Age" with the mean value of it
file['Age'].fillna(round(file['Age'].mean(),0),inplace = True)
#fill the income with "N" as a special type
file['Salary Range'].fillna('N',inplace = True)
# add "year", "month" and "day" fields to the file 
datem = file['Transaction Date'].str.split('-', expand=True)
new_col = ['year', 'month', 'day']
datem.columns = new_col
file = pd.merge(file,datem,how='left',left_index=True,right_index=True)
#check the data has covered every date of the first half of 2019
#print(len(set(file['Transaction Date'])))

#extract all the repeated transactions
repeated = file[file['count']>1]
# transactions that are identified as genuine
unrepeated = file[file['count']==1]
#re-set the index of repeated transaction
repeated['index'] = range(repeated.shape[0])
repeated = repeated.set_index('index')
#create the file of all repeated 
#du.to_csv(path_or_buf='D:/project FOR Money Dashboard/raw data/hash/2019sample.csv',encoding = 'utf-8')

#the ratio of repeated data to unrepeated data
#print(len(repeated)/len(unrepeated))


'''
The following part is the code for investigation of the data

'''

#####################################################################
#get points to divide the samples into 10 equal parts
q10 = np.percentile(file['Amount'], 10)
q20 = np.percentile(file['Amount'], 20)
q30 = np.percentile(file['Amount'], 30)
q40 = np.percentile(file['Amount'], 40)
q50 = np.percentile(file['Amount'], 50)
q60 = np.percentile(file['Amount'], 60)
q70 = np.percentile(file['Amount'], 70)
q80 = np.percentile(file['Amount'], 80)
q90 = np.percentile(file['Amount'], 90)
q100 = np.percentile(file['Amount'], 100)
#get the proportion of proportion of repeated data to the total repeated data
total_re = len(repeated)
len(repeated[repeated.Amount < q10])/total_re
len(repeated[(repeated.Amount>= q10) & (repeated.Amount < q20)])/total_re
len(repeated[(repeated.Amount>= q20) & (repeated.Amount < q30)])/total_re
len(repeated[(repeated.Amount>= q30) & (repeated.Amount < q40)])/total_re
len(repeated[(repeated.Amount>= q40) & (repeated.Amount < q50)])/total_re
len(repeated[(repeated.Amount>= q50) & (repeated.Amount < q60)])/total_re
len(repeated[(repeated.Amount>= q60) & (repeated.Amount < q70)])/total_re
len(repeated[(repeated.Amount>= q70) & (repeated.Amount < q80)])/total_re
len(repeated[(repeated.Amount>= q80) & (repeated.Amount < q90)])/total_re
len(repeated[(repeated.Amount>= q90) & (repeated.Amount < q100)])/total_re
#get the proportion of proportion of unrepeated data to the total unrepeated data
total_un = len(unrepeated)
len(unrepeated[unrepeated.Amount < q10])/total_un
len(unrepeated[(unrepeated.Amount>= q10) & (unrepeated.Amount < q20)])/total_un
len(unrepeated[(unrepeated.Amount>= q20) & (unrepeated.Amount < q30)])/total_un
len(unrepeated[(unrepeated.Amount>= q30) & (unrepeated.Amount < q40)])/total_un
len(unrepeated[(unrepeated.Amount>= q40) & (unrepeated.Amount < q50)])/total_un
len(unrepeated[(unrepeated.Amount>= q50) & (unrepeated.Amount < q60)])/total_un
len(unrepeated[(unrepeated.Amount>= q60) & (unrepeated.Amount < q70)])/total_un
len(unrepeated[(unrepeated.Amount>= q70) & (unrepeated.Amount < q80)])/total_un
len(unrepeated[(unrepeated.Amount>= q80) & (unrepeated.Amount < q90)])/total_un
len(unrepeated[(unrepeated.Amount>= q90) & (unrepeated.Amount < q100)])/total_un
####################################################################################
#count the number of transacrions in every date
date_total = file.groupby(['Transaction Date']).size().reset_index().rename(columns={0: 'count'})
#count the number of repeated transacrions in every date
date_re = repeated.groupby(['Transaction Date']).size().reset_index().rename(columns={0: 'c'})
date = pd.merge(date_total,date_re,on = ['Transaction Date'])
date['proportion_re'] =date['c']/date['count']
date['proportion_re'].plot()

#################################################################################################3


#count the number of repeated transacrions for every type of tag name
tag_re = repeated.groupby(['Auto Purpose Tag Name']).size().reset_index().rename(columns={0: 'c'})
#count the number of total transacrions for every type of tag name
tag_total = file.groupby(['Auto Purpose Tag Name']).size().reset_index().rename(columns={0: 'count'})
#count the number of unrepeated transacrions for every type of tag name
tag_un = unrepeated.groupby(['Auto Purpose Tag Name']).size().reset_index().rename(columns={0: 'count'})
#get  5 tags that have the highest proportion of repeated number
tag_name = pd.merge(tag_total,tag_re,on = ['Auto Purpose Tag Name'])
tag_name['proportion_re'] = tag_name['c']/tag_name['count']
#tag_name['proportion_gt'] = tag_name['count']/len(gt)
c = tag_name.sort_values(by="proportion_du" , ascending= False)


##################################################################################################

'''
The following part is the code for Chi-aquare test

'''



gender_re= repeated.groupby(['Derived Gender']).size().reset_index().rename(columns={0: 'c'})
gender_re['p_re'] = gender_re['c']/len(repeated)
gender_un = unrepeated.groupby(['Derived Gender']).size().reset_index().rename(columns={0: 'count'})
gender_un['proportion'] = gender_un['count']/len(unrepeated)

stats.chisquare(gender_re['p_re'], f_exp = gender_un['proportion'])


################################################################
#Chi-square test for income

income_re = repeated.groupby(['Salary Range']).size().reset_index().rename(columns={0: 'c'})
income_re['p_re'] = income_re['c']/len(repeated)
income_un = unrepeated.groupby(['Salary Range']).size().reset_index().rename(columns={0: 'count'})
income_un['proportion'] = income_un['count']/len(unrepeated)


stats.chisquare(income_re['p_re'], f_exp = income_un['proportion'])

##############################################################################
#年龄#Chi-square test for age

age_re = repeated.groupby(['Age']).size().reset_index().rename(columns={0: 'count_du'})
age_re['p_re'] = age_re['count_du']/len(repeated)
age_un = unrepeated.groupby(['Age']).size().reset_index().rename(columns={0: 'count'})
age_un['proportion'] = age_un['count']/len(unrepeated)

age = pd.merge(age_un,age_re,on = 'Age')
stats.chisquare(age['p_re'], f_exp = age['proportion'])

################################################################################
#Chi-square test for tag

tag_re = repeated.groupby(['Auto Purpose Tag Name']).size().reset_index().rename(columns={0: 'count_du'})
tag_re['p_re'] = tag_re['count_du']/len(repeated)
tag_un = unrepeated.groupby(['Auto Purpose Tag Name']).size().reset_index().rename(columns={0: 'count'})
tag_un['proportion'] = tag_un['count']/len(unrepeated)

tag = pd.merge(tag_re,tag_un,on = 'Auto Purpose Tag Name')
stats.chisquare(tag['p_re'], f_exp = tag['proportion'])

#####################################################################
#Chi-square test for amount

amount_re = repeated.groupby(['Amount']).size().reset_index().rename(columns={0: 'count_du'})
amount_re['p_re'] = amount_re['count_du']/len(repeated)
amount_un = unrepeated.groupby(['Amount']).size().reset_index().rename(columns={0: 'count'})
amount_un['proportion'] = amount_un['count']/len(unrepeated)

amount = pd.merge(amount_re,amount_un,on = 'Amount')
stats.chisquare(amount['p_re'], f_exp = amount['proportion'])

##########################################################################3
#Chi-square test for date

date_re = repeated.groupby(['Transaction Date']).size().reset_index().rename(columns={0: 'count_du'})
date_re['p_re'] = date_re['count_du']/len(repeated)
date_un = unrepeated.groupby(['Transaction Date']).size().reset_index().rename(columns={0: 'count'})
date_un['proportion'] = date_un['count']/len(unrepeated)

date = pd.merge(date_re,date_un,on = 'Transaction Date')
stats.chisquare(date['p_re'], f_exp = date['proportion'])

##################################################################################
#Chi-square test for month

month_re = repeated.groupby(['month']).size().reset_index().rename(columns={0: 'count_du'})
month_re['p_re'] = month_re['count_du']/len(repeated)
month_un = unrepeated.groupby(['month']).size().reset_index().rename(columns={0: 'count'})
month_un['proportion'] = month_un['count']/len(unrepeated)

month = pd.merge(month_un,month_re,on = 'month')
stats.chisquare(month['p_re'], f_exp = month['proportion'])
#####################################################################################


"""
The following two parts are the code for Naive Bayes

"""


############################################################
'''
The following part is for the calculation of correlation coefficient

'''

#get "Age","Amount"
data = file[['Age','Amount']]
#get "month"
month = file['month'].astype(int)
data['month'] = month
#get "Salary Range" as "income"
y = np.array(file['Salary Range'])
y[y=='10K to 20K'] = 1.5
y[y=='20K to 30K'] = 2.5
y[y=='30K to 40K'] = 3.5
y[y=='40K to 50K'] = 4.5
y[y=='50K to 60K'] = 5.5
y[y=='60K to 70K'] = 6.5
y[y=='70K to 80K'] = 7.5
y[y=='< 10K'] = 0.5
y[y=='> 80K'] = 8.5
y[y=='N']=0
y = y.astype(float)
data['income'] =y 
#get the correlation coefficient
c = data.corr()

###################################################################
'''
The following part is the code for modified Naive Bayes

'''

#Divide the total sample set by the label (replication times)
group_count = file.groupby('count')
# get the number of total samples
N_total = len(file)


def get_probability(count,amount,age,tag,income,date,gender):
    
    #Retrieve the subset 
    get_table = group_count.get_group(count)
    #get the number of all the transactions in the subset
    N_yj = len(get_table)
    # p(aam|yj)
    p_amount =  len(get_table[get_table['Amount']==amount])/N_yj
    # p(am|yj)
    p_age = len(get_table[get_table['Age']==age])/N_yj
    # p(aag|yj)
    p_date =  len(get_table[get_table['Transaction Date']==date])/N_yj
    # p(at|yj)
    p_tag =  len(get_table[get_table['Auto Purpose Tag Name']==tag])/N_yj
    # p(ai|yj)
    p_income = len(get_table[get_table['Salary Range']==income])/N_yj
    # p(ag|yj)
    p_gender = len(get_table[get_table['Derived Gender']==gender])/N_yj
    # p(yj)
    p_count = len(file[file['count']==count])/N_total
    #p(aam)
    p_am = len(file[file['Amount']==amount])/N_total
    #p(at)
    p_t = len(file[file['Auto Purpose Tag Name']==tag])/N_total
    #p(aag)
    p_ag = len(file[file['Age']==age])/N_total
    #p(ad)
    p_d = len(file[file['Transaction Date']==date])/N_total
    #p(ai)
    p_i = len(file[file['Salary Range']==income])/N_total
    #p(ag)
    p_g = len(file[file['Derived Gender']==gender])/N_total
    
    p_x = p_am*p_t*p_ag*p_d*p_i*p_g
    #the result of the conditional probability
    p = (p_amount*p_age*p_date*p_tag*p_income*p_gender*p_count)/p_x
    return p
#得出结果
#get all the conditional probability for every repeated trnsaction
repeated['probability'] = repeated.apply(lambda x: get_probability( x['count'], x['Amount'],x['Age'],x['Auto Purpose Tag Name'],x['Salary Range'],x['Transaction Date'],x['Derived Gender']), axis = 1)

###########################################################################################3

'''
The following two parts are the code for K-means

'''
#####################################################################################

'''
The following part is the code for applying K-means to total samples

'''
#get all the categorical data
data_word = file[['Derived Gender','Auto Purpose Tag Name']]
# use one-hot encoding to change them into numerical data, 190 dimensions
test = pd.get_dummies(data_word)
# use Principal Component Analysis for dimension reduction,
# and choose the features that cover 85% information in the total samples
estimator = PCA(0.85)
# reduce the number of dimension to 15
test_pca=estimator.fit_transform(test)
test_pca = pd.DataFrame(test_pca)
###################################################################################
# get "age", "Amount", "count" from the datafile
data_number = file[['Age','Amount','count']]
# get "month" from the datafile
month = file['month'].astype(int)
data_number['month'] = month
# get "Salary Range" as "income" from the datafile
y = np.array(file['Salary Range'])
# use label encoding to change "Salary Range" into numerical data
y[y=='10K to 20K'] = 1.5
y[y=='20K to 30K'] = 2.5
y[y=='30K to 40K'] = 3.5
y[y=='40K to 50K'] = 4.5
y[y=='50K to 60K'] = 5.5
y[y=='60K to 70K'] = 6.5
y[y=='70K to 80K'] = 7.5
y[y=='< 10K'] = 0.5
y[y=='> 80K'] = 8.5
y[y=='N'] = 0
y = y.astype(float)
data_number['income'] =y 


# conbine the categorical data and the numerical data to one datafile
test = pd.merge(data_number,test_pca,how='left',left_index=True,right_index=True)
# delete two outliers because the amount is too large
test = test[~test['Amount'].isin([3000000.0])]
test = test[~test['Amount'].isin([3053284.8])]



# set minmax to scale the data with the range (0,10)
minmax = preprocessing.MinMaxScaler(feature_range=(0, 10))
#fit minmax with each columns 
minmax_age =  minmax.fit(test['Age'].values.reshape(-1, 1))
minmax_amount =  minmax.fit(test['Amount'].values.reshape(-1, 1))
minmax_month =  minmax.fit(test['month'].values.reshape(-1, 1))
minmax_count =  minmax.fit(test['count'].values.reshape(-1, 1))
minmax_income =  minmax.fit(test['income'].values.reshape(-1, 1))
# scale the age, amount, month, count, income
test['Age'] = minmax_age.transform(test['Age'].values.reshape(-1, 1))
test['Amount'] = minmax_amount.transform(test['Amount'].values.reshape(-1, 1))
test['month'] =minmax_month.transform(test['month'].values.reshape(-1, 1))
test['count'] =minmax_count.transform(test['count'].values.reshape(-1, 1))
test['income'] = minmax_income.transform(test['income'].values.reshape(-1, 1))


# use K-means to calculate 2 cluster center and allocate each data to 
# one of the center
kmeans = KMeans(n_clusters=2).fit(test)
# add the label for each data to the datafile
test['label']=kmeans.labels_
# count the number of data in each cluster
df_count_type=test.groupby('label').size().reset_index().rename(columns={0: 'count'})


# get the 2 cluster centers
a = pd.DataFrame(kmeans.cluster_centers_)

# inverse first 5 features(age, amount, count, month, income) of 
# the cluster center to the original scale
a[0] = minmax_age.inverse_transform(a[0].values.reshape(-1, 1))
a[1] = minmax_amount.inverse_transform(a[1].values.reshape(-1, 1))
a[2] = minmax_count.inverse_transform(a[2].values.reshape(-1, 1))
a[3] = minmax_month.inverse_transform(a[3].values.reshape(-1, 1))
a[4] = minmax_income.inverse_transform(a[4].values.reshape(-1, 1))
# inverse the numerical data to original scale
test['Age'] = minmax_age.transform(test['Age'].values.reshape(-1, 1))
test['Amount'] = minmax_amount.transform(test['Amount'].values.reshape(-1, 1))
test['month'] =minmax_month.transform(test['month'].values.reshape(-1, 1))
test['count'] =minmax_count.transform(test['count'].values.reshape(-1, 1))
test['income'] = minmax_income.transform(test['income'].values.reshape(-1, 1))

#######################################################################

'''
The following part is the code for applying K-means to repeated data

'''
# get "age", "Amount", "count" from the repeated data
re_number = repeated[['Age','Amount','count']]
# get "month" from the repeated data
month = repeated['month'].astype(int)
re_number['month'] = month
# get "Salary Range" as "income" from the repeated data
y = np.array(repeated['Salary Range'])
# use label encoding to change "Salary Range" into numerical data
y[y=='10K to 20K'] = 1.5
y[y=='20K to 30K'] = 2.5
y[y=='30K to 40K'] = 3.5
y[y=='40K to 50K'] = 4.5
y[y=='50K to 60K'] = 5.5
y[y=='60K to 70K'] = 6.5
y[y=='70K to 80K'] = 7.5
y[y=='< 10K'] = 0.5
y[y=='> 80K'] = 8.5
y[y=='N']=0
y = y.astype(float)
re_number['income'] =y 
re_number['income'].fillna(re_number['income'].mean(),inplace = True)


#get all the categorical data
re_word = repeated[['Derived Gender','Auto Purpose Tag Name']]
# use one-hot encoding to change them into numerical data, 190 dimensions
test_re = pd.get_dummies(re_word)
# use Principal Component Analysis for dimension reduction, 
#and choose the features that cover 85% information in the total samples
estimator = PCA(0.85)
# reduce the number of dimension to 15
test_pca=estimator.fit_transform(test_re)
test_pca = pd.DataFrame(test_pca)
 

# conbine the categorical data and the numerical data to one datafile
test_re = pd.merge(re_number,test_pca,how='left',left_index=True,right_index=True)

# set minmax to scale the data with the range (0,10)
minmax = preprocessing.MinMaxScaler(feature_range=(0, 10))
#fit minmax with each columns 
minmax_age =  minmax.fit(test_re['Age'].values.reshape(-1, 1))
minmax_amount =  minmax.fit(test_re['Amount'].values.reshape(-1, 1))
minmax_month =  minmax.fit(test_re['month'].values.reshape(-1, 1))
minmax_count =  minmax.fit(test_re['count'].values.reshape(-1, 1))
minmax_income =  minmax.fit(test_re['income'].values.reshape(-1, 1))
# scale the age, amount, month, count, income
test_re['Age'] = minmax_age.transform(test_re['Age'].values.reshape(-1, 1))
test_re['Amount'] = minmax_amount.transform(test_re['Amount'].values.reshape(-1, 1))
test_re['month'] =minmax_month.transform(test_re['month'].values.reshape(-1, 1))
test_re['count'] =minmax_count.transform(test_re['count'].values.reshape(-1, 1))
test_re['income'] = minmax_income.transform(test_re['income'].values.reshape(-1, 1))

# use K-means to calculate 2 cluster center and allocate each data to 
# one of the center
kmeans = KMeans(n_clusters=2).fit(test_re)
test_re['label']=kmeans.labels_
re_count_type=test_re.groupby('label').size().reset_index().rename(columns={0: 'count'})


#get 2 cluster centers
d = pd.DataFrame(kmeans.cluster_centers_)
# inverse first 5 features(age, amount, count, month, income) of 
# the cluster center to the original scale
d[0] = minmax_age.inverse_transform(d[0].values.reshape(-1, 1))
d[1] = minmax_amount.inverse_transform(d[1].values.reshape(-1, 1))
d[2] = minmax_count.inverse_transform(d[2].values.reshape(-1, 1))
d[3] = minmax_month.inverse_transform(d[3].values.reshape(-1, 1))
d[4] = minmax_income.inverse_transform(d[4].values.reshape(-1, 1))
# inverse the numerical data to original scale
test_re['Age'] = minmax_age.transform(test_re['Age'].values.reshape(-1, 1))
test_re['Amount'] = minmax_amount.transform(test_re['Amount'].values.reshape(-1, 1))
test_re['month'] =minmax_month.transform(test_re['month'].values.reshape(-1, 1))
test_re['count'] =minmax_count.transform(test_re['count'].values.reshape(-1, 1))
test_re['income'] = minmax_income.transform(test_re['income'].values.reshape(-1, 1))






