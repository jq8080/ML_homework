#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import sys
import zipfile
import subprocess

from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from tqdm.notebook import tqdm
from copy import deepcopy

import json


# In[2]:


DATASET = 'ml-1m' 
RAW_PATH = os.path.join('./', DATASET)

RANDOM_SEED = 0
NEG_ITEMS = 99


# # Load data
# 
# 1. Load interaction data and item metadata
# 2. Filter out items with less than 5 interactions
# 3. Calculate basic statistics

# In[3]:


# download data if not exists

if not os.path.exists(RAW_PATH):
    subprocess.call('mkdir ' + RAW_PATH, shell=True)
if not os.path.exists(os.path.join(RAW_PATH, DATASET + '.zip')):
    print('Downloading data into ' + RAW_PATH)
    subprocess.call(
        'cd {} && curl -O http://files.grouplens.org/datasets/movielens/{}.zip'
        .format(RAW_PATH, DATASET), shell=True)
    print('Unzip files...')
    f = zipfile.ZipFile(os.path.join(RAW_PATH, DATASET + '.zip'),'r') 
    for file in f.namelist():
        print("Extract %s"%(file))
        f.extract(file,RAW_PATH)
    f.close()


# In[4]:


# read interaction data
interactions = []
user_freq, item_freq = dict(), dict()
file = os.path.join(RAW_PATH,"ratings.dat")
with open(file) as F:
    header = 0
    for line in tqdm(F):
        if header == 1:
            header = 0
            continue
        line = line.strip().split("::")
        uid, iid, rating, time = line[0], line[1], float(line[2]), float(line[3])
        if rating >= 4:
            label = 1
        else:
            label = 0
        interactions.append([uid,time,iid,label])
        if int(label)==1:
            user_freq[uid] = user_freq.get(uid,0)+1
            item_freq[iid] = item_freq.get(iid,0)+1


# In[5]:


# 5-core filtering
select_uid, select_iid = [],[]
while len(select_uid)<len(user_freq) or len(select_iid)<len(item_freq):
    select_uid, select_iid = [],[]
    for u in user_freq:
        if user_freq[u]>=5:
            select_uid.append(u)
    for i in item_freq:
        if item_freq[i]>=5:
            select_iid.append(i)
    print("User: %d/%d, Item: %d/%d"%(len(select_uid),len(user_freq),len(select_iid),len(item_freq)))

    select_uid = set(select_uid)
    select_iid = set(select_iid)
    user_freq, item_freq = dict(), dict()
    interactions_5core = []
    for line in tqdm(interactions):
        uid, iid, label = line[0], line[2], line[-1]
        if uid in select_uid and iid in select_iid:
            interactions_5core.append(line)
            if int(label)==1:
                user_freq[uid] = user_freq.get(uid,0)+1
                item_freq[iid] = item_freq.get(iid,0)+1
    interactions = interactions_5core


# In[6]:


print("Selected Interactions: %d, Users: %d, Items: %d"%(len(interactions),len(select_uid),len(select_iid)))


# In[7]:


# Get timestamp
ts = []
for i in tqdm(range(len(interactions))):
    ts.append(datetime.fromtimestamp(interactions[i][1]))


# In[13]:


# Construct and Save 5 core results with situation context
interaction_df = pd.DataFrame(interactions,columns = ["user_id","time","news_id","label"])
interaction_df['timestamp'] = ts
interaction_df['hour'] = interaction_df['timestamp'].apply(lambda x: x.hour)
interaction_df['weekday'] = interaction_df['timestamp'].apply(lambda x: x.weekday())
interaction_df['date'] = interaction_df['timestamp'].apply(lambda x: x.date())

def get_time_range(hour): # according to the Britannica dictionary
    # https://www.britannica.com/dictionary/eb/qa/parts-of-the-day-early-morning-late-morning-etc
    if hour>=5 and hour<=8:
        return 0
    if hour>8 and hour<11:
        return 1
    if hour>=11 and hour<=12:
        return 2
    if hour>12 and hour<=15:
        return 3
    if hour>15 and hour<=17:
        return 4
    if hour>=18 and hour<=19:
        return 5
    if hour>19 and hour<=21:
        return 6
    if hour>21:
        return 7
    return 8 # 0-4 am

interaction_df['period'] = interaction_df.hour.apply(lambda x: get_time_range(x))
min_date = interaction_df.date.min()
interaction_df['day'] = (interaction_df.date - min_date).apply(lambda x: x.days)

interaction_df.to_csv("interaction_5core.csv",index=False)
interaction_df["user_id"] = interaction_df["user_id"].astype(int)
interaction_df["item_id"] = interaction_df["news_id"].astype(int)


# # Prepare data for CTR & Reranking task
# 
# 1. Rename and organize all interaction features
# 2. Split dataset into training, validation, and test; Save interaction files
# 3. Assign impression ID (not necessary for CTR prediction)
# 4. Organize item metadata

# In[9]:





# # Prepare data for Top-k Recommendation Task
# 1. Rename all interaction features
# 2. Split dataset into training, validation, and test
# 3. Re-assign IDs to user, item, and context; Save interaction files
# 4. Organize item metadata

# In[ ]:





# In[14]:


TOPK_PATH='./ML_1MTOPK/'
os.makedirs(TOPK_PATH,exist_ok=True)


# In[15]:


# copy & rename columns
interaction_pos = interaction_df.loc[interaction_df.label==1].copy() # retain positive interactions
interaction_pos.rename(columns={'hour':'c_hour_c','weekday':'c_weekday_c','period':'c_period_c','day':'c_day_f',
                              'user_id':'original_user_id'}, inplace=True)


# In[16]:


# split training, validation, and test sets.
split_time1 = int(interaction_pos.c_day_f.max() * 0.8)
train = interaction_pos.loc[interaction_pos.c_day_f<=split_time1].copy()
val_test = interaction_pos.loc[(interaction_pos.c_day_f>split_time1)].copy()
val_test.sort_values(by='time',inplace=True)
split_time2 = int(interaction_pos.c_day_f.max() * 0.9)
val = val_test.loc[val_test.c_day_f<=split_time2].copy()
test = val_test.loc[val_test.c_day_f>split_time2].copy()

# Delete user&item in validation&test sets that not exist in training set
train_u, train_i = set(train.original_user_id.unique()), set(train.news_id.unique())
val_sel = val.loc[(val.original_user_id.isin(train_u))&(val.news_id.isin(train_i))].copy()
test_sel = test.loc[(test.original_user_id.isin(train_u))&(test.news_id.isin(train_i))].copy()
print("Train user: %d, item: %d"%(len(train_u),len(train_i)))
print("Validation user: %d, item:%d"%(val_sel.original_user_id.nunique(),val_sel.news_id.nunique()))
print("Test user: %d, item:%d"%(test_sel.original_user_id.nunique(),test_sel.news_id.nunique()))
train.label.sum(),train.label.mean(),val_sel.label.sum(),val_sel.label.mean(),test_sel.label.sum(),test_sel.label.mean()


# In[17]:


# Assign ids for users and items (to generate continous ids)
all_df = pd.concat([train,val_sel,test_sel],axis=0)
user2newid_topk = dict(zip(sorted(all_df.original_user_id.unique()), 
                      range(1,all_df.original_user_id.nunique()+1)))
 
for df in [train,val_sel,test_sel]:
    df['user_id'] = df.original_user_id.apply(lambda x: user2newid_topk[x])

item2newid_topk = dict(zip(sorted(all_df.news_id.unique()), 
                      range(1,all_df.news_id.nunique()+1)))
for df in [train,val_sel,test_sel]:
    df['item_id'] = df['news_id'].apply(lambda x: item2newid_topk[x])

all_df['user_id'] = all_df.original_user_id.apply(lambda x: user2newid_topk[x])
all_df['item_id'] = all_df['news_id'].apply(lambda x: item2newid_topk[x])


# In[18]:


nu2nid = dict()
ni2nid = dict()
for i in user2newid_topk.keys():
    oi = int(i)
    nu2nid[oi] = user2newid_topk[i]

for i in item2newid_topk.keys():
    oi = int(i)
    ni2nid[oi] = item2newid_topk[i]
json.dump(nu2nid,open(os.path.join(TOPK_PATH,"user2newid.json"),'w'))
json.dump(ni2nid,open(os.path.join(TOPK_PATH,"item2newid.json"),'w'))


# In[19]:


# generate negative items
def generate_negative(data_df,all_items,clicked_item_set,random_seed,neg_item_num=99):
    np.random.seed(random_seed)
    neg_items = np.random.choice(all_items, (len(data_df),neg_item_num))
    for i, uid in tqdm(enumerate(data_df['user_id'].values)):
        user_clicked = clicked_item_set[uid]
        for j in range(len(neg_items[i])):
            while neg_items[i][j] in user_clicked|set(neg_items[i][:j]):
                neg_items[i][j] = np.random.choice(all_items, 1)
    return neg_items.tolist()

clicked_item_set = dict()
for user_id, seq_df in all_df.groupby('user_id'):
    clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())
all_items = all_df.item_id.unique()
val_sel['neg_items'] = generate_negative(val_sel,all_items,clicked_item_set,random_seed=1)
test_sel['neg_items'] = generate_negative(test_sel,all_items,clicked_item_set,random_seed=2)


# In[20]:


select_columns = ['user_id','item_id','time','c_hour_c','c_weekday_c','c_period_c','c_day_f']
train[select_columns].to_csv(os.path.join(TOPK_PATH,'train.csv'),sep="\t",index=False)
val_sel[select_columns+['neg_items']].to_csv(os.path.join(TOPK_PATH,'dev.csv'),sep="\t",index=False)
test_sel[select_columns+['neg_items']].to_csv(os.path.join(TOPK_PATH,'test.csv'),sep="\t",index=False)


# In[21]:


# organize & save item metadata
item_meta = pd.read_csv(os.path.join(DATASET, "movies.dat"),
            sep='::',names=['movieId','title','genres'],encoding='latin-1',engine='python') # columns: movieId,title,genres
item_select = item_meta.loc[item_meta.movieId.isin(interaction_pos.news_id.unique())].copy()
item_select['item_id'] = item_select.movieId.apply(lambda x: item2newid_topk[x])
genres2id = dict(zip(sorted(item_select.genres.unique()),range(1,item_select.genres.nunique()+1)))
item_select['i_genre_c'] = item_select['genres'].apply(lambda x: genres2id[x])
title2id = dict(zip(sorted(item_select.title.unique()),range(1,item_select.title.nunique()+1)))
item_select['i_title_c'] = item_select['title'].apply(lambda x: title2id[x])

item_select[['item_id','i_genre_c','i_title_c']].to_csv(
    os.path.join(TOPK_PATH,'item_meta.csv'),sep="\t",index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




