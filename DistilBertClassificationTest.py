#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


import time
total_start_time = time.time()


# In[1]:


import numpy as np
import pandas as pd
import random
import re
from sklearn.model_selection import train_test_split
import string
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertConfig
import torch
import pickle
import os
from io import BytesIO


# In[2]:


data = pd.read_csv('preprocessed.csv')


# In[3]:


data.head()


# In[4]:


data.shape


#     Encoding the Labels

# In[5]:


from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
data['Label'] = enc.fit_transform(data['assignment_group'])


# In[6]:


label_counts = pd.DataFrame(data['assignment_group'].value_counts())
label_counts


# In[7]:


label_values = list(label_counts.index)
order = list(pd.DataFrame(data['assignment_group'].value_counts()).index)
label_values = [l for _,l in sorted(zip(order, label_values))]

label_values


# In[8]:


Y = data['Label'].values
X = data['full_description_en'].values


# In[9]:


text_lengths = [len(X[i].split()) for i in range(len(X))]
print(min(text_lengths))
print(max(text_lengths))


#     Checking for number of datapoints text lengths grater thatn 10

# In[10]:


sum([1 for i in range(len(text_lengths)) if text_lengths[i] >= 10])


# # Tokenizing with DistilBert Tokenizer

# In[11]:


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)


# In[12]:


text_ids = list()
for sent in X:
    encoded_sent = tokenizer.encode(sent, 
                                         add_special_tokens=True,
                                         max_length = 100)
    encoded_sent.extend([0]* (100 - len(encoded_sent)))

    text_ids.append(encoded_sent)

text_ids[0]


# In[13]:


text_ids_lengths = [len(text_ids[i]) for i in range(len(text_ids))]
print(min(text_ids_lengths))
print(max(text_ids_lengths))


# In[14]:


att_masks = []
for ids in text_ids:
    masks = [int(id > 0) for id in ids]
    att_masks.append(masks)
    
att_masks[0]


# # Creating Train, Validation and Test data tensors

# In[15]:


train_x, test_val_x, train_y, test_val_y = train_test_split(text_ids, Y, random_state=111, test_size=0.3)
train_m, test_val_m = train_test_split(att_masks, random_state=111, test_size=0.3)

test_x, val_x, test_y, val_y = train_test_split(test_val_x, test_val_y, random_state=111, test_size=0.2)
test_m, val_m = train_test_split(test_val_m, random_state=111, test_size=0.2)


# In[16]:


train_x = torch.tensor(train_x)
test_x = torch.tensor(test_x)
val_x = torch.tensor(val_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)
val_y = torch.tensor(val_y)
train_m = torch.tensor(train_m)
test_m = torch.tensor(test_m)
val_m = torch.tensor(val_m)

print(train_x.shape)
print(test_x.shape)
print(val_x.shape)
print(train_y.shape)
print(test_y.shape)
print(val_y.shape)
print(train_m.shape)
print(test_m.shape)
print(val_m.shape)


# In[17]:


batch_size = 32

train_data = TensorDataset(train_x, train_m, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_x, val_m, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


# # Defining the model

# Here, We have used DistilBertForSequenceClassification pretrained model to further train on our updsampled data

# In[18]:


num_labels = len(set(Y))

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels,
                                                            output_attentions=False, output_hidden_states=False)


# In[19]:


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)

model = model.to(device)


# # Defining Train Parameters

# In[20]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of trainable parameters:', count_parameters(model), '\n', model)


# In[21]:


learning_rate = 1e-4
adam_epsilon = 1e-8

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.2},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)


# In[22]:


#from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup

num_epochs = 4
total_steps = len(train_dataloader) * num_epochs

#scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=0, t_total=total_steps)


# In[23]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[24]:


seed_val = 111

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# # Model Training

# The following code trains the model on our preprocessed data and prints the train and validation losses after each epoch

# In[ ]:


train_losses = []
val_losses = []
num_mb_train = len(train_dataloader)
num_mb_val = len(val_dataloader)

if num_mb_val == 0:
    num_mb_val = 1

for n in range(num_epochs):
    train_loss = 0
    val_loss = 0
    start_time = time.time()
    
    for k, (mb_x, mb_m, mb_y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        model.train()
        
        mb_x = mb_x.to(device)
        mb_m = mb_m.to(device)
        mb_y = mb_y.to(device)
        
        outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)
        
        loss = outputs[0]
        #loss = model_loss(outputs[1], mb_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        #scheduler.step()
        
        train_loss += loss.data / num_mb_train
    
    print ("\nTrain loss after itaration %i: %f" % (n+1, train_loss))
    train_losses.append(train_loss.cpu())
    
    with torch.no_grad():
        model.eval()
        
        for k, (mb_x, mb_m, mb_y) in enumerate(val_dataloader):
            mb_x = mb_x.to(device)
            mb_m = mb_m.to(device)
            mb_y = mb_y.to(device)
        
            outputs = model(mb_x, attention_mask=mb_m, labels=mb_y)
            
            loss = outputs[0]
            #loss = model_loss(outputs[1], mb_y)
            
            val_loss += loss.data / num_mb_val
            
        print ("Validation loss after itaration %i: %f" % (n+1, val_loss))
        val_losses.append(val_loss.cpu())
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Time: {epoch_mins}m {epoch_secs}s')


# # Saving the Model

# In[37]:


out_dir = './'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

with open(out_dir + '/train_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)
    
with open(out_dir + '/val_losses.pkl', 'wb') as f:
    pickle.dump(val_losses, f)


# In[ ]:


total_end_time = time.time()
mins, secs = epoch_time(total_start_time, total_end_time)
print(f'Total Time: {mins}m {secs}s')

