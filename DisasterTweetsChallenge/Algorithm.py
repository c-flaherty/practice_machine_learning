#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm as loading_bar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import transformers
import torch.utils.data as data_helpers


# In[ ]:


# Load data

train_data = pd.read_csv("nlp-getting-started/train.csv")
eval_data = pd.read_csv("nlp-getting-started/test.csv")

print(f'Training data shape: {train_data.shape}')
print(train_data.head())

print(f'Testing data shape: {eval_data.shape}')
print(eval_data.head())


# In[ ]:


# Check to see if there are NaN's in text or target columns

print("Counts of NaN entries by column: \n",train_data.isnull().sum())


# In[ ]:


# Split training data into features and labels

train_x, train_y = train_data.to_numpy()[:, :-1], train_data.iloc[:, -1].to_numpy(np.int64)
eval_x = eval_data.to_numpy()


print(f'Training Data: (Tweet Info: {train_x.shape}, Labels: {train_y.shape})')
print(f'Submission Data: (Tweet Info: {eval_x.shape})')


# In[ ]:


# Split training data for 2-Fold Cross Validation

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1)


print(f'Training Data: (Tweet Info: {train_x.shape}, Labels: {train_y.shape})')
print(f'Testing Data: (Tweet Info: {test_x.shape}, Labels: {test_y.shape})')
print(f'Submission Data: (Tweet Info: {eval_x.shape})')


# In[ ]:


# Import pre-trained tokenizer

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')


# In[ ]:


# Determine a good max length for strings

max_len = 0

# For every sentence...
for sent in train_x[:,3]:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)


# In[ ]:


# Test that add_special_tokens adds [CLS] to start of sequence

print(tokenizer.encode("HI my name is", add_special_tokens=True))
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)


# In[ ]:


# Tokenize train_x, test_x, and eval_x

def encode(tweet):
    return tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length = 120,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors = 'pt',
    )

encoded_train_x = [encode(tweet) for tweet in train_x[:,3]]
print(len(encoded_train_x))
encoded_train_x = torch.cat([ 
    torch.reshape(
        torch.cat((tweet['input_ids'], tweet['attention_mask'])), (1,2,-1)
    ) 
    for tweet 
    in encoded_train_x 
])
print(encoded_train_x.size())

encoded_test_x = [encode(tweet) for tweet in test_x[:,3]]
print(len(encoded_test_x))
encoded_test_x = torch.cat([ 
    torch.reshape(
        torch.cat((tweet['input_ids'], tweet['attention_mask'])), (1,2,-1)
    ) 
    for tweet 
    in encoded_test_x 
])
print(encoded_test_x.size())

encoded_eval_x = [encode(tweet) for tweet in eval_x[:,3]]
print(len(encoded_eval_x))
encoded_eval_x = torch.cat([ 
    torch.reshape(
        torch.cat((tweet['input_ids'], tweet['attention_mask'])), (1,2,-1)
    ) 
    for tweet 
    in encoded_eval_x 
])
print(encoded_eval_x.size())

encoded_train_x[0]

# Dim(encoded_train_x) = (Num Samples, 2, Length of Input_Ids/Attention_Mask)


# In[ ]:


# Store labels in tensors

train_y = torch.Tensor(train_y).long()
test_y = torch.Tensor(test_y).long()

print(train_y.dtype, test_y.dtype)


# In[ ]:


# Define Model

'''
IDEA: I could create a custom classification head on BERT that takes as input the output of BERT model,
         AND the location and keyword information
'''
model = transformers.BertForSequenceClassification.from_pretrained(
    'bert-base-cased',
    num_labels=2,
    output_attentions = False,
    output_hidden_states = False,
)


# In[ ]:


# Define optimizer and loss

train_sampler = data_helpers.BatchSampler(
            data_helpers.SubsetRandomSampler(range(train_x.shape[0])),
            batch_size=4,
            drop_last=False
        )

test_sampler = data_helpers.BatchSampler(
            data_helpers.SubsetRandomSampler(range(test_x.shape[0])),
            batch_size=4,
            drop_last=False
        )

optimizer = transformers.AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

# Test that sampler works properly
for batch in sampler:
    batch_input_ids = encoded_train_x[batch, 0, :]
    batch_attention_mask = encoded_train_x[batch, 1, :]
    batch_labels = train_y[batch]
    
    print(encoded_train_x[batch,:,:].size())
    print(batch_input_ids.size())
    print(batch_attention_mask.size())
    print(len(batch_labels))
    
    break


# In[ ]:


# If GPA/CUDA is available, move model and data to GPU

if torch.cuda.is_available():  
    print("cuda is available")
    cuda = torch.device('cuda')  
    model = model.to(cuda)
    encoded_train_x = encoded_train_x.to(cuda)
    train_y = train_y.to(cuda)
    encoded_test_x = encoded_test_x.to(cuda)
    test_y = test_y.to(cuda)
    encoded_eval_x = encoded_eval_x.to(cuda)


# In[ ]:


# Training

model.train()

epochs = 4
for epoch in range(epochs):
    for batch in loading_bar(train_sampler, desc=f'(Current Epoch: {epoch})'):
        batch_input_ids = encoded_train_x[batch, 0, :]
        batch_attention_mask = encoded_train_x[batch, 1, :]
        batch_labels = train_y[batch]
        
        optimizer.zero_grad()
        
        loss, logits = model(batch_input_ids, 
                             token_type_ids=None, 
                             attention_mask=batch_attention_mask,
                             labels=batch_labels)
        
        loss.backward()
        
        optimizer.step()
        
        


# In[ ]:


# Testing

model.eval()

total_loss = 0
num_correct = 0
for batch in loading_bar(test_sampler):
    batch_input_ids = encoded_test_x[batch, 0, :]
    batch_attention_mask = encoded_test_x[batch, 1, :]
    batch_labels = test_y[batch]
    
    with torch.no_grad():
        loss, logits = model(batch_input_ids, 
                             token_type_ids=None, 
                             attention_mask=batch_attention_mask,
                             labels=batch_labels)
    
    total_loss += loss.item()
    
    MAP = torch.argmax(logits, dim=1)
    num_correct += (batch_labels == MAP).sum().item()
    
accuracy = num_correct/test_x.shape[0]
print(accuracy)


# In[ ]:


# Evaluate

predictions = []
for i in loading_bar(range(len(encoded_eval_x))):
    input_ids = encoded_eval_x[i, 0, :].reshape(1,-1)
    attention_mask = encoded_eval_x[i, 1, :].reshape(1,-1)
    
    with torch.no_grad():
        logits = model(input_ids, 
                             token_type_ids=None, 
                             attention_mask=attention_mask)
    
    MAP = torch.argmax(logits[0]).item()
    predictions.append(MAP)


# In[ ]:


# Save Results

output = pd.DataFrame({'id': eval_x[:,0], 'target': predictions})
output.to_csv("submission.csv", index=False)

