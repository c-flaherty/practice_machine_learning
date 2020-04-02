#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Load libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as loading_bar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import transformers
import torch.utils.data as data_helpers
import torch.nn as nn


# In[15]:


# Load data

train_data = pd.read_csv("../input/nlp-getting-started/train.csv")
eval_data = pd.read_csv("../input/nlp-getting-started/test.csv")

print(f'Training data shape: {train_data.shape}')
print(train_data.head())

print(f'Testing data shape: {eval_data.shape}')
print(eval_data.head())


# In[3]:


print("Counts of NaN entries by column: \n",train_data.isnull().sum())


# In[30]:


# Split training data into features and labels

train_x, train_y = train_data.to_numpy()[:, :-1], train_data.iloc[:, -1].to_numpy(np.int64)
eval_x = eval_data.to_numpy()


print(f'Training Data: (Tweet Info: {train_x.shape}, Labels: {train_y.shape})')
print('-------------')
print(train_x[0:2])
print('-------------')
print(f'Submission Data: (Tweet Info: {eval_x.shape})')


# In[31]:


# Split training data for 2-Fold Cross Validation

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1)


print(f'Training Data: (Tweet Info: {train_x.shape}, Labels: {train_y.shape})')
print(f'Testing Data: (Tweet Info: {test_x.shape}, Labels: {test_y.shape})')
print(f'Submission Data: (Tweet Info: {eval_x.shape})')


# In[20]:


# Load and configure tokenizer

tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'cls_token': '[CLS]', 'pad_token': '[PAD]'})


# In[7]:


# Determine a good max length for strings

max_len = 0

# For every sentence...
for sent in train_x[:,3]:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent+" [CLS]", add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)


# In[43]:


# Tokenize train_x, test_x, and eval_x

def encode(tweet):
    return tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length = 100,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors = 'pt',
    )

encoded_train_x = [encode(tweet+" [CLS]") for tweet in train_x[:,3]]
print(len(encoded_train_x))
encoded_train_x = torch.cat([ 
    torch.reshape(
        torch.cat((tweet['input_ids'], tweet['attention_mask'])), (1,2,-1)
    ) 
    for tweet 
    in encoded_train_x 
])
print(encoded_train_x.size())

encoded_test_x = [encode(tweet+" [CLS]") for tweet in test_x[:,3]]
print(len(encoded_test_x))
encoded_test_x = torch.cat([ 
    torch.reshape(
        torch.cat((tweet['input_ids'], tweet['attention_mask'])), (1,2,-1)
    ) 
    for tweet 
    in encoded_test_x 
])
print(encoded_test_x.size())

encoded_eval_x = [encode(tweet+" [CLS]") for tweet in eval_x[:,3]]
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


# In[39]:


# Store labels in tensors

train_y = torch.Tensor(train_y).long()
test_y = torch.Tensor(test_y).long()

print(train_y.dtype, test_y.dtype)


# In[7]:


# Define Custom Sequence Classification Model for GPT-2

class GPT2ForSequenceClassification(transformers.GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels 
        self.transformer = transformers.GPT2Model(config)
        
        model.config.hidden_dropout_prob = 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.n_embd, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        cls_token_ids=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        batch_size = cls_token_ids.shape[0]
        cls_states = torch.cat([torch.reshape(transformer_outputs[0][i, cls_token_ids[i], :], (1, -1)) for i in range(batch_size)])

        cls_states = self.dropout(cls_states)
        logits = self.classifier(cls_states)

        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits

model = GPT2ForSequenceClassification.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))


# In[6]:


a = {'t': 3}
b = a['c'] or 4
print(b)


# In[87]:


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
for batch in train_sampler:
    batch_input_ids = encoded_train_x[batch, 0, :]
    batch_attention_mask = encoded_train_x[batch, 1, :]
    batch_labels = train_y[batch]
    
    print(encoded_train_x[batch,:,:].size())
    print(batch_input_ids.size())
    print(batch_attention_mask.size())
    print(len(batch_labels))
    
    break


# In[88]:


# Move model and data to GPU if GPU is available

if torch.cuda.is_available():  
    print("cuda is available")
    cuda = torch.device('cuda')  
    model = model.to(cuda)
    encoded_train_x = encoded_train_x.to(cuda)
    train_y = train_y.to(cuda)
    encoded_test_x = encoded_test_x.to(cuda)
    test_y = test_y.to(cuda)
    encoded_eval_x = encoded_eval_x.to(cuda)


# In[90]:


# Training

model.train()

epochs = 4
for epoch in range(epochs):
    for batch in loading_bar(train_sampler, desc=f'(Current Epoch: {epoch})'):
        batch_input_ids = encoded_train_x[batch, 0, :]
        batch_attention_mask = encoded_train_x[batch, 1, :]
        batch_labels = train_y[batch]
        batch_cls_token_ids = (batch_input_ids==tokenizer.cls_token_id).nonzero(as_tuple=True)[1]
        
        optimizer.zero_grad()
        
        loss, logits = model(batch_input_ids, 
                             token_type_ids=None, 
                             attention_mask=batch_attention_mask,
                             labels=batch_labels,
                             cls_token_ids=batch_cls_token_ids)
        
        loss.backward()
        
        optimizer.step()
        
        


# In[91]:


# Testing

model.eval()

total_loss = 0
num_correct = 0
for batch in loading_bar(test_sampler):
    batch_input_ids = encoded_test_x[batch, 0, :]
    batch_attention_mask = encoded_test_x[batch, 1, :]
    batch_labels = test_y[batch]
    batch_cls_token_ids = (batch_input_ids==tokenizer.cls_token_id).nonzero(as_tuple=True)[1]
    
    with torch.no_grad():
        loss, logits = model(batch_input_ids, 
                             token_type_ids=None, 
                             attention_mask=batch_attention_mask,
                             labels=batch_labels,
                             cls_token_ids=batch_cls_token_ids)
    
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
    cls_token_id = (input_ids==tokenizer.cls_token_id).nonzero(as_tuple=True)[1]
    
    with torch.no_grad():
        logits = model(input_ids, 
                       token_type_ids=None, 
                       attention_mask=attention_mask,
                       cls_token_ids=cls_token_id)
    
    MAP = torch.argmax(logits[0]).item()
    predictions.append(MAP)


# In[ ]:


# Save Predictions

output = pd.DataFrame({'id': eval_x[:,0], 'target': predictions})
output.to_csv("submission.csv", index=False)

