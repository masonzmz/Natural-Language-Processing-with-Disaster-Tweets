'''
Author: Mingzhe Zhang (s4566656)
Date: 2022-08-28 20:22:17
LastEditTime: 2022-09-27 22:53:23
FilePath: ./src/train.py
'''


import argparse
import random
import time
import warnings

import numpy as np
import pandas as pd
import scipy.io as scio
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Back, Fore, Style
from sklearn import metrics
from sklearn.metrics import f1_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (AdamW, AutoTokenizer, RobertaTokenizer,
                          get_linear_schedule_with_warmup)

from data_loader import data_preprocess
from plot import evaluate_roc
from premodel import BertClassifier, RobertaClassifier

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=7203)
parser.add_argument('--model_name', type=str, default='roberta_base')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--epoch', default=3, type=int)
parser.add_argument('--threshold', default=0.5, type=float, help='Threshold for classification.')
parser.add_argument('--k', default=10, type=int)
parser.add_argument('--batchsize', default=8, type=int)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--layer', default=1, type=int)

args = None

# Set the max length of each sentence.
MAX_LENGTH = 128


'''
description: Set the random seed
param {int} seed_value
return {*}
'''
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


'''
description: get the corresponding token id and attention masks of RoBERTa
param {tensor} data
param {*} tokenizer corresponding tokenizer
return {*}
'''
def preprocessing_for_roberta(data, tokenizer):
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=MAX_LENGTH,            
            pad_to_max_length=True,        
            return_attention_mask=True 
        )
        
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


'''
description: get the corresponding token id and attention masks of BERT
param {tensor} data
param {*} tokenizer corresponding tokenizer
return {*}
'''
def preprocessing_for_bert(data, tokenizer):
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            sent,
            padding='max_length',           
            max_length=MAX_LENGTH,        
            truncation=True 
        )
        
        input_ids.append(encoded_sent["input_ids"])
        attention_masks.append(encoded_sent["attention_mask"])
        token_type_ids.append(encoded_sent["token_type_ids"])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

    return input_ids, attention_masks, token_type_ids


'''
description: training function
param {*} model
param {*} train_dataloader
param {*} loss_fn
param {*} optimizer
param {*} scheduler
param {*} device
param {*} index
param {*} val_dataloader
param {*} epochs
param {*} evaluation
return {*}
'''
def train(model, train_dataloader, loss_fn, optimizer, scheduler, device, index, val_dataloader=None, epochs=5, evaluation=False):
    for epoch_i in range(epochs):

        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1

            if args.model_name == 'roberta_base':
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
                model.zero_grad()
                logits = model(b_input_ids, b_attn_mask)
                loss = loss_fn(logits, b_labels.float().view(-1, 1))
                batch_loss += loss.item()
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

            elif args.model_name == 'bert_base':
                b_input_ids, b_attn_mask, b_token_ids, b_labels = tuple(t.to(device) for t in batch)
                model.zero_grad()
                logits = model(b_input_ids, b_attn_mask, b_token_ids)
                loss = loss_fn(logits, b_labels.float().view(-1, 1))
                batch_loss += loss.item()
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)

        if evaluation == True:
            val_loss, val_accuracy, f1 = evaluate(model, val_dataloader, loss_fn, device)
            time_elapsed = time.time() - t0_epoch
            # if epoch_i + 1 == 3:
                # print(Fore.RED + f"{index + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {f1:^9.2f} | {time_elapsed:^9.2f}")
                # print(Fore.BLUE + "-"*70)
            
    if evaluation == True:       
        return val_accuracy


'''
description: evaluation method
param {*} model
param {*} val_dataloader
param {*} loss_fn
param {*} device
return {*}
'''
def evaluate(model, val_dataloader, loss_fn, device):
    model.eval()

    val_accuracy = []
    val_loss = []
    val_f1 = []

    for batch in val_dataloader:

        if args.model_name == 'roberta_base':
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
        elif args.model_name == 'bert_base':
            b_input_ids, b_attn_mask, b_token_ids, b_labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask, b_token_ids)

        loss = loss_fn(logits, b_labels.float().view(-1, 1))
        val_loss.append(loss.item())

        logits = torch.sigmoid(logits).cpu().detach()
        preds = logits.flatten()

        array_preds = np.where(preds.cpu().numpy() > args.threshold, 1, 0)
        
        accuracy = (array_preds == b_labels.cpu().numpy()).mean() * 100
        
        val_f1.append(f1_score(b_labels.cpu().numpy(), array_preds))
        val_accuracy.append(accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    f1 = np.mean(val_f1)

    return val_loss, val_accuracy, f1


'''
description: Prediction function
param {*} model
param {*} test_dataloader
param {*} device
return {*}
'''
def predict(model, test_dataloader, device):
    model.eval()

    all_logits = []
    
    for batch in test_dataloader:
        if args.model_name == 'roberta_base':
            b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
        elif args.model_name == 'bert_base':
            b_input_ids, b_attn_mask, b_token_ids = tuple(t.to(device) for t in batch)[:3]
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask, b_token_ids)
        
        all_logits.append(logits)
    
    all_logits = torch.cat(all_logits, dim=0)
    probs = all_logits.cpu().numpy()

    return probs


'''
description: Initiallize the model, including optimizer and scheduler.
param {float} lr
param {float} eps
param {*} device
param {string} model_name
param {int} num_train_steps
return {*}
'''
def initialize_model(lr, eps, device, model_name, num_train_steps):
    if model_name == 'roberta_base':
        classifier = RobertaClassifier(args.dropout, args.layer)
    elif model_name == 'bert_base':
        classifier = BertClassifier(args.dropout, args.layer)
    classifier.to(device)

    optimizer = AdamW(classifier.parameters(), lr=lr, eps=eps)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_train_steps)
    return classifier, optimizer, scheduler


'''
description: Split the original data into k fold. Initially, k is equal to 10.
param {int} k
param {int} i
param {tensor} data_inputs
param {tensor} data_masks
param {tensor} data_token_ids
param {tensor} y_data
return {*} Data for the cross validation
'''
def get_k_fold_data(k, i, data_inputs, data_masks, data_token_ids, y_data): 
    
    assert k > 1
    fold_size = y_data.shape[0] // k 

    if args.model_name == 'roberta_base':
        x_train_inputs, x_train_masks, y_train = None, None, None
        
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size) 
            x_part_inputs, x_part_masks, y_part = data_inputs[idx, :], data_masks[idx, :], y_data[idx]
            
            if j == i:
                x_val_inputs, x_val_masks, y_val = x_part_inputs, x_part_masks, y_part
            elif x_train_inputs is None:
                x_train_inputs, x_train_masks, y_train = x_part_inputs, x_part_masks, y_part
            else:
                x_train_inputs = torch.cat((x_train_inputs, x_part_inputs), dim=0) 
                x_train_masks = torch.cat((x_train_masks, x_part_masks), dim=0)
                y_train = np.concatenate((y_train, y_part), axis=0)
            
        return x_train_inputs, x_train_masks, x_val_inputs, x_val_masks, y_train, y_val
    
    elif args.model_name == 'bert_base':
        x_train_inputs, x_train_masks, x_train_token_ids, y_train = None, None, None, None
        
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size) 
            x_part_inputs, x_part_masks, x_part_token_ids, y_part = data_inputs[idx, :], data_masks[idx, :], data_token_ids[idx, :], y_data[idx]
            
            if j == i:
                x_val_inputs, x_val_masks, x_val_token_ids, y_val = x_part_inputs, x_part_masks, x_part_token_ids, y_part
            elif x_train_inputs is None:
                x_train_inputs, x_train_masks, x_train_token_ids, y_train = x_part_inputs, x_part_masks, x_part_token_ids, y_part
            else:
                x_train_inputs = torch.cat((x_train_inputs, x_part_inputs), dim=0) 
                x_train_masks = torch.cat((x_train_masks, x_part_masks), dim=0)
                x_train_token_ids = torch.cat((x_train_token_ids, x_part_token_ids), dim=0)
                y_train = np.concatenate((y_train, y_part), axis=0)
            
        return x_train_inputs, x_train_masks, x_train_token_ids, x_val_inputs, x_val_masks, x_val_token_ids, y_train, y_val


'''
description: Main function
return {*}
'''
def main():
    global args

    # Print param information
    args = parser.parse_args()
    print(Fore.BLUE + f"{args}")
    print("\n")

    # Set the device.
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k = args.k

    # Set random seed
    set_seed(args.seed)

    batch_size = args.batchsize
    learning_rate = 3e-5
    epsilon = 1e-08

    # Data pro-processing
    x_data, y_data, data_test = data_preprocess()

    # Define the tokenizer depends on different model
    if args.model_name == 'roberta_base':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer.pad_token = tokenizer.eos_token
        data_inputs, data_masks = preprocessing_for_roberta(x_data, tokenizer)
    elif args.model_name == 'bert_base':
        tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        data_inputs, data_masks, data_token_ids = preprocessing_for_bert(x_data, tokenizer)

    radio = (k - 1) / k


    num_train_steps = int((radio * len(x_data)) / args.batchsize * args.epoch)

    classifier, optimizer, scheduler = initialize_model(learning_rate, epsilon, device, args.model_name, num_train_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    # print(Fore.RED + f"{'k-fold':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'F1 Score':^9} | {'Time':^9}")
    # print(Fore.RED + "-"*70)

    folds_accuracy = []

    for i in range(k):
        if args.model_name == 'roberta_base':
            x_train_inputs, x_train_masks, x_val_inputs, x_val_masks, y_train, y_val = get_k_fold_data(k, i, data_inputs, data_masks, None, y_data)
            train_labels = torch.tensor(y_train)
            val_labels = torch.tensor(y_val)
            
            train_data = TensorDataset(x_train_inputs, x_train_masks, train_labels)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
            
            val_data = TensorDataset(x_val_inputs, x_val_masks, val_labels)
            val_sampler = SequentialSampler(val_data)
            val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
        
        elif args.model_name == 'bert_base':
            x_train_inputs, x_train_masks, x_train_token_ids, x_val_inputs, x_val_masks, x_val_token_ids, y_train, y_val = get_k_fold_data(k, i, data_inputs, data_masks, data_token_ids, y_data)
            train_labels = torch.tensor(y_train)
            val_labels = torch.tensor(y_val)
            
            train_data = TensorDataset(x_train_inputs, x_train_masks, x_train_token_ids, train_labels)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
            
            val_data = TensorDataset(x_val_inputs, x_val_masks, x_val_token_ids, val_labels)
            val_sampler = SequentialSampler(val_data)
            val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
        
        fold_i_acc = train(classifier, train_dataloader, loss_fn, optimizer, scheduler, device, i, val_dataloader, epochs=args.epoch, evaluation=True)
        folds_accuracy.append(fold_i_acc)
    
    folds_accuracy = np.array(folds_accuracy)
    # print("%0.2f accuracy with a standard deviation of %0.2f" % (folds_accuracy.mean(), folds_accuracy.std()))
    # print(Fore.RED + "-"*70)

    # Concatenate the training data and the validation data
    full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])
    full_train_sampler = RandomSampler(full_train_data)
    full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size=batch_size)

    train(classifier, full_train_dataloader, loss_fn, optimizer, scheduler, device, i, epochs=args.epoch)

    print(Fore.RED + 'Tokenizing training and valid data...')
    print(Fore.RED + "-"*70)
    
    if args.model_name == 'roberta_base':
        test_inputs, test_masks = preprocessing_for_roberta(data_test['text'], tokenizer)

        test_dataset = TensorDataset(test_inputs, test_masks)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

        probs = predict(classifier, test_dataloader, device)

    elif args.model_name == 'bert_base':
        test_inputs, test_masks, x_test_token_ids = preprocessing_for_bert(data_test['text'], tokenizer)

        test_dataset = TensorDataset(test_inputs, test_masks, x_test_token_ids)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)
        
        probs = predict(classifier, test_dataloader, device)

    print(probs)
    predictions = np.where(probs > args.threshold, 1, 0)
    print(Fore.RED + f"Number of predictions - non-negative: {predictions.sum()}")
    print(Fore.RED + "-"*70)

    print(Fore.GREEN + "Saving results to csv...")
    
    submission = pd.read_csv('./data/sample_submission.csv')
    submission['target'] = list(map(int, predictions))

    original_predictions = pd.read_csv("./data/perfect_submission.csv")
    final_accuracy = metrics.accuracy_score(submission.target.values, original_predictions.target.values)
    print(Fore.RED + f"Final accuracy is: {final_accuracy}")

    savename = f'./results/result_{args.model_name}_{args.threshold}_{args.batchsize}_{args.dropout}_{args.layer}.mat'
    scio.savemat(savename,
                {'accuracy':final_accuracy,
                'model_name':args.model_name,
                'threshold': args.threshold,
                'batchsize': args.batchsize,
                'dropout': args.dropout,
                'layer': args.layer})

    submission.to_csv('./results/submission.csv', index=False)


if __name__ == "__main__":
    main()
