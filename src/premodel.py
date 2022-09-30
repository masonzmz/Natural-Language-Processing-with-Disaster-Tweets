'''
Author       : Mingzhe Zhang (s4566656)
Date         : 2022-08-28 20:22:12
LastEditTime : 2022-09-29 11:27:05
FilePath     : /s4566656/anaconda3/envs/mason/Kaggle_Disaster/src/premodel.py
'''

import torch.nn as nn

from transformers import RobertaModel, AutoModel
from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

class RobertaClassifier(nn.Module):
    def __init__(self, dropout, layer=2):
        super(RobertaClassifier, self).__init__()
        self.dropout = dropout
        self.layer = layer
        H1, H2, H3, H4 = 768, 256, 64, 1
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.drop = nn.Dropout(dropout)

        if self.layer == 1:
            self.net = nn.Linear(H1, H4)
        elif self.layer == 2:
            self.net = nn.Sequential(
                           nn.Linear(H1, H3),
                           nn.ReLU(),
                           nn.Linear(H3, H4), 
                       )
        elif self.layer == 3:
            self.net = nn.Sequential(
                           nn.Linear(H1, H2),
                           nn.ReLU(),
                           nn.Linear(H2, H3),
                           nn.ReLU(),
                           nn.Linear(H3, H4),
                       )

    def forward(self, input_ids, attention_mask):
        x = self.roberta(input_ids=input_ids,
                         attention_mask=attention_mask)
        x = x[0][:, 0, :]
        x = self.drop(x)
        x = self.net(x)
        return x


class BertClassifier(nn.Module):
    def __init__(self, dropout, layer=2):
        super(BertClassifier, self).__init__()
        self.dropout = dropout
        self.layer = layer
        H1, H2, H3, H4 = 768, 256, 64, 1
        self.bert = AutoModel.from_pretrained('vinai/bertweet-base')
        self.drop = nn.Dropout(dropout)

        if self.layer == 1:
            self.net = nn.Linear(H1, H4)
        elif self.layer == 2:
            self.net = nn.Sequential(
                           nn.Linear(H1, H3),
                           nn.ReLU(),
                           nn.Linear(H3, H4)
                       )
        elif self.layer == 3:
            self.net = nn.Sequential(
                           nn.Linear(H1, H2),
                           nn.ReLU(),
                           nn.Linear(H2, H3),
                           nn.ReLU(),
                           nn.Linear(H3, H4)
                       )

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.bert(input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids)
        x = self.drop(x["pooler_output"])
        x = self.net(x)
        return x



