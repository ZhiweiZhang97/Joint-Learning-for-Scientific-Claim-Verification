import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from transformers import AutoModel


class BertCNN(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a CNN layer on top of
    the pooled output.
    """
    def __init__(self, args, num_labels=103):
        super(BertCNN, self).__init__()
        self.num_labels = num_labels
        self.num_channels = args.hidden_size//3
        self.bert = AutoModel.from_pretrained(args.model)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(2, args.hidden_size))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(3, args.hidden_size))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(4, args.hidden_size))

        self.pool1 = nn.MaxPool1d(kernel_size=args.max_position_embeddings - 2 + 1)
        self.pool2 = nn.MaxPool1d(kernel_size=args.max_position_embeddings - 3 + 1)
        self.pool3 = nn.MaxPool1d(kernel_size=args.max_position_embeddings - 4 + 1)

        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.classifier = nn.Linear(args.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, encoded_dict, labels=None):
        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        bert_out = self.bert(**encoded_dict)[0]
        pooled_output = bert_out.unsqueeze(1)

        h1 = self.conv1(pooled_output)
        h2 = self.conv2(pooled_output)
        h3 = self.conv3(pooled_output)

        h1 = self.pool1(h1.squeeze())
        h2 = self.pool2(h2.squeeze())
        h3 = self.pool3(h3.squeeze())

        pooled_output = torch.cat([h1, h2, h3], 1).squeeze()

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
