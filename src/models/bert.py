#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel

import transformers

class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model)
        # self.bert = transformers.BertForSequenceClassification.from_pretrained("huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli")

    def forward(self, txt, mask, segment):
        _, out = self.bert(
            txt,
            token_type_ids=segment,
            attention_mask=mask,
            output_all_encoded_layers=False,
        )
        return out

class BertClf(nn.Module):
    def __init__(self, args):
        super(BertClf, self).__init__()
        self.args = args
        self.text_encoder = BertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)
        self.clf.apply(self.text_encoder.bert.init_bert_weights)

    def forward(self, txt, mask, segment):
        x = self.text_encoder(txt, mask, segment)
        return self.clf(x)


class PruneBertClf(nn.Module):
    def __init__(self, args):
        super(PruneBertClf, self).__init__()
        self.bert = transformers.BertForSequenceClassification.from_pretrained("huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli").bert
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, txt, mask, segment):
        x = self.bert(txt, token_type_ids=segment,attention_mask=mask,).last_hidden_state[:, 0, :]
        return self.clf(x)

