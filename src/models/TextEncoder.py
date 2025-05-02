import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from ..models.classifier import Classifier

# bert base model from https://huggingface.co/bert-base-uncased
# bert large model from https://huggingface.co/bert-large-uncased


class BertTextClf(nn.Module):
    def __init__(self, args):
        super(BertTextClf, self).__init__()
        self.text_encoder = BertModel.from_pretrained(args.bert_model)
        self.clf = Classifier(dropout=args.dropout, in_dim=768, post_dim=256, out_dim=args.n_classes)

    def forward(self, txt, mask, segment):
        x = self.text_encoder(txt, token_type_ids=segment,attention_mask=mask,).last_hidden_state[:, 0, :]
        return self.clf(x)

    # def forward(self, text):
    #     """
    #     text: (batch_size, 3, seq_len)
    #     3: input_ids, input_mask, segment_ids
    #     input_ids: input_ids,
    #     input_mask: attention_mask,
    #     segment_ids: token_type_ids
    #     """
    #     input_ids = torch.squeeze(text[0], 1)
    #     input_mask = torch.squeeze(text[2], 1)
    #     segment_ids = torch.squeeze(text[1], 1)
    #     # input_ids, input_mask, segment_ids = input_ids, attention_mask, token_type_ids
    #     last_hidden_states = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]
    #
    #     return last_hidden_states


# if __name__ == "__main__":
#     text_normal = TextEncoder()
