from random import betavariate
from re import S
import re
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx.symbolic_opset9 import dim, unsqueeze
from transformers import BertModel,RobertaModel

from bert import BertPreTrainedModel,RobertaPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable


class RobertaStance(RobertaPreTrainedModel):
    def __init__(self, roberta_config):
        """
        :param bert_config: configuration for bert model
        """
        super(RobertaStance, self).__init__(roberta_config)
        self.roberta_config = roberta_config
        self.roberta = RobertaModel(roberta_config)
        penultimate_hidden_size = roberta_config.hidden_size*2
        self.g1 = nn.Linear(100, 768)
        self.g1_drop = nn.Dropout(0.1)
        #decode
        self.d1 = nn.Linear(768, 768)
        self.d2 = nn.Linear(768, 100)
        self.d_drop = nn.Dropout(0.1)
        self.sent_loss = CrossEntropyLoss()
        self.sent_cls = nn.Linear(penultimate_hidden_size, roberta_config.sent_number)

    def encode2(self, x2):
        x2 = self.g1(x2)
        x2 = F.relu(x2)
        x2 = self.g1_drop(x2)
        return x2

    def decode(self, z):
        z = self.d1(z)
        z = F.relu(z)
        z = self.d_drop(z)
        z = self.d2(z)
        return z 

    def loss_ae(self,recon_x, x):
        dim = x.size(1)
        MSE = F.mse_loss(recon_x, x.view(-1, dim), reduction='mean')
        return MSE

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sent_labels=None,
                position_ids=None, graph_feature =None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        hidden = outputs[0]
        w1 = hidden[:,0,:]


        z2 = self.encode2(graph_feature)
        reconstruct = self.decode(z2)
        f = torch.cat((w1,z2),dim = 1)

        recon_loss = self.loss_ae(reconstruct,graph_feature)

        sent_logits = self.sent_cls(f)

        if len(sent_logits.shape) == 1:
            sent_logits = sent_logits.unsqueeze(0)
        sent_loss = self.sent_loss(sent_logits, sent_labels)
        loss =sent_loss+recon_loss

        return sent_logits,loss

class BertStance(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        :param bert_config: configuration for bert model
        """
        super(BertStance, self).__init__(bert_config)
        self.bert_config = bert_config
        self.bert = BertModel(bert_config)

        penultimate_hidden_size = bert_config.hidden_size*2

        #encode
        self.g1 = nn.Linear(100, 768)
        self.g1_drop = nn.Dropout(0.1)
        #decode
        self.d1 = nn.Linear(768, 768)
        self.d2 = nn.Linear(768, 100)
        self.d_drop = nn.Dropout(0.1)
        self.sent_loss = CrossEntropyLoss()
        self.sent_cls = nn.Linear(penultimate_hidden_size, bert_config.sent_number)

    def encode2(self, x2):
        x2 = self.g1(x2)
        x2 = F.relu(x2)
        x2 = self.g1_drop(x2)
        return x2

    def decode(self, z):
        z = self.d1(z)
        z = F.relu(z)
        z = self.d_drop(z)
        z = self.d2(z)
        return z 

    def loss_ae(self,recon_x, x):
        dim = x.size(1)
        MSE = F.mse_loss(recon_x, x.view(-1, dim), reduction='mean')
        return MSE

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sent_labels=None,
                position_ids=None, head_mask=None,graph_feature =None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        hidden = outputs[0]
        w1 = hidden[:,0,:]


        z2 = self.encode2(graph_feature)
        reconstruct = self.decode(z2)
        f = torch.cat((w1,z2),dim = 1)

        recon_loss = self.loss_ae(reconstruct,graph_feature)

        sent_logits = self.sent_cls(f)

        if len(sent_logits.shape) == 1:
            sent_logits = sent_logits.unsqueeze(0)
        sent_loss = self.sent_loss(sent_logits, sent_labels)
        loss =sent_loss+recon_loss

        return sent_logits,loss

class LSTMStance(nn.Module):
    def __init__(self, embedding_dim,hidden_dim,sent_size,dropout=0.5):
        """

        :param bert_config: configuration for bert model
        """
        super(LSTMStance, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True,dropout=dropout)
        self.sent_loss = CrossEntropyLoss()
        self.sent_cls = nn.Linear(hidden_dim*2+100, sent_size)

    def forward(self, embedding=None, sent_labels=None, SentEmb = None, attention_mask=None,graph_features = None,meg='Encode'):
        hidden, n = self.lstm(embedding)
        h = torch.bmm(torch.transpose(hidden,1,2),attention_mask.unsqueeze(2).to(torch.float32)).squeeze()
        if meg == 'Encode':
            return h
        if len(SentEmb.shape) == 1:
                SentEmb = SentEmb.unsqueeze(0)
                h = h.unsqueeze(0)
        SentEmb = torch.cat((SentEmb,h),dim=1)
        SentEmb = torch.cat((SentEmb,graph_features),dim=1)
        sent_logits = self.sent_cls(SentEmb)
       
        sent_loss = self.sent_loss(sent_logits, sent_labels)
        return sent_logits,sent_loss