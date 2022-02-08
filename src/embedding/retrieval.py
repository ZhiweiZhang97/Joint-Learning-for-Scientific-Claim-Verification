import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AutoModel


class TimeDistributed(nn.Module):
    def __init__(self, input_size, output_size, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = nn.Linear(input_size, output_size, bias=True)
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # reshape input data --> (samples * timesteps, input_size)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class AbstractAttention(nn.Module):
    """
    word-level attention. abstract representation for abstract retrieval.
    """

    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super(AbstractAttention, self).__init__()
        self.dense = TimeDistributed(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = ClassificationHead(hidden_size, num_labels, dropout=dropout)
        self.hidden_size = hidden_size
        self.att_scorer = TimeDistributed(hidden_size, 1)
        # self.lstm = nn.LSTM(hidden_size, self.hidden_size // 2, 1, dropout=dropout, bidirectional=True)

    def forward(self, x, token_mask, claim_reps):
        att_s = self.dropout(x.view(-1, x.size(-1)))
        att_s = self.dense(att_s)
        att_s = self.dropout(torch.tanh(att_s))  # [batch_size * num_sentence * num_token, hidden_dim]  # nan
        raw_att_scores = self.att_scorer(att_s).squeeze(-1).view(x.size(0), x.size(1),
                                                                 x.size(2))  # [batch_size, num_sentence, num_token]
        u_w = raw_att_scores.masked_fill((1 - token_mask).bool(), float('-inf'))
        att_scores = torch.softmax(u_w, dim=-1)
        att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores),
                                 att_scores)  # replace NaN with 0
        # batch_att_scores: word attention scores. [batch_size * num_sentence, num_token]
        word_att_scores = att_scores.view(-1, att_scores.size(-1))  # word attention weight matrix.
        # out:  # sentence_representations. [batch_size, num_sentence, hidden_dim]
        out = torch.bmm(word_att_scores.unsqueeze(1), x.view(-1, x.size(2), x.size(3))).squeeze(1)
        sentence_reps = out.view(x.size(0), x.size(1), x.size(-1))  # [batch_size, num_sentence, hidden_dim]
        # sentence_reps = torch.mul(claim_reps.unsqueeze(1), sentence_reps)
        sentence_mask = token_mask[:, :, 0]

        sentence_mask = torch.logical_and(sentence_mask, sentence_mask)
        # h_0 = Variable(torch.zeros(2, sentence_reps.size(1), self.hidden_size // 2).cuda())
        # c_0 = Variable(torch.zeros(2, sentence_reps.size(1), self.hidden_size // 2).cuda())
        # print(sentence_reps.shape)
        sentence_embedding = self.dropout(sentence_reps)
        # sentence_embedding, (_, _) = self.lstm(sent_embeddings, (h_0, c_0))
        att_s = self.dense(sentence_embedding)  # [batch_size, num_sentence, hidden_size,]
        u_i = self.dropout(torch.tanh(att_s))  # u_i = tanh(W_s * h_i + b). [batch_size, num_sentence, hidden_size,]
        u_w = self.att_scorer(u_i).squeeze(-1).view(sentence_reps.size(0),
                                                    sentence_reps.size(1))  # [batch_size, num_sentence]
        u_w = u_w.masked_fill((~sentence_mask).bool(), -1e4)
        # sentence_att_scores: sentence attention scores. [batch_size, num_sentence]
        # print('u_w: ', u_w)
        sentence_att_scores = torch.softmax(u_w, dim=-1)
        # result: abstract representations. [batch_size, hidden_dim]
        paragraph_reps = torch.bmm(sentence_att_scores.unsqueeze(1), sentence_reps).squeeze(1)
        claim_paragraph = torch.mul(claim_reps, paragraph_reps)
        output = self.classifier(claim_paragraph)
        sentence_att_scores = sentence_att_scores[:, range(1, sentence_att_scores.shape[1])]
        sentence_att_scores = torch.softmax(sentence_att_scores, dim=-1)
        # print('abstract_sentence_reps: ', sentence_reps)
        return output, sentence_att_scores


class ClassificationHead(nn.Module):
    """ Head for sentence-level classification tasks. """

    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.dense = TimeDistributed(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = TimeDistributed(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class SelfAttentionNetwork(nn.Module):

    def __init__(self, hidden_dim, dropout=0.1):
        super(SelfAttentionNetwork, self).__init__()
        self.dense = TimeDistributed(hidden_dim, hidden_dim)
        self.att_scorer = TimeDistributed(hidden_dim, 1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, token_mask):
        att_s = self.dropout_layer(x)
        att_s = self.dense(att_s)
        u_i = self.dropout_layer(torch.tanh(att_s))
        u_w = self.att_scorer(u_i).squeeze(-1).view(x.size(0), x.size(1))
        u_w = u_w.masked_fill((1 - token_mask).bool(), float('-inf'))
        att_scores = torch.softmax(u_w, dim=-1)
        att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores),
                                 att_scores)
        out = torch.bmm(att_scores.unsqueeze(1), x).squeeze(1)
        return out


class JointModelRetrieval(nn.Module):
    def __init__(self, args):
        super(JointModelRetrieval, self).__init__()
        self.num_abstract_label = 3
        self.num_rationale_label = 2
        self.sim_label = 2
        self.bert = AutoModel.from_pretrained(args.model)
        self.retrieval_criterion = nn.CrossEntropyLoss()
        # self.abstract_criterion = MultiFocalLoss(3, alpha=[0.1, 0.6, 0.3])
        # self.rationale_criterion = FocalLoss(weight=torch.tensor([0.25, 0.75]), reduction='mean')
        # self.retrieval_criterion = FocalLoss(weight=torch.tensor([0.25, 0.75]), reduction='mean')
        self.dropout = args.dropout
        self.hidden_dim = args.hidden_dim
        self.abstract_retrieval = AbstractAttention(self.hidden_dim, self.sim_label, dropout=self.dropout)
        self.self_attention = SelfAttentionNetwork(self.hidden_dim, dropout=self.dropout)

        self.extra_modules = [
            self.retrieval_criterion,
            self.self_attention,
            # self.abstract_retrieval
        ]

    def reinitialize(self):
        self.abstract_retrieval = AbstractAttention(self.hidden_dim, self.sim_label, dropout=self.dropout)
        self.self_attention = SelfAttentionNetwork(self.hidden_dim, dropout=self.dropout)
        self.extra_modules = [
            self.retrieval_criterion,
            self.self_attention,
            # self.abstract_retrieval
        ]

    def forward(self, encoded_dict, transformation_indices, retrieval_label=None):
        batch_indices, indices_by_batch, mask = transformation_indices
        # match_batch_indices, match_indices_by_batch, match_mask = match_indices
        # (batch_size, num_sep, num_token)
        # print(encoded_dict['input_ids'].shape, batch_indices.shape, indices_by_batch.shape, mask.shape)
        bert_out = self.bert(**encoded_dict)[0]  # [batch_size, sequence_len, hidden_dim]

        title_abstract_token = range(1, batch_indices.shape[1])
        title_abstract_tokens = bert_out[batch_indices[:, title_abstract_token, :],
                                indices_by_batch[:, title_abstract_token, :], :]
        title_abstract_mask = mask[:, title_abstract_token, :]

        claim_token = bert_out[batch_indices[:, 0, :], indices_by_batch[:, 0, :], :]
        claim_mask = mask[:, 0, :]
        claim_representation = self.self_attention(claim_token, claim_mask)

        abstract_retrieval, sentence_att_scores = self.abstract_retrieval(title_abstract_tokens,
                                                                          title_abstract_mask, claim_representation)

        retrieval_out = torch.argmax(abstract_retrieval.cpu(), dim=-1).detach().numpy().tolist()
        retrieval_loss = self.retrieval_criterion(abstract_retrieval,
                                                  retrieval_label) if retrieval_label is not None else None

        if retrieval_label == None:
            return retrieval_out
        return abstract_retrieval, retrieval_loss
