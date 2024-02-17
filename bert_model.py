from torch import nn
import torch.nn.functional as F

from transformers import BertModel, AutoModel


class BertClassifier(nn.Module):

    def __init__(self, n_classes, freeze_bert=True, fine_tune=True):

        super(BertClassifier, self).__init__()
        # Instantiating BERT model object
        self.bert = BertModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT", return_dict=False)

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

            # If fine tuning bert, turn on only last layer.
            if fine_tune:
                for param in self.bert.encoder.layer[-1].parameters():
                    param.requires_grad = True

        # Linear layer to reduce bert output size 
        self.bert_output_reducer = nn.Linear(self.bert.config.hidden_size, 50)

        self.bert_drop_1 = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size,
                            self.bert.config.hidden_size)  # (768, 64)
        self.bn = nn.BatchNorm1d(768)  # (768)
        self.bert_drop_2 = nn.Dropout(0.25)
        self.out = nn.Linear(self.bert.config.hidden_size,
                             n_classes)  # (768,2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # TODO
        # Reduce the dimensionality of the BERT output
        # bert_output = self.bert_output_reducer(output)

        # Concatenate BERT output and static data for the fully connected layers
        # inputs = torch.cat([bert_output, x_static], dim=1)

        # only get first token 'cls'
        output = output[:, 0, :]
        # output = output.view(-1, 768)

        output = self.bert_drop_1(output)
        output = self.fc(output)
        output = self.bn(output)
        output = self.bert_drop_2(output)
        output = self.out(output)

        return F.log_softmax(output, dim=1)


class BertClassifier2(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()

        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")

        # BERT hidden state size is 768, class number is 2
        self.linear = nn.Linear(768, 2)

        # initialing weights and bias
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, input_ids, attention_mask):

        # get last_hidden_state
        vec, _ = self.bert(input_ids)

        # only get first token 'cls'
        vec = vec[:, 0, :]
        vec = vec.view(-1, 768)

        out = self.linear(vec)
        return F.log_softmax(out)
