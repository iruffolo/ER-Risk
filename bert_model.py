from torch import nn, cat
import torch.nn.functional as F

from transformers import BertModel, AutoModel


class BertClassifier(nn.Module):

    def __init__(self, n_classes, static_size=10,
                 freeze_bert=True, fine_tune=True):

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

        reduced_size = 50
        linear_size_1 = 256
        linear_size_2 = 64

        bert_size = self.bert.config.hidden_size

        # Linear layer to reduce bert output size
        self.bert_output_reducer = nn.Linear(bert_size, reduced_size)

        # self.linear1 = nn.Linear(reduced_size + static_size, linear_size_1)
        self.linear1 = nn.Linear(static_size, linear_size_1)
        self.bn1 = nn.BatchNorm1d(linear_size_1)
        self.drop1 = nn.Dropout(0.3)

        self.linear2 = nn.Linear(linear_size_1, linear_size_2)
        self.bn2 = nn.BatchNorm1d(linear_size_2)
        self.drop2 = nn.Dropout(0.25)

        self.out = nn.Linear(linear_size_2, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids, static_data):
        # bert_output, _ = self.bert(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )

        # only get first token 'cls'
        # bert_output = bert_output[:, 0, :]
        # output = output.view(-1, 768)

        # reduce bert output size
        # bert_output = self.bert_output_reducer(bert_output)

        # Stack bert output with tabular data
        # inputs = cat([bert_output, static_data], dim=1).float()

        # First fully connected layer
        output = F.relu(self.linear1(static_data.float()))
        output = self.bn1(output)
        output = self.drop1(output)

        # Second fully connected layer
        output = F.relu(self.linear2(output))
        output = self.bn2(output)
        output = self.drop2(output)

        # Classifier
        output = self.out(output)

        return output
        # return F.softmax(output, dim=1)


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
