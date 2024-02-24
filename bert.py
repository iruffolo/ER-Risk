import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
from transformers import (AutoTokenizer, get_linear_schedule_with_warmup)
from torch.utils.data import (TensorDataset, DataLoader,
                              RandomSampler, SequentialSampler)

from load_data.database import DatabaseLoader
from train import TrainingLoop
from bert_model import BertClassifier


def get_data_loader(data, tokenizer, sampler, static_features, batch_size=32):
    """
    Creates encoding using the provided tokenizer, converts to a torch
    tensor, and then creates a DataLoader object
    """

    # Create encoding from tokenizer
    encoding = tokenizer.batch_encode_plus(
        data['clean_notes'].values.tolist(),
        max_length=512,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_token_type_ids=True,
        truncation=True,
        padding='longest',
        return_attention_mask=True
    )

    # convert lists to tensors
    seq = torch.tensor(encoding['input_ids'])
    mask = torch.tensor(encoding['attention_mask'])
    token_ids = torch.tensor(encoding['token_type_ids'])
    static_data = torch.tensor(data[static_features].values)

    one_hot_cols = ['outcome_0', 'outcome_1', 'outcome_2', 'outcome_3']
    y_one_hot = torch.tensor(data[one_hot_cols].values)
    y = torch.tensor(data['OUTCOME'].values)

    tensor_dataset = TensorDataset(
        seq, mask, token_ids, y, y_one_hot, static_data)

    return DataLoader(tensor_dataset, batch_size=batch_size,
                      sampler=sampler(tensor_dataset),
                      pin_memory=True,
                      drop_last=True)


if __name__ == "__main__":

    # Set device GPU
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Set all the seed values
    seed_val = 42
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Get dataset from sqlite database
    database = DatabaseLoader("data/sqlite/db.sqlite")

    data = database.get_data()
    notes = database.get_notes()
    data['clean_notes'] = database.clean_notes(notes, "load_data/acronyms.txt")
    data['clean_notes'] = database.add_to_notes(data)

    enc = OneHotEncoder(handle_unknown='ignore')
    # data['labels'] = pd.DataFrame(enc.fit_transform(data['OUTCOME'].toarray()))
    labels = pd.get_dummies(data['OUTCOME'], prefix='outcome').astype(int)

    data = data.join(labels)

    data.dropna(inplace=True)

    # Test for binary classifier
    # data['OUTCOME'] = np.where(data['OUTCOME'] > 0, 1, 0)

    static_features = ["age", "systolic", "diastolic", "MAP", "pulse_pressure",
                       "TEMPERATURE", "PULSE", "RESP", "SpO2", "ACUITY"]

    print(f"{data[static_features]}")

    # sns.histplot(data['OUTCOME'], discrete=True)
    # plt.xticks([1, 2, 3])
    # plt.show()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT")

    # Create data split
    train_df, test_df = train_test_split(data, shuffle=True, train_size=0.80)
    train_df, val_df = train_test_split(
        train_df, shuffle=True, train_size=0.80)

    print('Train data:', train_df.shape)
    print('Val data:', val_df.shape)
    print('Test data:', test_df.shape)

    # Create pytorch dataloaders
    traindata = get_data_loader(train_df, tokenizer,
                                sampler=RandomSampler,
                                static_features=static_features)
    valdata = get_data_loader(val_df, tokenizer,
                              sampler=SequentialSampler,
                              static_features=static_features)
    testdata = get_data_loader(test_df, tokenizer,
                               sampler=SequentialSampler,
                               static_features=static_features)

    print('Number of batches in the train set', len(traindata))
    print('Number of batches in the validation set', len(valdata))
    print('Number of batches in the test set', len(testdata))

    # compute the class weights
    class_wts = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(data['OUTCOME'].values.tolist()),
        y=data['OUTCOME'])

    # convert class weights to tensor
    weights = torch.tensor(class_wts, dtype=torch.float)
    weights = weights.to(device)

    # loss function
    # loss_function = torch.nn.NLLLoss()
    cross_entropy = torch.nn.CrossEntropyLoss(weight=weights)

    class_names = np.unique(data['OUTCOME'])

    print('Downloading the pretrained BERT model...')
    model = BertClassifier(len(class_names),
                           static_size=len(static_features),
                           fine_tune=True)
    model.to(device)  # Model to GPU.

    # The pre-learned sections should have a smaller learning rate,
    # and the last total combined layer should be larger.
    # classifier = BertClassifier()
    # classifier.to(device)
    # optimizer = optim.Adam([
    #     {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    #     {'params': classifier.linear.parameters(), 'lr': 1e-4}
    # ])

    # optimizer parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    print('Preparing the optimizer...')
    learning_rate = 1e-3
    steps = 20

    optimizer = torch.optim.AdamW(optimizer_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=steps
    )

    # Train and Test model
    fn = "models/with_static_data_2.pt"
    tl = TrainingLoop(model, cross_entropy, optimizer, device, fn)
    # tl.train(traindata, valdata, epochs=20)
    predictions, true_y = tl.test(testdata, folds=1)

    # Accuracy and classification report
    acc = accuracy_score(true_y, predictions)
    cr = classification_report(true_y, predictions,
                               target_names=['label', 'predicted'],
                               labels=class_names)

    print(f'Accuracy: {acc}')
    print(f"Classifiction:\n {cr}")

    confusion_matrix = pd.crosstab(true_y, predictions,
                                   rownames=['True'], colnames=['Predicted'])

    sns.heatmap(confusion_matrix, annot=True)
    plt.show()
