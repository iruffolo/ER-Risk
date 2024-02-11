import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from transformers import (BertModel, AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)
from torch.utils.data import (TensorDataset, DataLoader,
                              RandomSampler, SequentialSampler)

from load_data.database import DatabaseLoader


class BERT_Arch(nn.Module):

    def __init__(self, n_classes, freeze_bert=True, fine_tune=True):

        super(BERT_Arch, self).__init__()
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

        self.bert_drop_1 = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size,
                            self.bert.config.hidden_size)  # (768, 64)
        self.bn = nn.BatchNorm1d(768)  # (768)
        self.bert_drop_2 = nn.Dropout(0.25)
        self.out = nn.Linear(self.bert.config.hidden_size,
                             n_classes)  # (768,2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output = self.bert_drop_1(output)
        output = self.fc(output)
        output = self.bn(output)
        output = self.bert_drop_2(output)
        output = self.out(output)
        return output


class BertClassifier(nn.Module):
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


class NotesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataloader = DatabaseLoader("data/sqlite/db.sqlite")

    notes = dataloader.get_notes()
    clean = dataloader.clean_notes(notes, "load_data/acronyms.txt")

    data = dataloader.get_data()
    data['clean_notes'] = clean

    data['binary_outcome'] = np.where(data['OUTCOME'] > 0, 1, 0)

    # plt.hist(data['OUTCOME'])
    # plt.show()

    # Load BERT models
    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT")

    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    print(model)

    # Check note length, drop notes above 512 (max bert input tokens)
    length = data['clean_notes'].map(tokenizer.encode).map(len)
    idx = length[length > 512].index
    data.drop(index=idx, inplace=True)

    # Create data split
    train_df, test_df = train_test_split(data, shuffle=True, train_size=0.80)
    train_df, val_df = train_test_split(
        train_df, shuffle=True, train_size=0.80)
    print('train data', train_df.shape)
    print('val data', val_df.shape)
    print('test data', test_df.shape)

    # text = list(train_df['clean_notes'])[0]
    # print("sample text: ", text)
    # tokens = tokenizer.encode(text, return_tensors='pt')
    # print(tokens)
    # print(tokenizer.convert_ids_to_tokens(tokens[0].tolist()))

    # def bert_tokenizer(text):
    #     return tokenizer.encode(text, return_tensors='pt',
    #                             padding=True)[0]

    train_tok = tokenizer(list(train_df['clean_notes'].values),
                          return_tensors='pt', padding=True)
    val_tok = tokenizer(list(val_df['clean_notes'].values),
                        return_tensors='pt', padding=True)
    test_tok = tokenizer(list(test_df['clean_notes'].values),
                         return_tensors='pt', padding=True)

    train_dataset = NotesDataset(train_tok, train_df['OUTCOME'])
    val_dataset = NotesDataset(val_tok, val_df['OUTCOME'])
    test_dataset = NotesDataset(test_tok, test_df['OUTCOME'])

    classifier = BertClassifier()
    classifier.to(device)

    # First, turn off the gradient for all parameters.
    for param in classifier.parameters():
        param.requires_grad = False

    # Second, turn on only last BERT layer.
    for param in classifier.bert.encoder.layer[-1].parameters():
        param.requires_grad = True

    # Finally, turn on classifier layer.
    for param in classifier.linear.parameters():
        param.requires_grad = True

    # The pre-learned sections should have a smaller learning rate,
    # and the last total combined layer should be larger.
    optimizer = optim.Adam([
        {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
        {'params': classifier.linear.parameters(), 'lr': 1e-4}
    ])

    # loss function
    loss_function = nn.NLLLoss()

    train_encoding = tokenizer.batch_encode_plus(
        list(train_df['clean_notes'].values),
        max_length=512,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_token_type_ids=True,
        truncation=True,
        padding='longest',
        return_attention_mask=True,
    )

    val_encoding = tokenizer.batch_encode_plus(
        list(val_df['clean_notes'].values),
        max_length=512,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_token_type_ids=True,
        truncation=True,
        padding='longest',
        return_attention_mask=True,
    )

    test_encoding = tokenizer.batch_encode_plus(
        list(test_df['clean_notes'].values),
        max_length=512,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_token_type_ids=True,
        truncation=True,
        padding='longest',
        return_attention_mask=True,
    )

    # compute the class weights
    class_wts = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(data['OUTCOME'].values.tolist()),
        y=data['OUTCOME'])

    # convert class weights to tensor
    weights = torch.tensor(class_wts, dtype=torch.float)
    weights = weights.to(device)

    # loss function
    cross_entropy = nn.CrossEntropyLoss(weight=weights)

    # convert lists to tensors
    train_seq = torch.tensor(train_encoding['input_ids'])
    train_mask = torch.tensor(train_encoding['attention_mask'])
    train_token_ids = torch.tensor(train_encoding['token_type_ids'])
    train_y = torch.tensor(train_df['OUTCOME'].tolist())

    val_seq = torch.tensor(val_encoding['input_ids'])
    val_mask = torch.tensor(val_encoding['attention_mask'])
    val_token_ids = torch.tensor(val_encoding['token_type_ids'])
    val_y = torch.tensor(val_df['OUTCOME'].tolist())

    test_seq = torch.tensor(test_encoding['input_ids'])
    test_mask = torch.tensor(test_encoding['attention_mask'])
    test_token_ids = torch.tensor(test_encoding['token_type_ids'])
    test_y = torch.tensor(test_df['OUTCOME'].tolist())

    BATCH_SIZE = 16
    learning_rate = 1e-3
    steps_per_epoch = 20

    # Wrap tensors and create data loaders
    train_data = TensorDataset(train_seq, train_mask, train_token_ids, train_y)
    traindata = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        sampler=RandomSampler(train_data),
        pin_memory=True,
        drop_last=True,
    )

    val_data = TensorDataset(val_seq, val_mask, val_token_ids, val_y)
    valdata = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        sampler=SequentialSampler(val_data),
        pin_memory=True,
        drop_last=True
    )

    test_data = TensorDataset(test_seq, test_mask, test_token_ids, test_y)
    testdata = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        sampler=SequentialSampler(test_data),
        pin_memory=True,
        drop_last=True
    )

    print('Number of data in the train set', len(traindata))
    print('Number of data in the validation set', len(valdata))
    print('Number of data in the test set', len(testdata))

    class_names = np.unique(data['OUTCOME'])
    print('Downloading the BERT custom model...')
    model = BERT_Arch(len(class_names))
    model.to(device)  # Model to GPU.

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
    optimizer = optim.AdamW(optimizer_parameters, lr=learning_rate)
    steps = steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=steps
    )

# function to train the bert model
    def trainBERT():

        print('Training...')
        model.train()
        total_loss = 0

        # empty list to save model predictions
        total_preds = []

        # iterate over batches
        for step, batch in enumerate(traindata):

            # progress update after every 50 batches.
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(
                    step, len(traindata)))

            # push the batch to gpu
            batch = [r.to(device) for r in batch]

            sent_id, mask, token_type_ids, labels = batch

            # clear previously calculated gradients
            model.zero_grad()

            # get model predictions for the current batch
            preds = model(sent_id, mask, token_type_ids)

            # compute loss
            loss = cross_entropy(preds, labels)

            # add on to the total loss
            total_loss = total_loss + loss.item()
            # backward pass to calculate the gradients
            loss.backward()

            # clip the the gradients to 1.0.
            # It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update parameters
            optimizer.step()
            # model predictions are stored on GPU. So, push it to CPU
            preds = preds.detach().cpu().numpy()
            # append the model predictions
            total_preds.append(preds)

            torch.cuda.empty_cache()

        # compute the training loss of the epoch
        avg_loss = total_loss / len(traindata)

        # predictions are in the form of
        # (# of batches, size of batch, # of classes).
        # reshape the predictions in form of (# of samples, # of classes)
        total_preds = np.concatenate(total_preds, axis=0)

        # returns the loss and predictions
        return avg_loss, total_preds

    # function for evaluating the model
    def evaluate():

        print("\nEvaluating...")

        model.eval()  # deactivate dropout layers
        total_loss = 0

        # empty list to save the model predictions
        total_preds = []

        # iterate over batches
        for step, batch in enumerate(valdata):
            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:
                # Report progress.
                print('\tBatch {:>5,}  of  {:>5,}.'.format(step, len(valdata)))

            # push the batch to gpu
            batch = [t.to(device) for t in batch]

            sent_id, mask, token_type_ids, labels = batch

            # deactivate autograd
            # Dont store any previous computations, thus freeing GPU space
            with torch.no_grad():

                # model predictions
                preds = model(sent_id, mask, token_type_ids)
                # compute the validation loss between actual and prediction
                loss = cross_entropy(preds, labels)
                total_loss = total_loss + loss.item()
                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)

            torch.cuda.empty_cache()
        # compute the validation loss of the epoch
        avg_loss = total_loss / len(valdata)
        # reshape the predictions in form of (# of samples, # of classes)
        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds

    best_valid_loss = float('inf')

    # Empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # for each epoch perform training and evaluation
    epochs = 10
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, _ = trainBERT()

        # evaluate model
        valid_loss, _ = evaluate()

        print('Evaluation done for epoch {}'.format(epoch + 1))
        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('Saving model...')
            # Save model weight's (you can also save it in .bin format)
            torch.save(model.state_dict(), 'bert_weights.pt')

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    print('\nTest Set...')

    test_preds = []

    print('Total batches:', len(testdata))

    for fold_index in range(0, 3):

        print('\nFold Model', fold_index)

        # Load the fold model
        path_model = 'bert_weights.pt'
        model.load_state_dict(torch.load(path_model))

        # Send the model to the GPU
        model.to(device)

        stacked_val_labels = []

        # Put the model in evaluation mode.
        model.eval()

        # Turn off the gradient calculations.
        # This tells the model not to compute or store gradients.
        # This step saves memory and speeds up validation.
        torch.set_grad_enabled(False)

        # Reset the total loss for this epoch.
        total_val_loss = 0

        for j, test_batch in enumerate(testdata):

            inference_status = 'Batch ' + str(j + 1)

            print(inference_status, end='\r')

            b_input_ids = test_batch[0].to(device)
            b_input_mask = test_batch[1].to(device)
            b_token_type_ids = test_batch[2].to(device)
            b_test_y = test_batch[3].to(device)

            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            token_type_ids=b_token_type_ids)

            # Get the preds
            preds = outputs[0]

            # Move preds to the CPU
            val_preds = preds.detach().cpu().numpy()

            # true_labels.append(b_test_y.to('cpu').numpy().flatten())

            # Stack the predictions.
            if j == 0:  # first batch
                stacked_val_preds = val_preds

            else:
                stacked_val_preds = np.vstack((stacked_val_preds, val_preds))

        test_preds.append(stacked_val_preds)

    print('\nPrediction complete.')
    print(len(test_preds))
    print(test_preds[:5])

    # Sum the predictions of all fold models
    for i, item in enumerate(test_preds):
        if i == 0:
            preds = item
        else:
            # Sum the matrices
            preds = item + preds

    # Average the predictions
    avg_preds = preds/(len(test_preds))

    # print(preds)
    # print()
    # print(avg_preds)

    # Take the argmax.
    # This returns the column index of the max value in each row.
    test_predictions = np.argmax(avg_preds, axis=1)

    # Take a look of the output
    print(type(test_predictions))
    print(len(test_predictions))
    print()
    print(test_predictions)

    true_y = []
    for j, test_batch in enumerate(testdata):
        true_y.append(int(test_batch[3][0].numpy().flatten()))
    print(true_y)

    # Accuracy and classification report
    target_names = ['true_y', 'predicted_y']

    data = {'true_y': true_y,
            'predicted_y': test_predictions}

    df_pred_BERT = pd.DataFrame(data, columns=['true_y', 'predicted_y'])

    confusion_matrix = pd.crosstab(df_pred_BERT['true_y'],
                                   df_pred_BERT['predicted_y'], rownames=[
                                   'True'], colnames=['Predicted'])

    print('Accuracy of BERT model', accuracy_score(true_y, test_predictions))
    print(classification_report(true_y,
                                test_predictions,
                                target_names=target_names,
                                labels=class_names))

    sns.heatmap(confusion_matrix, annot=True)
    plt.show()
