import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F


class TrainingLoop:

    def __init__(self, model, loss_fn, optimizer, device,
                 fn="models/bert_weights.pt"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        self.best_valid_loss = float('inf')

        # Empty lists to store training and validation loss of each epoch
        self.train_losses = list()
        self.valid_losses = list()

        self.filename = fn

    def _train_bert(self, data):
        """
        Training loop - on training data set
        """

        print('Training...')
        self.model.train()
        total_loss = 0

        # empty list to save model predictions
        total_preds = []

        # iterate over batches
        for step, batch in enumerate(data):

            # progress update after every 50 batches.
            if step % 50 == 0 and not step == 0:
                print(f"\tBatch {step} of {len(data)}.")

            # push the batch to gpu
            batch = [r.to(self.device) for r in batch]

            sent_id, mask, token_type_ids, labels, one_hot, static_data = batch

            # clear previously calculated gradients
            self.model.zero_grad()

            # get model predictions for the current batch
            preds = self.model(sent_id, mask, token_type_ids, static_data)

            # compute loss
            loss = self.loss_fn(preds, labels)

            # add on to the total loss
            total_loss = total_loss + loss.item()
            # backward pass to calculate the gradients
            loss.backward()

            # clip the the gradients to 1.0.
            # It helps in preventing the exploding gradient problem
            clip_grad_norm_(self.model.parameters(), 1.0)

            # update parameters
            self.optimizer.step()

            # model predictions are stored on GPU. So, push it to CPU
            preds = preds.detach().cpu().numpy()
            # append the model predictions
            total_preds.append(preds)

            torch.cuda.empty_cache()

        # compute the training loss of the epoch
        avg_loss = total_loss / len(data)

        # predictions are in the form of
        # (# of batches, size of batch, # of classes).
        # reshape the predictions in form of (# of samples, # of classes)
        total_preds = np.concatenate(total_preds, axis=0)

        # returns the loss and predictions
        return avg_loss, total_preds

    def _evaluate(self, data):
        """
        Eval loop - on validation data set
        """

        print("\nEvaluating...")

        self.model.eval()  # deactivate dropout layers
        total_loss = 0

        # empty list to save the model predictions
        total_preds = []

        # iterate over batches
        for step, batch in enumerate(data):

            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:
                print(f"\tBatch {step} of {len(data)}.")

            # push the batch to gpu
            batch = [t.to(self.device) for t in batch]
            sent_id, mask, token_type_ids, labels, one_hot, static_data = batch

            # deactivate autograd
            # Dont store any previous computations, thus freeing GPU space
            with torch.no_grad():

                # model predictions
                preds = self.model(sent_id, mask, token_type_ids, static_data)
                # compute the validation loss between actual and prediction
                loss = self.loss_fn(preds, labels)

                total_loss = total_loss + loss.item()

                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)

            torch.cuda.empty_cache()

        # compute the validation loss of the epoch
        avg_loss = total_loss / len(data)
        # reshape the predictions in form of (# of samples, # of classes)
        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds

    def train(self, traindata, valdata, epochs=10, save=True):
        """
        Main training loop - trains and evals model, saving best version
        """

        for epoch in range(epochs):

            print(f"\nEpoch {epoch+1} / {epochs}")

            # train model
            train_loss, _ = self._train_bert(traindata)

            # evaluate model
            valid_loss, _ = self._evaluate(valdata)

            print(f'Evaluation done for epoch {epoch+1}')
            print(f"Losses: {train_loss}, {valid_loss}")

            # save the best model
            if save:
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    print('Saving model...')
                    # Save model weight's (you can also save it in .bin format)
                    torch.save(self.model.state_dict(), self.filename)

            # append training and validation loss
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')

    def _get_accuracy(out, actual_labels, batchSize):
        """
        Computes the accuracy of a model's predictions for a given batch.

        Parameters:
        - out (Tensor): The log probabilities or logits returned by the model.
        - actual_labels (Tensor): The actual labels for the batch.
        - batchSize (int): The size of the batch.

        Returns:
        float: The accuracy for the batch.
        """
        # Get the predicted labels from the maximum value of log probabilities
        predictions = out.max(dim=1)[1]
        # Count the number of correct predictions
        correct = (predictions == actual_labels).sum().item()
        # Compute the accuracy for the batch
        accuracy = correct / batchSize

        return accuracy

    def test(self, testdata, folds, load=True):
        """
        Test loop
        """

        print('\nTesting')
        print('Total batches:', len(testdata))

        test_preds = []

        for fold_index in range(0, folds):

            print('\nFold Model', fold_index)

            if load:
                self.model.load_state_dict(torch.load(self.filename))

            # Put the model in evaluation mode.
            self.model.eval()

            # Turn off the gradient calculations.
            torch.set_grad_enabled(False)

            for step, batch in enumerate(testdata):
                # Progress update every 50 batches.
                if step % 50 == 0 and not step == 0:
                    print(f"\tBatch {step} of {len(testdata)}.")

                # push the batch to gpu
                batch = [t.to(self.device) for t in batch]
                sent_id, mask, token_type_ids, labels, one_hot, static_data = batch

                outputs = self.model(
                    sent_id, mask, token_type_ids, static_data)

                # Get the preds
                probs = F.softmax(outputs, dim=1)

                # Move preds to the CPU
                val_probs = probs.detach().cpu().numpy()

                # Stack the predictions.
                if step == 0:  # first batch
                    stacked_val_preds = val_probs

                else:
                    stacked_val_preds = np.vstack(
                        (stacked_val_preds, val_probs))

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

        # Take the argmax.
        # This returns the column index of the max value in each row.
        predictions = np.argmax(avg_preds, axis=1)

        true_y = np.array([x[3] for x in testdata]).flatten()[:len(predictions)]

        return predictions, true_y
