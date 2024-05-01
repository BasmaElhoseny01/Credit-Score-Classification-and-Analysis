# import train test split from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


class Trainer:
    def __init__(self, model):
        self.model = model
    def split_data(self, X, y, test_size=0.2, random_state=42):
        # split data into train and test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    def train(self):
        # train the model
        self.model.fit(self.X_train, self.y_train)
    def evaluate(self):
        # evaluate the model
        y_pred = self.model.predict(self.X_test)  

        # calculate accuracy
        accuracy = np.mean(y_pred == self.y_test)

        # calculate confusion matrix
        confusion_mat = confusion_matrix(self.y_test, y_pred)

        # calculate precision, recall, f1-score from confusion matrix
        precision = np.diag(confusion_mat) / np.sum(confusion_mat, axis=0)
        recall = np.diag(confusion_mat) / np.sum(confusion_mat, axis=1)
        f1_score = 2 * precision * recall / (precision + recall)

        print(f"Accuracy: {accuracy}")
        print(f"Confusion matrix: {confusion_mat}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1_score}")

    def save_model(self, path):
        # save the model
        self.model.save(path)

    def load_model(self, path):
        # load the model
        self.model.load(path)

    def predict(self, X, preprocessor=None):
        if preprocessor:
            # preprocess the data
            X = preprocessor.transform(X)
        # make prediction
        return self.model.predict(X)