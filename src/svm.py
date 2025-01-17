# implement svm model using sklearn library
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from trainer import Trainer
import pandas as pd
import numpy as np
import pickle
from data_preprocessing import DataPreprocessing

class SVMTrainer(Trainer):
    def __init__(self, C=4.0, kernel='rbf', degree=4, gamma='scale', max_iter=1000):
        # initialize the model
        model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, max_iter=max_iter)
        super().__init__(model)

    def evaluate(self):
        # evaluate the model
        super().evaluate()

    def train(self):
        # train the model
        super().train()

    def save_model(self, path):
        # save the model
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        # load the model
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, X, preprocessor=None):
        if preprocessor:
            # preprocess the data
            X = preprocessor.transform(X)
        # make prediction
        return self.model.predict(X)

    def tune_hyperparameters(self, hyperparameters, X, y):
        # tune hyperparameters using grid search
        grid_search = GridSearchCV(self.model, hyperparameters, cv=5)
        grid_search.fit(X, y)
        return grid_search.best_params_
    def grid_search(self, X, y):
        # grid search for hyperparameters
        hyperparameters = {
            'C': [0.1, 1, 10, 100],
            'kernel': [ 'rbf', 'poly'],
            'degree': [3, 4, 5],
            'gamma': ['scale', 'auto']
        }
        best_params = self.tune_hyperparameters(hyperparameters, X, y)

        self.model = SVC(C=best_params['C'], kernel=best_params['kernel'], degree=best_params['degree'], gamma=best_params['gamma'])
        self.train()
        self.evaluate(X,y)
        self.save_model('models/svm_model.pkl')
        




# Path: src/main.py

if __name__ == "__main__":
    # load the data
    data_preprocessing = DataPreprocessing('../dataset/train_preprocessed.csv')
    data_preprocessing.load_data()
    y = data_preprocessing.convert_Y_to_numerical()
    data_preprocessing.drop_columns(['Credit_Score'])
    X = data_preprocessing.convert_catgories_to_numerical()

    svm_trainer = SVMTrainer()
    svm_trainer.split_data(X, y)
    svm_trainer.train()
    svm_trainer.evaluate()
