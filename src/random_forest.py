from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from trainer import Trainer
import pandas as pd
import numpy as np
import pickle
from data_preprocessing import DataPreprocessing

class RandomForestTrainer(Trainer):
    def __init__(self, n_estimators=1000, criterion='gini', max_depth=None, max_features='auto'):
        # initialize the model
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, max_features=max_features)
        super().__init__(model)

    def evaluate(self):
        # evaluate the model
        return super().evaluate()

    def train(self):
        # train the model
        return super().train()

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
            'n_estimators': [50, 100, 150],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'max_features': ['auto', 'sqrt']
        }
        best_params = self.tune_hyperparameters(hyperparameters, X, y)

        self.model = RandomForestClassifier(n_estimators=best_params['n_estimators'], criterion=best_params['criterion'], max_depth=best_params['max_depth'], max_features=best_params['max_features'])
        self.train()
        self.evaluate()


if __name__ == "__main__":
    data_preprocessing = DataPreprocessing('dataset/train_preprocessed.csv')
    data_preprocessing.load_data()
    y = data_preprocessing.convert_Y_to_numerical()
    data_preprocessing.drop_columns(['Credit_Score'])
    X = data_preprocessing.convert_catgories_to_numerical2()
    random_forest_trainer = RandomForestTrainer()
    random_forest_trainer.split_data(X, y)
    random_forest_trainer.train()
    random_forest_trainer.evaluate()