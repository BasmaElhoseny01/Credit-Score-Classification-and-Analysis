from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from trainer import Trainer
import pandas as pd
import numpy as np
import pickle
from data_preprocessing import DataPreprocessing

class GradientBoostingTrainer(Trainer):
    def __init__(self,n_estimators=120, learning_rate=1,max_depth=20, random_state=0 ):
        # initialize the model
        print ("GradientBoostingTrainer")
        print("Train XGBoost with n_estimators: ", n_estimators, " learning_rate: ", learning_rate, " max_depth: ", max_depth, " random_state: ", random_state)
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
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
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 7 , 8],
            'learning_rate': [0.1, 0.01, 1]
        }
        best_params = self.tune_hyperparameters(hyperparameters, X, y)

        self.model = GradientBoostingClassifier (n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
        self.train()
        self.evaluate()


if __name__ == "__main__":
    data_preprocessing = DataPreprocessing('dataset/train_preprocessed.csv')
    data_preprocessing.load_data()
    y = data_preprocessing.convert_Y_to_numerical()
    data_preprocessing.drop_columns(['Credit_Score'])
    X = data_preprocessing.convert_catgories_to_numerical2()
    xgboost_trainer = GradientBoostingTrainer()
    xgboost_trainer.split_data(X, y)
    xgboost_trainer.train()
    xgboost_trainer.evaluate()
    xgboost_trainer.save_model('models/xgboost_model.pkl')


# python -m src.xgboost