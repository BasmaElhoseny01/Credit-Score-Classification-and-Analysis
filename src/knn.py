from sklearn.neighbors import KNeighborsClassifier 

from trainer import Trainer
import pickle
import pandas as pd

from data_preprocessing import DataPreprocessing


class KNNTrainer(Trainer):
    def __init__(self,n_neighbors):
        model=KNeighborsClassifier(n_neighbors=n_neighbors)
        super().__init__(model)

    def evaluate(self):
        super().evaluate()

    def train(self):
        super().train()

    def save_model(self,path):
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
    
if __name__ == "__main__":
    # Load the data
    data_preprocessing = DataPreprocessing('./dataset/train_preprocessed.csv')
    data_preprocessing.load_data()

    # Drop Target Column
    X = data_preprocessing.drop_columns(['Credit_Score'])

    # Convert Categorical Columns to Numerical
    X = data_preprocessing.convert_catgories_to_numerical()

    # Trainer
    print("Training KNN Model")
    Knn_trainer = KNNTrainer(n_neighbors=5)

    # Start Training
    Knn_trainer.train()

    # Evaluate the model
    Knn_trainer.evaluate()

    # Save the model
    Knn_trainer.save_model('./models/knn_model.pkl')

# Path: python src/knn.py