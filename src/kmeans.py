from sklearn.cluster import KMeans  
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score


import pickle
import pandas as pd

from data_preprocessing import DataPreprocessing


class KMeansTrainer():
    def __init__(self,n_clusters):
        self.model = KMeans(n_clusters=n_clusters, init='k-means++', random_state= 42)  

    def set_data(self, X):
        self.X_train = X

    # def evaluate(self):
        # super().evaluate()

    def train(self):
        self.model.fit(self.X_train)


        self.centroids=self.model.cluster_centers_
        self.labels=self.model.labels_

        # evaluate the model
        # 1. Inertia: Inertia measures how tightly clustered the points are in each cluster. It's the sum of squared distances of samples to their closest cluster center. Lower inertia indicates better clustering.
        print(f"Inertia:",self.model.inertia_)

        # 2. Silhouette Score: Silhouette score measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). 
        # The silhouette score ranges from -1 to 1. A score close to 1 indicates that the data point is very similar to other data points in the cluster,
        print(f"Silhouette Score:",silhouette_score(self.X_train, self.labels))

        # 3. Calinski-Harabasz Index: is the ratio of the between-cluster dispersion mean to the within-cluster dispersion. Higher values indicate better clustering.
        print(f"Calinski-Harabasz Index:",calinski_harabasz_score(self.X_train, self.labels))
    

    def save_model(self,path):
        # save the model
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        # load the model
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    # def predict(self, X, preprocessor=None):
    #     if preprocessor:
    #         # preprocess the data
    #         X = preprocessor.transform(X)
    #     # make prediction
    #     return self.model.predict(X)
    
if __name__ == "__main__":
    # Load the data
    data_preprocessing = DataPreprocessing('./dataset/train_preprocessed.csv')
    data_preprocessing.load_data()


    # Drop Target Column
    X = data_preprocessing.drop_columns(['Credit_Score'])

    # Convert Categorical Columns to Numerical & Scale continuous columns
    X = data_preprocessing.convert_catgories_to_numerical()

    # Trainer
    K_means_trainer = KMeansTrainer(n_clusters=5)

    # Set Data
    K_means_trainer.set_data(X)

    # Start Training
    print("Training Kmeans Model")
    K_means_trainer.train()

    # Evaluate the model
    # Knn_trainer.evaluate()

    # Save the model
    # Knn_trainer.save_model('./models/knn_model.pkl')

# Path: python src/knn.py