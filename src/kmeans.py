from sklearn.cluster import KMeans  
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt


import pickle
import pandas as pd
import time

from data_preprocessing import DataPreprocessing


class KMeansTrainer():
    def __init__(self,n_clusters):
        self.model = KMeans(n_clusters=n_clusters, init="random", random_state= 42)  

    def set_data(self, X,columns):
        self.X_train = X
        self.columns = columns

    def train(self):
        # Compute Train time
          # Start time
        start_time = time.time()
        
        self.model.fit(self.X_train)

        # End time
        end_time = time.time()
        
        # Compute time taken
        time_taken = end_time - start_time

        self.centroids=self.model.cluster_centers_
        self.labels=self.model.labels_

        # evaluate the model
        # Silhouette Score: Silhouette score measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). 
        # The silhouette score ranges from -1 to 1. A score close to 1 indicates that the data point is very similar to other data points in the cluster,
        print(f"Silhouette Score:",silhouette_score(self.X_train, self.labels))

        return time_taken

    def evaluate(self):
        super().evaluate()


    def plot_clusters(self,feature1,feature2):
        # Get the indices of the features
        f1_index = self.columns.get_loc(feature1)
        f2_index = self.columns.get_loc(feature2)
        
        # Plot the clusters
        plt.scatter(self.X_train[:, f1_index], self.X_train[:, f2_index], c=self.labels, cmap='viridis')
        plt.scatter(self.centroids[:, f1_index], self.centroids[:, f2_index], marker='x', s=100, c='red', label='Centroids')
        plt.title('Cluster Plot')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.legend()
        plt.show()
        

    def save_model(self,path):
        # save the model
        with open(path, 'wb') as f:
            pickle.dump(self.centroids, f)

    # def load_model(self, path):
    #     # load the model
    #     with open(path, 'rb') as f:
    #         self.model = pickle.load(f)
    #         self.centroids
    
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
    K_means_trainer.set_data(X,columns=data_preprocessing.get_columns())


    # Start Training
    print("Training Kmeans Model")
    train_time=K_means_trainer.train()
    print("Training Time:",train_time)

    # Save the model
    K_means_trainer.save_model('./models/knn_sklearn.pkl')

    K_means_trainer.plot_clusters('Age','Annual_Income')

# Path: python src/knn.py