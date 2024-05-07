from sklearn.cluster import KMeans  
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np

import pickle
import pandas as pd
import time

from data_preprocessing import DataPreprocessing


class KMeansTrainer():
    def __init__(self,n_clusters):
        self.model = KMeans(n_clusters=n_clusters, init="random", random_state= 42,n_init=1,max_iter=100)  

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
        print("Training Time:",end_time - start_time)


        self.centroids=self.model.cluster_centers_
        self.labels=self.model.labels_

        # evaluate the model
        # Silhouette Score: Silhouette score measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). 
        # The silhouette score ranges from -1 to 1. A score close to 1 indicates that the data point is very similar to other data points in the cluster,
        print(f"Silhouette Score:",silhouette_score(self.X_train, self.labels))

        return None

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


    def plot_T_SNE_2D(self, X, y, classes, centroids=None):
        # Initialize t-SNE object
        tsne = TSNE(n_components=2, random_state=42)

        # Fit and transform the word vectors
        x_2d = tsne.fit_transform(X)  # (100000, 2)

        # Define colors for each unique class
        num_classes = len(classes)
        color_list = plt.cm.tab10(np.linspace(0, 1, num_classes))
        colors = {i: color_list[i] for i, class_label in enumerate(classes)}

        # Create a single scatter plot
        plt.figure(figsize=(10, 8))

        # Plot each class separately
        for class_label in range(num_classes):
            # Filter data for the current class
            x_class = x_2d[y == class_label]

            # Scatter plot for the current class
            plt.scatter(x_class[:, 0], x_class[:, 1], marker='.', c=colors[class_label], label=f'Class {classes[class_label]}')

            # Plot centroids if provided
            if centroids is not None:
                centroids_transformed = tsne.transform(centroids)
                plt.scatter(centroids_transformed[class_label, 0], centroids_transformed[class_label, 1], marker='x', s=100, c='red', label='Centroid')

        # Set labels and title
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('t-SNE Visualization')

        # Show legend
        plt.legend()

        # Save and show the plot
        plt.tight_layout()
        plt.savefig('t-SNE_plot.png')
        plt.show()
    # def plot_T_SNE_2D(self, X, y, classes, centroids=None):
    #     # Initialize t-SNE object
    #     tsne = TSNE(n_components=2, random_state=42)

    #     # Fit and transform the word vectors
    #     x_2d = tsne.fit_transform(X)  # (100000, 2)

    #     # Define colors for each unique class
    #     num_classes = len(classes)
    #     color_list = plt.cm.tab10(np.linspace(0, 1, num_classes))
    #     colors = {i: color_list[i] for i, class_label in enumerate(classes)}


    #     # Initialize subplots
    #     fig, axes = plt.subplots(1, len(classes), figsize=(15, 5))

    #     # Iterate over unique classes and plot them separately
    #     for class_label, ax in zip(range(len(classes)), axes):
    #         # Filter data for the current class
    #         x_class = x_2d[y == class_label]
            
    #         # Scatter plot for the current class
    #         ax.scatter(x_class[:, 0], x_class[:, 1], marker='.', c=colors[class_label], label=f'Class {classes[class_label]}')

    #         # Plot centroids if provided
    #         if centroids is not None:
    #             centroids_transformed = tsne.transform(centroids)
    #             ax.scatter(centroids_transformed[class_label, 0], centroids_transformed[class_label, 1], marker='x', s=100, c='red', label='Centroid')

    #         ax.set_xlabel('t-SNE Dimension 1')
    #         ax.set_ylabel('t-SNE Dimension 2')
    #         ax.set_title(f't-SNE Visualization for Class {classes[class_label]}')
    #         ax.legend()

    #     plt.tight_layout()
    #     plt.savefig('t-SNE_subplots.png')
    #     plt.show()

    def plot_T_SNE_3D(self, X, y, classes, centroids=None):
        # Initialize t-SNE object
        tsne = TSNE(n_components=3, random_state=42)

        # Fit and transform the word vectors
        x_3d = tsne.fit_transform(X)  # (100000, 3)

        # Define colors for each unique class
        num_classes = len(classes)
        color_list = plt.cm.tab10(np.linspace(0, 1, num_classes))
        colors = {i: color_list[i] for i, class_label in enumerate(classes)}

        # Initialize a 3D subplot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each class separately
        for class_label in range(num_classes):
            # Filter data for the current class
            x_class = x_3d[y == class_label]

            # Scatter plot for the current class
            ax.scatter(x_class[:, 0], x_class[:, 1], x_class[:, 2], marker='.', c=colors[class_label], label=f'Class {classes[class_label]}')

            # Plot centroids if provided
            if centroids is not None:
                centroids_transformed = tsne.transform(centroids)
                ax.scatter(centroids_transformed[class_label, 0], centroids_transformed[class_label, 1], centroids_transformed[class_label, 2], marker='x', s=100, c='red', label='Centroid')

        # Set labels and title
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')
        ax.set_title('t-SNE 3D Visualization')
        ax.legend()


        plt.tight_layout()
        # plt.savefig('t-SNE_subplots.png')

        # Enable interactive rotation
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

    # Drop Category Column
    categorical_cols = ['Occupation','Credit_Mix','Payment_of_Min_Amount', 'Payment_Behaviour']
    X = data_preprocessing.drop_columns(categorical_cols)
    
    # Convert Categorical Columns to Numerical & Scale continuous columns
    X = data_preprocessing.convert_categories_to_one_hot_normalize_numerical()

    # Trainer
    K_means_trainer = KMeansTrainer(n_clusters=5)

    # Set Data
    K_means_trainer.set_data(X,columns=data_preprocessing.get_columns())


    # Start Training
    print("Training Kmeans Model")
    K_means_trainer.train()

    # Save the model
    K_means_trainer.save_model('./models/kmeans_sklearn.pkl')

# Path: python src/knn.py

# Training Kmeans Model
# Training Time: 0.6300568580627441
# Silhouette Score: 0.09255414134312365