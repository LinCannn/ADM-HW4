from pyspark import SparkContext
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd

class MyKMeans:
    def __init__(self, k, num_iterations=10, tolerance=1e-4):
        """
        Initialize the KMeans class.
        
        Args:
            k (int): Number of clusters.
            num_iterations (int): Maximum number of iterations.
            tolerance (float): Convergence threshold for centroid shifts.
        """
        self.k = k # Number of clusters
        self.num_iterations = num_iterations # Maximum number of iterations
        self.tolerance = tolerance # Convergence threshold for centroid shifts. determines when to stop iterating
        self.centroids = None # Centroids of the clusters

    def initialize_centroids(self, data_rdd, seed=210):
        """
        Randomly initialize centroids from the dataset.
        
        Args:
            data (pd.DataFrame): Dataset in Pandas DataFrame format.
            seed (int): Random seed for reproducibility.
        """
        np.random.seed(seed) # Set random seed for reproducibility
        # Convert the RDD to a numpy array
        points = np.array(data_rdd.map(lambda row: row[1:]).collect())
        # Randomly select k indices
        idxs = np.random.choice(len(points), size=self.k, replace=False)
        # set those k indices as the initial centroids
        self.centroids = points[idxs]

    @staticmethod
    def euclidean_distance(point, centroid):
        """
        Compute the Euclidean distance between a point and a centroid.
        """
        return np.sqrt(np.sum((np.array(point) - np.array(centroid)) ** 2))

    def assign_to_centroid(self, row):
        """
        Assign a data point to the nearest centroid.
        
        Args:
        row (list): Row of data with [PointID, Feature1, Feature2, ...]. It will be a spark rdd row
            
        Returns:
            Tuple[int, Tuple[int, list]]: (ClusterID, (PointID, Features))
        """
        point_id = row[0] # PointID
        features = row[1:] # Features
        # Compute the distance between the point and each centroid
        distances = [self.euclidean_distance(features, centroid) for centroid in self.centroids]
        # Assign the point to the nearest centroid
        cluster_id = np.argmin(distances)
        # return the cluster id and the point id and features
        return cluster_id, (point_id, features)

    def compute_inertia(self, points_rdd):
        """
        Compute the inertia (sum of squared distances to centroids).
        """
        # Compute the squared distance between each point and each centroid
        inertia = points_rdd.map(
            lambda row: min([np.linalg.norm(row[1:] - centroid) ** 2 for centroid in self.centroids])
        ).sum()
        return inertia / points_rdd.count()

    def fit(self, data_rdd):
        """
        Perform the KMeans clustering.
        
        Args:
            data (pd.DataFrame): Dataset in Pandas DataFrame format.
        
        Returns:
            pd.DataFrame: DataFrame containing cluster assignments and features.
        """
        # Convert dataset to PySpark RDD
        self.initialize_centroids(data_rdd)
        
        for iteration in range(self.num_iterations):
            # Assign each point to the nearest centroid
            mapped = data_rdd.map(lambda row: self.assign_to_centroid(row))

            # Group points by cluster ID and compute new centroids
            clustered_points = mapped.groupByKey().mapValues(list)
            new_centroids = (
                clustered_points.mapValues(lambda points: np.mean([point[1] for point in points], axis=0))
                .collectAsMap() # collectAsMap() returns a dictionary of the form {ClusterID: NewCentroid}
            )
            updated_centroids = np.array([new_centroids[i] for i in range(self.k)]) # convert the dictionary to an array of centroids

            # Check for convergence
            centroid_shift = np.linalg.norm(updated_centroids - self.centroids)
            print(f"Iteration {iteration + 1}: Centroid shift = {centroid_shift:.4f}")
            inertia = self.compute_inertia(data_rdd)
            print(f"Inertia: {inertia}")
            
            if centroid_shift < self.tolerance:
                print(f"Converged after {iteration + 1} iterations.")
                break

            self.centroids = updated_centroids

        # Assign final cluster labels
        final_clusters = data_rdd.map(lambda row: self.assign_to_centroid(row)).collect()
        return self.create_cluster_dataframe(final_clusters)

    @staticmethod
    def create_cluster_dataframe(final_clusters):
        """
        Convert the cluster assignments to a Pandas DataFrame.
        """
        # Initialize a DataFrame with data of final_clusters columns "ClusterID" and "Point" (containing PointID and Features)
        cluster_assignments = pd.DataFrame(final_clusters, columns=["ClusterID", "Point"])
        # Split the "Point" column into "PointID" and "Features" columns
        cluster_assignments["PointID"] = cluster_assignments["Point"].apply(lambda x: x[0])
        cluster_assignments["Features"] = cluster_assignments["Point"].apply(lambda x: x[1])
        cluster_assignments = cluster_assignments.drop("Point", axis=1) # Drop the "Point" column
        return cluster_assignments

class KMeansPlusPlus(MyKMeans):
    # Inherits from MyKMeans
    def __init__(self, k, num_iterations=10, tolerance=1e-4):
        """
        Initializing the KMeansPlusPlus class.
        
        Args:
            k (int): Number of clusters.
            num_iterations (int): Maximum number of iterations.
            tolerance (float): Convergence threshold for centroid shifts.
        """
        # Call the parent class's __init__ method to initialize the parameters and inherit the methods
        super().__init__(k, num_iterations, tolerance) 

    def initialize_centroids(self, data_rdd, seed=210):
        """
        initialization of the centroids using the KMeans++ logic.
        
        Args:
            data_rdd (RDD): PySpark RDD containing the dataset.
            seed (int): Random seed for reproducibility.
        """
        np.random.seed(seed)
        # Convert the RDD to a numpy array
        points = np.array(data_rdd.map(lambda row: row[1:]).collect())

        # Step 1: Randomly pick the first centroid
        centroids = [points[np.random.choice(len(points))]]

        # Step 2: Choose remaining centroids using KMeans++ logic
        for _ in range(1, self.k):
            # Compute distances from points to the nearest centroid
            distances = np.min([np.linalg.norm(points - c, axis=1) for c in centroids], axis=0)

            # Select the next centroid with probability proportional to distance^2
            probabilities = distances**2 / np.sum(distances**2)
            # Choose the next centroid randomly based on the probabilities
            next_centroid = points[np.random.choice(len(points), p=probabilities)]
            # Append the next centroid to the list of centroids
            centroids.append(next_centroid)
        # Set the centroids attribute to centroids (but as an array)
        self.centroids = np.array(centroids)

    def fit(self, data_rdd):
        """
        Fit the KMeans++ algorithm using the inherited methods from MyKMeans.
        
        Args:
            data (pd.DataFrame): Dataset in Pandas DataFrame format.
        
        Returns:
            pd.DataFrame: DataFrame containing cluster assignments and features.
        """
        # Use the new KMeans++ initialization method
        self.initialize_centroids(data_rdd)

        # Call the parent class's fit method for the rest of the algorithm
        return super().fit(data_rdd)
    
