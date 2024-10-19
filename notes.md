Cluster Analysis and Clustering
# Cluster Analysis and Clustering

## Table of Contents
- [What is Clustering?](#what-is-clustering)
- [Cluster Analysis](#cluster-analysis)
- [Applications of Cluster Analysis](#applications-of-cluster-analysis)
- [Example of Clustering](#example-of-clustering)

---

## What is Clustering? 
**Definition:** A cluster is a group of data points in a dataset that are more similar to each other than to those in other groups.
- Clusters can be formed based on various features or characteristics inherent in the data.
  - **Example:** In a dataset of animals, one cluster might consist of mammals, another of birds, and another of reptiles, based on characteristics such as body temperature regulation, presence of feathers, etc.
- Similarity can be measured using various distance metrics like Euclidean distance, Manhattan distance, etc.

---

## Cluster Analysis 
**Definition:** Cluster analysis is a technique used in unsupervised learning to group a set of objects in such a way that objects in the same group (or cluster) are more similar to each other than to those in other groups (clusters).
- It involves various methods and algorithms to identify and assign data points to clusters based on their similarities.

---

## Applications of Cluster Analysis 
- **Exploratory Data Analysis:** Understanding the natural grouping or structure in the data.
- **Data Reduction:** Reducing the number of data points by grouping them into clusters, simplifying the dataset.
- **Pattern Recognition:** Identifying patterns or trends within the data.

---

## Example of Clustering 
**Example:** Document Clustering. The collection of news articles can be grouped based on their respective topics.
- Each article is represented as a set of word-frequency pairs (w, c), where w is a word and c is the number of times the word appears in the article.
- There are two natural clusters in the dataset. The first cluster consists of the first four articles, which correspond to news about the economy, while the second cluster contains the last four articles, which correspond to news about healthcare.
- A good clustering algorithm should be able to identify these clusters accurately.
Unsupervised Learning Applications
# Unsupervised Learning Applications 

## Table of Contents 

* [Clustering](#Clustering) 
* [Dimensionality Reduction](#Dimensionality-Reduction) 
* [Anomaly Detection](#Anomaly-Detection) 
* [Association Mining](#Association-Mining) 
* [Data Compression](#Data-Compression) 
* [Generative Modeling](#Generative-Modeling) 

## Clustering 

- **Customer Segmentation** 
  - Grouping customers based on purchasing behavior, demographics, and preferences to tailor marketing strategies 
  - *Example:* Analyzing customer data to identify distinct groups with similar buying habits 
  - *Key points:* 
    - **Demographic-based segmentation**: Grouping customers by age, location, or income 
    - **Behavioral-based segmentation**: Grouping customers by purchasing patterns or interaction history 

- **Document Clustering** 
  - Organizing a large collection of documents into thematic clusters for improved information retrieval and topic modeling 
  - *Example:* Clustering news articles by topic for easier browsing 
  - *Key points:* 
    - **Text preprocessing**: Cleaning and normalizing text data for clustering 
    - **Clustering algorithms**: Using techniques like k-means or hierarchical clustering for document grouping 

- **Image Segmentation** 
  - Partitioning an image into segments for object detection and recognition tasks in computer vision 
  - *Example:* Segmenting an image of a scene into objects like cars, trees, and buildings 
  - *Key points:* 
    - **Image preprocessing**: Preparing images for segmentation by converting to grayscale or normalizing pixel values 
    - **Segmentation algorithms**: Using techniques like thresholding or edge detection for image segmentation 

--- 

## Dimensionality Reduction 

- **Data Visualization** 
  - Reducing high-dimensional data to 2D or 3D for visualization, making it easier to identify patterns and relationships 
  - *Example:* Using PCA to reduce dimensionality of a dataset for visualization 
  - *Key points:* 
    - **Dimensionality reduction techniques**: Using methods like PCA, t-SNE, or Autoencoders for dimensionality reduction 
    - **Visualization tools**: Using libraries like Matplotlib or Seaborn for visualizing reduced data 

- **Feature Selection and Extraction** 
  - Reducing the number of features while retaining the most important information to improve model performance and reduce computational cost 
  - *Example:* Selecting the most relevant features for a machine learning model using mutual information 
  - *Key points:* 
    - **Feature selection methods**: Using techniques like correlation analysis or recursive feature elimination 
    - **Feature extraction methods**: Using techniques like PCA or Autoencoders for feature extraction 

--- 

## Anomaly Detection 

- **Fraud Detection** 
  - Identifying unusual transactions or activities in financial data that may indicate fraudulent behavior 
  - *Example:* Detecting unusual patterns in credit card transactions 
  - *Key points:* 
    - **Anomaly detection algorithms**: Using techniques like One-class SVM or Local Outlier Factor (LOF) 
    - **Data preprocessing**: Preparing data for anomaly detection by handling missing values or outliers 

- **Network Security** 
  - Detecting unusual patterns of network traffic that may indicate cyber-attacks or intrusions 
  - *Example:* Identifying unusual patterns in network traffic logs 
  - *Key points:* 
    - **Network traffic analysis**: Analyzing network traffic data to identify patterns and anomalies 
    - **Intrusion detection systems**: Using systems like IDS or IPS for detecting and preventing intrusions 

- **Industrial Equipment Monitoring** 
  - Identifying deviations in sensor data from manufacturing equipment that may indicate malfunctions or the need for maintenance 
  - *Example:* Monitoring sensor data from manufacturing equipment to detect anomalies 
  - *Key points:* 
    - **Sensor data analysis**: Analyzing sensor data to identify patterns and anomalies 
    - **Predictive maintenance**: Using anomaly detection for predictive maintenance and reducing downtime 

--- 

## Association Mining 

- **Market Basket Analysis** 
  - Discovering associations between products in transaction data to understand purchasing patterns and optimize product placement and cross-selling strategies 
  - *Example:* Analyzing transaction data to identify frequent itemsets 
  - *Key points:* 
    - **Association rule mining**: Using techniques like Apriori or FP-growth for association rule mining 
    - **Market basket analysis**: Analyzing transaction data to identify patterns and associations between products 

- **Recommender Systems** 
  - Finding patterns in user behavior to recommend products, movies, or other items based on previous interactions 
  - *Example:* Building a recommender system for a movie streaming service 
  - *Key points:* 
    - **Collaborative filtering**: Using user behavior and ratings to recommend items 
    - **Content-based filtering**: Using item attributes and features to recommend items 

--- 

## Data Compression 

- **Image and Video Compression** 
  - Reducing the amount of data required to represent images and videos while maintaining quality, using techniques such as clustering similar pixel values 
  - *Example:* Compressing images using clustering-based techniques 
  - *Key points:* 
    - **Image compression algorithms**: Using techniques like JPEG or WebP for image compression 
    - **Video compression algorithms**: Using techniques like H.264 or H.265 for video compression 

--- 

## Generative Modeling 

- **Image Generation** 
  - Creating realistic images from random noise using models like Generative Adversarial Networks (GANs) 
  - *Example:* Generating realistic images of faces using GANs 
  - *Key points:* 
    - **Generative models**: Using models like GANs or Variational Autoencoders (VAEs) for image generation 
    - **Image synthesis**: Generating realistic images from random noise or existing images 

- **Text Generation** 
  - Producing coherent and contextually relevant text based on learned patterns from large corpora 
  - *Example:* Generating text summaries using sequence-to-sequence models 
  - *Key points:* 
    - **Language models**: Using models like sequence-to-sequence or transformer-based models for text generation 
    - **Text synthesis**: Generating coherent and contextually relevant text based on learned patterns 

--- 
Metrics for Clustering
# Metrics for Clustering

## Evaluating Clustering Results

Evaluating the quality of clustering results can be challenging because there is no ground truth in unsupervised learning.

## Types of Evaluation Metrics

- **Internal Evaluation Metrics**
  - Evaluate the clustering based on the inherent properties of the data without any external reference.
- **External Evaluation Metrics**
  - Not discussed in the provided content
- **Relative Evaluation Metrics**
  - Compare different clustering solutions or algorithms on the same dataset to determine the best one.

---

## Internal Evaluation Metrics

### Silhouette Score

- **Measures**: how similar an object is to its own cluster compared to other clusters.
- **Values**: range from -1 to 1. Higher values indicate better-defined clusters.
- **Formula**: Silhouette = b - a / max(a, b)
- **Variables**:
  - a: Average distance between a sample and all other points in the same cluster.
  - b: Average distance between a sample and all points in the nearest cluster.

---

## Relative Evaluation Metrics

### Elbow Method

- **Method**: Plots the explained variance as a function of the number of clusters.
- **Optimal Number of Clusters**: The point where the explained variance starts to diminish.

---

## Table of Contents

- [Evaluating Clustering Results](#evaluating-clustering-results)
- [Types of Evaluation Metrics](#types-of-evaluation-metrics)
- [Internal Evaluation Metrics](#internal-evaluation-metrics)
- [Relative Evaluation Metrics](#relative-evaluation-metrics)

---

## Important Notes

* Clustering evaluation metrics can be broadly categorized into internal, external, and relative evaluation measures.
* The choice of evaluation metric depends on the specific problem and dataset.

---


K-Means Clustering Geometric Intuition
# K-Means Clustering Geometric Intuition

## Table of Contents
- [K-Means Clustering Overview](#k-means-clustering-overview)
- [Initialization](#initialization)
- [Assignment Step](#assignment-step)
- [Update Step](#update-step)
- [Iteration](#iteration)

---

## K-Means Clustering Overview
- **Definition**: K-means clustering is a popular and simple method for partitioning a dataset into K distinct, non-overlapping subsets (clusters).
- *Key Characteristics*: 
  - Non-overlapping clusters
  - Each data point belongs to only one cluster

---

## Initialization
- **Randomly Select K Centroids**: The algorithm begins by randomly selecting K points from the dataset as the initial centroids (cluster centers).
- **Centroid Definition**: The centroid is the mean point (central point) of a cluster.
- **Intersection of Clusters**: The intersection of two different clusters is always a null set.

---

## Assignment Step
- **Assign Points to Nearest Centroid**: Each data point in the dataset is assigned to the nearest centroid.
- **Geometric Intuition**: Imagine each centroid having a region of influence. Each data point belongs to the centroid whose region it lies in. This is akin to partitioning the space into Voronoi cells around each centroid.

---

## Update Step
- **Recompute Centroids**: Once all points are assigned to clusters, the centroids are recalculated as the mean of all points in each cluster.
- **Geometric Intuition**: The new centroid is the geometric center (mean) of all the points in the cluster, pulling it towards the middle of the cluster.

---

## Iteration
- **Repeat Assignment and Update Steps**: Steps 2 and 3 are repeated iteratively. In each iteration, points are re-assigned to the nearest centroid, and centroids are recalculated.
- **Convergence**: This process continues until the centroids no longer move significantly, indicating convergence.

---
K-means Clustering
# K-means Clustering

## Table of Contents
- [Initialization](#initialization)
- [Assignment Step](#assignment-step)
- [Update Step](#update-step)
- [Centroids and Objective Functions](#centroids-and-objective-functions)

## Initialization
- **Randomly Select K Centroids**: The algorithm begins by randomly selecting K points from the dataset as the initial centroids (cluster centers).
  - **Centroid Definition**: The centroid is the mean point (central point) of each cluster. For each cluster, we get one centroid, and the intersection of two different clusters is always a null set.

---

## Assignment Step
- **Assign Points to Nearest Centroid**: Each data point in the dataset is assigned to the nearest centroid. This forms K clusters based on the current positions of the centroids.
  - **Geometric Intuition**: Imagine each centroid having a region of influence. Each data point belongs to the centroid whose region it lies in. This is akin to partitioning the space into Voronoi cells around each centroid.

---

## Update Step
- **Re-compute Centroids**: Once all points are assigned to clusters, the centroids are recalculated as the mean of all points in each cluster.
  - **Geometric Intuition**: The new centroid is the geometric mean of all points in the cluster.

---

## Centroids and Objective Functions
- **Centroids**: The centroid can vary depending on the proximity measure for the data and the goal of the clustering.
  - **Objective Function**: The goal of the clustering is typically expressed by an objective function that depends on the proximities of the points to one another or to the cluster centroids. For example, minimize the squared distance of each point to its closest centroid.
  - **Mathematical Determination**: Once we have specified a proximity measure and an objective function, the centroid that we should choose can often be determined mathematically.

---

K-Means Clustering Algorithm
# K-Means Clustering Algorithm

## Table of Contents
- [K-Means Mathematical Formulation](#k-means-mathematical-formulation)
- [Geometric Intuition](#geometric-intuition)
- [Iteration](#iteration)
- [Key Concepts](#key-concepts)
- [Objective Function](#objective-function)

## K-Means Mathematical Formulation
- **Overview**
  - The k-means clustering algorithm is an iterative process of moving the centers of clusters or centroids to the mean position of their constituent points, and reassigning instances to their closest clusters iteratively until there is no significant change in the number of cluster centers possible or number of iterations reached.
  - The cost function of k-means is determined by the Euclidean distance (square-norm) between the observations belonging to that cluster with its respective centroid value.
- **Intuitive Explanation**
  - If there is only one cluster (k=1), then the distances between all the observations are compared with its single mean.
  - If the number of clusters increases to 2 (k=2), then two-means are calculated and a few of the observations are assigned to cluster 1 and other observations are assigned to cluster two based on proximity.

---

## Geometric Intuition
- **Mean of All Points in Each Cluster**
  - The new centroid is the geometric center (mean) of all the points in the cluster, pulling it towards the middle of the cluster.

---

## Iteration
- **Repeat Assignment and Update Steps**
  - Steps 2 and 3 are repeated iteratively.
  - In each iteration, points are reassigned to the nearest centroid, and centroids are recalculated.
  - This process continues until the centroids no longer move significantly, indicating convergence.

---

## Key Concepts
- **Distance Metric**
  - The algorithm commonly uses Euclidean distance to measure the distance between points and centroids.
  - The Euclidean distance between two points (x1, y1) and (x2, y2) is given by: D=(𝑥 2−𝑥 1)2 +(𝑦 2−𝑦 1)2
- **Cluster Shape**
  - K-means tends to form spherical clusters because the assignment of points is based on the nearest centroid using Euclidean distance.

---

## Objective Function
- **Goal of K-Means**
  - The goal of K-means is to minimize the within-cluster sum of squares (WCSS), also known as inertia.
  - This is the sum of the squared distances of each point to its closest centroid.
- **Centroids and Objective Functions**
  - The goal of the clustering is typically expressed by an objective function that depends on the proximities of the points to one another or to the cluster centroids.
  - The centroid that we should choose can often be determined mathematically once we have specified a proximity measure and an objective function.
K-Means Clustering
# K-Means Clustering

## Table of Contents
- [K-Means Mathematical Formulation](#k-means-mathematical-formulation)
- [K-Means Geometric Intuition](#k-means-geometric-intuition)
- [K-Means Algorithm Steps](#k-means-algorithm-steps)

## K-Means Mathematical Formulation
- **Cost Function**: The cost function of k-means is determined by the Euclidean distance (square-norm) between the observations belonging to that cluster with its respective centroid value.
  - *Intuitive Understanding*: If there is only one cluster (k=1), then the distances between all the observations are compared with its single mean. If the number of clusters increases to 2 (k=2), then two-means are calculated and a few of the observations are assigned to cluster 1 and other observations are assigned to cluster 2 based on proximity.
  - *Key Point*: The cost function is used to evaluate the quality of the clustering.

---

## K-Means Geometric Intuition
- **Partitioning the Space**: K-means clustering is a popular and simple method for partitioning a dataset into K distinct, non-overlapping subsets (clusters).
  - *Geometric Intuition*: Imagine each centroid having a region of influence. Each data point belongs to the centroid whose region it lies in. This is akin to partitioning the space into Voronoi cells around each centroid.
  - **Centroid**: The centroid is the mean point (central point) of each cluster.

---

## K-Means Algorithm Steps
- **Initialization**
  - **Randomly Select K Centroids**: The algorithm begins by randomly selecting K points from the dataset as the initial centroids (cluster centers).
- **Assignment Step**
  - **Assign Points to Nearest Centroid**: Each data point in the dataset is assigned to the nearest centroid. This forms K clusters based on the current positions of the centroids.
  - *Geometric Intuition*: Each centroid has a region of influence. Each data point belongs to the centroid whose region it lies in.
- **Update Step**
  - **Re-compute Centroids**: Once all points are assigned to clusters, the centroids are recalculated as the mean of all points in each cluster.
  - *Geometric Intuition*: The new centroid is the geometric center of the cluster.

---


K-means Clustering and K-means++
# K-means Clustering and K-means++

## Table of Contents
- [K-means Clustering](#k-means-clustering)
- [K-means++](#k-means++)
- [Initialization Sensitivity and K-means++](#initialization-sensitivity-and-k-means++)

## K-means Clustering
- **What is K-means Clustering?**
  - K-means clustering is a popular and simple method for partitioning a dataset into K distinct, non-overlapping subsets (clusters).
- **Geometric Intuition**
  - Imagine each centroid having a region of influence. Each data point belongs to the centroid whose region it lies in. This is akin to partitioning the space into Voronoi cells around each centroid.

### Steps of K-means Clustering
- **Initialization**
  - Randomly select K points from the dataset as the initial centroids (cluster centers).
- **Assignment Step**
  - Assign points to nearest centroid: Each data point in the dataset is assigned to the nearest centroid.
  - Forms K clusters based on the current positions of the centroids.
- **Update Step**
  - Re-compute centroids: Once all points are assigned to clusters, the centroids are recalculated as the mean of all points in each cluster.
  - Geometric Intuition: The new centroid is the geometric

---

## K-means++
- **What is K-means++?**
  - K-means++ is an enhanced version of the standard K-means clustering algorithm.
  - It addresses a key limitation of K-means: the sensitivity to the initial placement of centroids.
- **Key Idea of K-means++**
  - K-means++ improves the initialization step of K-means by ensuring that the initial centroids are spread out across the data space.
  - This increases the likelihood of convergence to a globally optimal solution.

### Steps of K-means++
- **Initialization**
  - Choose the first centroid randomly: Select the first centroid randomly from the data points.

---

## Initialization Sensitivity and K-means++
- **Initialization Sensitivity**
  - The final clusters can depend on the initial choice of centroids, leading to different results.
- **Improving Initialization with K-means++**
  - Methods like K-means++ are used to improve initialization.
  - By iteratively refining the positions of centroids and reassigning points, K-means clustering effectively partitions the data into K cohesive clusters, providing a clear geometric grouping of the dataset.

---

K-Means++ Clustering Algorithm
# K-Means++ Clustering Algorithm

## Table of Contents
- [Introduction to K-Means++](#introduction-to-k-means)
- [Key Idea of K-Means++](#key-idea-of-k-means)
- [Steps of K-Means++](#steps-of-k-means)
- [Advantages and Limitations of K-Means++](#advantages-and-limitations-of-k-means)

## Introduction to K-Means
- **Definition**: K-Means++ is an enhanced version of the standard K-means clustering algorithm.
  - *It addresses a key limitation of K-means: the sensitivity to the initial placement of centroids.*
  - *Poor initialization can lead to suboptimal clustering and slow convergence, as K-means might get stuck in local minima.*

---

## Key Idea of K-Means++
- **Improving Initialization**: K-means++ improves the initialization step of K-means by ensuring that the initial centroids are spread out across the data space.
  - *This increases the likelihood of convergence to a globally optimal solution.*

---

## Steps of K-Means++
- **Initialization**
  1. **Choose the First Centroid Randomly**:
    - Select the first centroid randomly from the data points.
  2. **Iterative Refining**:
    - *Iteratively refine the positions of centroids and reassign points.*
    - *K-means clustering effectively partitions the data into K cohesive clusters, providing a clear geometric grouping of the dataset.*

---

## Advantages and Limitations of K-Means++
- **Advantages**:
  - *K-means++ combines the simplicity of K-means with improved performance, making it a go-to method for many clustering tasks.*
  - *Most modern implementations of K-means, including those in libraries like scikit-learn, use K-means++ by default for centroid initialization.*
- **Limitations**:
  - *K-means++ works well with spherical, equally-sized clusters, but struggles with non-spherical or varied-sized clusters.*
  - *Initialization sensitivity: the final clusters can depend on the initial choice of centroids, leading to different results.*
K-means and K-means++ Clustering Algorithms
# K-means and K-means++ Clustering Algorithms

## Table of Contents
* [K-means Clustering](#k-means-clustering)
* [K-means++](#k-means)
* [Key Idea of K-means++](#key-idea-of-k-means)
* [Steps of K-means++](#steps-of-k-means)

## K-means Clustering
K-means clustering is a popular and simple method for partitioning a dataset into K distinct, non-overlapping subsets (clusters).

- **Initialization**
  - Randomly Select K Centroids: The algorithm begins by randomly selecting K points from the dataset as the initial centroids (cluster centers).
- **Assignment Step**
  - Assign Points to Nearest Centroid: Each data point in the dataset is assigned to the nearest centroid. This forms K clusters based on the current positions of the centroids.
  - **Geometric Intuition**: Imagine each centroid having a region of influence. Each data point belongs to the centroid whose region it lies in. This is akin to partitioning the space into Voronoi cells around each centroid.
- **Update Step**
  - Re-compute Centroids: Once all points are assigned to clusters, the centroids are recalculated as the mean of all points in each cluster.
  - **Geometric Intuition**: The new centroid is the geometric 

---

## K-means++
K-means++ is an enhanced version of the standard K-means clustering algorithm. It addresses a key limitation of K-means: the sensitivity to the initial placement of centroids. Poor initialization can lead to suboptimal clustering and slow convergence, as K-means might get stuck in local minima.

---

## Key Idea of K-means++
K-means++ improves the initialization step of K-means by ensuring that the initial centroids are spread out across the data space. This increases the likelihood of convergence to a globally optimal solution.

---

## Steps of K-means++
### Initialization
1. **Choose the First Centroid Randomly**: Select the first centroid randomly from the data points.

---

**Why K-means++?**
K-means++ is widely used because it combines the simplicity of K-means with improved performance, making it a go-to method for many clustering tasks. Most modern implementations of K-means, including those in libraries like scikit-learn, use K-means++ by default for centroid initialization.
K-Means Clustering
# K-Means Clustering

## Table of Contents
- [K-Means Geometric Intuition](#k-means-geometric-intuition)
- [K-Means Mathematical Formulation](#k-means-mathematical-formulation)

---

## K-Means Geometric Intuition

K-means clustering is a popular and simple method for partitioning a dataset into K distinct, non-overlapping subsets (clusters).

- **Initialization**
  - **Centroid**: The centroid is the mean point (central point) of a cluster.
  - **Randomly Select K Centroids**: The algorithm begins by randomly selecting K points from the dataset as the initial centroids (cluster centers).
- **Assignment Step**
  - **Assign Points to Nearest Centroid**: Each data point in the dataset is assigned to the nearest centroid. This forms K clusters based on the current positions of the centroids.
  - *Geometric Intuition*: Imagine each centroid having a region of influence. Each data point belongs to the centroid whose region it lies in. This is akin to partitioning the space into Voronoi cells around each centroid.

---

## K-Means Mathematical Formulation

The k-means clustering algorithm is an iterative process of moving the centers of clusters or centroids to the mean position of their constituent points, and reassigning instances to their closest clusters iteratively until there is no significant change in the number of cluster centers possible or number of iterations reached.

- **Cost Function**: The cost function of k-means is determined by the Euclidean distance (square-norm) between the observations belonging to that cluster with its respective centroid value.
  - **Intuitive Understanding**: If there is only one cluster (k=1), then the distances between all the observations are compared with its single mean. Whereas, if the number of clusters increases to 2 (k=2), then two-means are calculated and a few of the observations are assigned to cluster 1 and other observations are assigned to cluster two based on proximity.

---

K-Means Clustering Strategies and Techniques
# K-Means Clustering Strategies and Techniques

## Introduction to K-Means Clustering
- **What is K-Means Clustering?**: A popular unsupervised machine learning algorithm used for clustering data points into K clusters based on their similarities.
  - **Goal**: Minimize the Sum of Squared Errors (SSE) between each data point and its assigned cluster centroid.

## Challenges in K-Means Clustering
- **Local Minima**: K-Means typically converges to a local minimum, which may not be the optimal solution.
  - **Solution**: Use various techniques to improve the clustering solution and reduce the SSE.

## Techniques to Improve K-Means Clustering
- **Splitting and Merging Clusters**: Alternate between splitting and merging clusters to escape local minima and produce a better clustering solution.
  - **Splitting Phase**: Divide clusters to reduce SSE.
  - **Merging Phase**: Combine clusters to reduce SSE.

---

## Agglomerative Clustering
- **Definition**: Start with individual data points as clusters and merge the closest pair of clusters at each step.
  - **Requirements**: Define a notion of cluster proximity.
  - **Example**:
```python
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

# Generate random data points
np.random.seed(0)
data_points = np.random.rand(10, 2)

# Perform agglomerative clustering
Z = linkage(data_points, method='ward')

dendrogram(Z)
```

---

## Divisive Clustering
- **Definition**: Start with a single, all-inclusive cluster and split a cluster at each step until only singleton clusters remain.
  - **Requirements**: Decide which cluster to split at each step and how to perform the splitting.
  - **Example**:
```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Generate random data points
np.random.seed(0)
data_points = np.random.rand(10, 2)

# Perform divisive clustering
cluster = AgglomerativeClustering(n_clusters=5, linkage='ward')
cluster.fit(data_points)
```

---

## Table of Contents
- [Introduction to K-Means Clustering](#introduction-to-k-means-clustering)
- [Challenges in K-Means Clustering](#challenges-in-k-means-clustering)
- [Techniques to Improve K-Means Clustering](#techniques-to-improve-k-means-clustering)
- [Agglomerative Clustering](#agglomerative-clustering)
- [Divisive Clustering](#divisive-clustering)


K-Means and K-Means++ Clustering Algorithms
# K-Means and K-Means++ Clustering Algorithms

## Table of Contents
- [K-Means Clustering](#k-means-clustering)
- [K-Means++ Clustering](#k-means-clustering-1)
- [Comparison of K-Means and K-Means++](#comparison-of-k-means-and-k-means)

## K-Means Clustering

- **What is K-Means Clustering?**
  - K-means clustering is a popular and simple method for partitioning a dataset into K distinct, non-overlapping subsets (clusters).
- **Geometric Intuition**
  - Imagine each centroid having a region of influence. Each data point belongs to the centroid whose region it lies in. This is akin to partitioning the space into Voronoi cells around each centroid.

### Steps of K-Means Clustering

- **Initialization**
  - Randomly Select K Centroids: The algorithm begins by randomly selecting K points from the dataset as the initial centroids.
- **Assignment Step**
  - Assign Points to Nearest Centroid: Each data point in the dataset is assigned to the nearest centroid. This forms K clusters based on the current positions of the centroids.
- **Update Step**
  - Re-compute Centroids: Once all points are assigned to clusters, the centroids are recalculated as the mean of all points in each cluster.

---

## K-Means++ Clustering

- **What is K-Means++ Clustering?**
  - K-means++ is an enhanced version of the standard K-means clustering algorithm. It addresses a key limitation of K-means: the sensitivity to the initial placement of centroids.
- **Key Idea of K-Means++**
  - K-means++ improves the initialization step of K-means by ensuring that the initial centroids are spread out across the data space. This increases the likelihood of convergence to a globally optimal solution.

### Steps of K-Means++ Clustering

- **Initialization**
  - Choose the First Centroid Randomly: Select the first centroid randomly from the data points.

### Advantages of K-Means++ Clustering

- **Improved Initialization**
  - K-means++ ensures that the initial centroids are spread out across the data space, leading to better clustering results.
- **Faster Convergence**
  - K-means++ converges faster than K-means due to its improved initialization step.

---

## Comparison of K-Means and K-Means++

- **Similarities**
  - Both K-means and K-means++ are popular clustering algorithms used for partitioning datasets.
- **Differences**
  - K-means is sensitive to the initial placement of centroids, while K-means++ addresses this limitation by ensuring that the initial centroids are spread out across the data space.

---

Hierarchical Clustering Techniques
# Hierarchical Clustering Techniques

## Table of Contents
- [Agglomerative versus Divisive Hierarchical Clustering](#agglomerative-versus-divisive-hierarchical-clustering)
- [Agglomerative Hierarchical Clustering](#agglomerative-hierarchical-clustering)
- [Divisive Hierarchical Clustering](#divisive-hierarchical-clustering)
- [Types of Hierarchical Clustering Methods](#types-of-hierarchical-clustering-methods)
- [Advantages and Limitations](#advantages-and-limitations)
- [Time and Space Complexity](#time-and-space-complexity)
- [Limitations of Hierarchical Clustering](#limitations-of-hierarchical-clustering)

---

## Agglomerative versus Divisive Hierarchical Clustering
- **Definition**: A hierarchical clustering method can be either agglomerative or divisive, depending on whether the hierarchical decomposition is formed in a bottom-up (merging) or top-down (splitting) fashion.
- **Agglomerative Approach**: Starts with each object forming a separate group, and successively merges the objects or groups close to one another.
- **Divisive Approach**: Starts with all the objects in the same cluster, and successively splits the cluster into smaller clusters.

---

## Agglomerative Hierarchical Clustering
- **Bottom-up Strategy**: Starts with each object forming its own cluster, and iteratively merges clusters into larger and larger clusters.
- **Merging Step**: Finds the two clusters that are closest to each other (according to some similarity measure), and combines the two to form one cluster.
- **Termination Condition**: The process stops when all the objects are in a single cluster or a certain termination condition is satisfied.

---

## Divisive Hierarchical Clustering
- **Top-down Strategy**: Starts with all the objects in the same cluster, and successively splits the cluster into smaller clusters.
- **Splitting Step**: Splits a cluster into smaller clusters based on some similarity measure.
- **Termination Condition**: The process stops when each object is in one cluster or a certain termination condition is satisfied.

---

## Types of Hierarchical Clustering Methods
- **Distance-based Methods**: Use a distance metric to measure the similarity between objects.
- **Density- and Continuity-based Methods**: Use density and continuity measures to identify clusters.

---

## Advantages and Limitations
- **Advantages**: Hierarchical clustering methods can handle complex data structures, and provide a visual representation of the data using dendrograms.
- **Limitations**: Hierarchical clustering methods can be computationally expensive, and may not perform well with high-dimensional data.

---

## Time and Space Complexity
- **Time Complexity**: O(n^2), where n is the number of objects.
- **Space Complexity**: O(n), where n is the number of objects.

---

## Limitations of Hierarchical Clustering
- **Sensitive to Initial Conditions**: Hierarchical clustering methods can be sensitive to the initial conditions, such as the choice of distance metric.
- **Difficulty in Handling High-Dimensional Data**: Hierarchical clustering methods can struggle with high-dimensional data, where the number of features is much larger than the number of objects.
Hierarchical Clustering Techniques
# Hierarchical Clustering Techniques

## Table of Contents
- [Hierarchical Clustering](#hierarchical-clustering)
- [Dendrograms](#dendrograms)
- [Agglomerative Clustering](#agglomerative-clustering)
- [Divisive Clustering](#divisive-clustering)
- [Proximity Methods](#proximity-methods)
- [Advantages and Limitations](#advantages-and-limitations)
- [Time and Space Complexity](#time-and-space-complexity)
- [Limitations of Hierarchical Clustering](#limitations-of-hierarchical-clustering)

## Hierarchical Clustering

**Definition:**

Hierarchical clustering is separating data into groups based on some measure of similarity, finding a way to measure how they’re alike and different, and further narrowing down the data. It forms tree like structure called as Dendrogram.

- **Importance of Hierarchical Clustering:**
  - Representing data objects in the form of a hierarchy is useful for data summarization and visualization.
  - While partitioning methods meet the basic clustering requirement of organizing a set of objects into a number of exclusive groups, in some situations we may want to partition our data into groups at different levels such as in a hierarchy.

---

## Dendrograms

**Definition:**

A tree structure called a dendrogram is commonly used to represent the process of hierarchical clustering. It shows how objects are grouped together (in an agglomerative method) or partitioned (in a divisive method) step-by-step.

### Example of Dendrogram

- A dendrogram for the five objects presented in Figure 10.6, where l 0 shows the five objects as singleton clusters at level 0.
- At l 1, objects a and b are grouped together to form the first cluster, and they stay together at all subsequent levels.

### Using a Vertical Axis in Dendrogram

- We can also use a vertical axis to show the similarity scale between clusters.
- For example, when the similarity of two groups of objects, a,b and c,d,e , is roughly 0.16, they are merged together to form a single cluster.

---

## Agglomerative Clustering

Agglomerative clustering is a type of hierarchical clustering that starts with individual data points and groups them together based on their similarity.

### Steps involved in Agglomerative Clustering

- Step 1: Begin with individual data points.
- Step 2: Find the most similar data points and group them together.
- Step 3: Repeat step 2 until all data points are grouped together.

---

## Divisive Clustering

Divisive clustering is a type of hierarchical clustering that starts with a single cluster containing all data points and divides it into smaller clusters based on their similarity.

### Steps involved in Divisive Clustering

- Step 1: Begin with a single cluster containing all data points.
- Step 2: Find the most dissimilar data points and divide the cluster into two sub-clusters.
- Step 3: Repeat step 2 until all data points are divided into individual clusters.

---

## Proximity Methods

Proximity methods are used to measure the similarity between data points in hierarchical clustering.

### Types of Proximity Methods

- **Euclidean Distance:** The most commonly used proximity method, which calculates the straight-line distance between two data points.
- **Manhattan Distance:** A proximity method that calculates the sum of the absolute differences between two data points.
- **Minkowski Distance:** A proximity method that calculates the sum of the differences between two data points raised to a power.

---

## Advantages and Limitations

### Advantages

- **Flexibility:** Hierarchical clustering allows for flexibility in the level of granularity.
- **Visualization:** Hierarchical clustering can be visualized using dendrograms, making it easier to understand the structure of the data.

### Limitations

- **Computational Complexity:** Hierarchical clustering can be computationally intensive, especially for large datasets.
- **Difficulty in Choosing the Number of Clusters:** Hierarchical clustering does not provide a clear way to determine the number of clusters.

---

## Time and Space Complexity

The time complexity of hierarchical clustering is O(n^2), where n is the number of data points. The space complexity is O(n), where n is the number of data points.

---

## Limitations of Hierarchical Clustering

- **Difficult to Handle Noisy Data:** Hierarchical clustering can be sensitive to noisy data and outliers.
- **Difficult to Handle High-Dimensional Data:** Hierarchical clustering can be challenging to apply to high-dimensional data.
Agglomerative Clustering and Hierarchical Clustering Methods
# Agglomerative Clustering and Hierarchical Clustering Methods

## Table of Contents
- [Agglomerative Clustering](#agglomerative-clustering)
- [Agglomerative versus Divisive Hierarchical Clustering](#agglomerative-versus-divisive-hierarchical-clustering)
- [Single-Linkage or MIN or SLINK](#single-linkage-or-min-or-slink)

---

## Agglomerative Clustering

**Definition**: _Agglomerative clustering is a type of hierarchical clustering method that uses a bottom-up strategy to merge clusters into larger and larger clusters until all objects are in a single cluster or certain termination conditions are satisfied._

- **Steps**:
  - Calculate dissimilarity: Calculate the dissimilarity between the objects.
  - Cluster objects: Cluster two objects together if they minimize a given agglomeration criterion when clustered together. This creates a class that contains the two objects.
  - Calculate dissimilarity again: Calculate the dissimilarity between the class and the remaining objects using the agglomeration criterion.
  - Cluster again: Cluster the two objects or classes of objects together if they minimize the agglomeration criterion.
  - Repeat: Repeat steps 2–4 until only one cluster remains.

---

## Agglomerative versus Divisive Hierarchical Clustering

**Agglomerative Hierarchical Clustering**:

* _Bottom-up strategy_: Start by letting each object form its own cluster and iteratively merge clusters into larger and larger clusters.
* _Merging step_: Find the two clusters that are closest to each other (according to some similarity measure), and combine the two to form one cluster.
* _Termination conditions_: Stop when all objects are in a single cluster or certain termination conditions are satisfied.

**Divisive Hierarchical Clustering**:

* _Top-down strategy_: Start by placing all objects in a single cluster and iteratively split the cluster into smaller and smaller clusters.

---

## Single-Linkage or MIN or SLINK

**Definition**: _Single-linkage or MIN or SLINK is a type of agglomerative hierarchical clustering algorithm that uses the minimum distance to measure the distance between clusters._

- **Characteristics**:
  - _Minimum distance measure_: Use the minimum distance between clusters to merge them.
  - _Nearest-neighbor clustering algorithm_: Terminate the clustering process when the distance between nearest clusters exceeds a user-defined threshold.
  - _Minimal spanning tree algorithm_: The resulting graph will generate a tree, and the algorithm is also called a minimal spanning tree algorithm.


Hierarchical Clustering Techniques
# Hierarchical Clustering Techniques

## Table of Contents
- [Introduction](#introduction)
- [Agglomerative and Divisive Hierarchical Clustering](#agglomerative-and-divisive-hierarchical-clustering)
- [Agglomerative Clustering](#agglomerative-clustering)
- [Divisive Clustering](#divisive-clustering)
- [Proximity Methods](#proximity-methods)
- [Advantages and Limitations](#advantages-and-limitations)
- [Time and Space Complexity](#time-and-space-complexity)
- [Limitations of Hierarchical Clustering](#limitations-of-hierarchical-clustering)

---

## Introduction
**Definition**: *A hierarchical method creates a hierarchical decomposition of the given set of data objects.*

A hierarchical method can be classified as being either agglomerative or divisive, based on how the hierarchical decomposition is formed.

- **Agglomerative Approach**: *The agglomerative approach, also called the bottom-up approach, starts with each object forming a separate group.*
  - Successively merges the objects or groups close to one another, until all the groups are merged into one (the topmost level of the hierarchy), or a termination condition holds.
- **Divisive Approach**: *The divisive approach, also called the top-down approach, starts with all the objects in the same cluster.*
  - In each successive iteration, a cluster is split into smaller clusters, until eventually each object is in one cluster, or a termination condition holds.

---

## Agglomerative and Divisive Hierarchical Clustering
- **Agglomerative Hierarchical Clustering**: *An agglomerative hierarchical clustering method uses a bottom-up strategy.*
  - Starts by letting each object form its own cluster and iteratively merges clusters into larger and larger clusters, until all the objects are in a single cluster or certain termination conditions are satisfied.
  - The single cluster becomes the hierarchy’s root.
  - For the merging step, it finds the two clusters that are closest to each other (according to some similarity measure), and combines the two to form one cluster.
- **Divisive Hierarchical Clustering**: *A divisive hierarchical clustering method employs a top-down strategy.*
  - Starts by placing all objects in the same cluster.
  - In each successive iteration, a cluster is split into smaller clusters, until eventually each object is in one cluster, or a termination condition holds.

---

## Agglomerative Clustering
- **Merging Step**: *Two clusters are merged per iteration, where each cluster contains at least one object.*
  - An agglomerative method requires at most n iterations.
- **Similarity Measure**: *The similarity measure is used to find the two clusters that are closest to each other.*
  - Common similarity measures include Euclidean distance, Manhattan distance, and cosine similarity.

---

## Divisive Clustering
- **Splitting Step**: *A cluster is split into smaller clusters in each successive iteration.*
  - The splitting step can be based on various criteria, such as the density of the cluster or the distance between objects.
- **Termination Condition**: *The divisive clustering process stops when each object is in one cluster or a termination condition is satisfied.*
  - Common termination conditions include a maximum number of clusters or a minimum cluster size.

---

## Proximity Methods
- **Proximity Measure**: *A proximity measure is used to calculate the similarity or distance between objects.*
  - Common proximity measures include Euclidean distance, Manhattan distance, and cosine similarity.
- **Proximity Matrix**: *A proximity matrix is used to store the proximity measures between objects.*
  - The proximity matrix is used to determine the similarity between clusters.

---

## Advantages and Limitations
- **Advantages**: *Hierarchical clustering methods can handle large datasets and can be used for both clustering and dimensionality reduction.*
  - Hierarchical clustering methods can also be used to identify clusters at different levels of granularity.
- **Limitations**: *Hierarchical clustering methods can be sensitive to the choice of similarity measure and can be computationally expensive.*
  - Hierarchical clustering methods can also be difficult to interpret, especially for large datasets.

---

## Time and Space Complexity
- **Time Complexity**: *The time complexity of hierarchical clustering methods depends on the similarity measure used and the size of the dataset.*
  - The time complexity can range from O(n^2) to O(n^3), where n is the number of objects.
- **Space Complexity**: *The space complexity of hierarchical clustering methods depends on the size of the dataset and the number of clusters.*
  - The space complexity can range from O(n) to O(n^2), where n is the number of objects.

---

## Limitations of Hierarchical Clustering
- **Sensitivity to Noise**: *Hierarchical clustering methods can be sensitive to noise in the data, which can affect the quality of the clusters.*
- **Difficulty in Choosing the Number of Clusters**: *Hierarchical clustering methods require the user to choose the number of clusters, which can be difficult to determine.*
- **Computational Expense**: *Hierarchical clustering methods can be computationally expensive, especially for large datasets.*
Hierarchical Clustering Techniques
# Hierarchical Clustering Techniques

## Table of Contents
- [Hierarchical Clustering Definition](#hierarchical-clustering-definition)
- [Agglomerative and Divisive Clustering](#agglomerative-and-divisive-clustering)
- [Dendrograms](#dendrograms)
- [Proximity Methods](#proximity-methods)
- [Advantages and Limitations](#advantages-and-limitations)
- [Time and Space Complexity](#time-and-space-complexity)
- [Limitations of Hierarchical Clustering](#limitations-of-hierarchical-clustering)

## Hierarchical Clustering Definition
- **Definition**: *Hierarchical clustering is separating data into groups based on some measure of similarity, finding a way to measure how they’re alike and different, and further narrowing down the data.*
  - It forms tree like structure called as **Dendrogram**.
  - While partitioning methods meet the basic clustering requirement of organizing a set of objects into a number of exclusive groups, in some situations we may want to partition our data into groups at different levels such as in a hierarchy.

---

## Agglomerative and Divisive Clustering
- **Agglomerative Clustering**: This method starts with each data point in its own cluster and then merges the closest clusters until a stopping criterion is met.
  - **Example**: Suppose we have a dataset of customers and we want to group them based on their purchasing behavior. We can start by assigning each customer to their own cluster and then merge the clusters based on their similarity.
- **Divisive Clustering**: This method starts with all data points in a single cluster and then splits the cluster into smaller clusters until a stopping criterion is met.
  - **Example**: Suppose we have a dataset of images and we want to group them based on their features. We can start by assigning all images to a single cluster and then split the cluster into smaller clusters based on their similarity.

---

## Dendrograms
- **Dendrogram**: A dendrogram is a tree-like diagram that shows the hierarchical structure of the clusters.
  - **Example**: A dendrogram can be used to visualize the hierarchical clustering of a dataset.

---

## Proximity Methods
- **Proximity Methods**: Proximity methods are used to measure the similarity between data points.
  - **Example**: Common proximity methods include Euclidean distance, Manhattan distance, and cosine similarity.

---

## Advantages and Limitations
- **Advantages**: Hierarchical clustering methods can handle datasets with varying densities and can be used to identify clusters at different levels of granularity.
  - However, hierarchical methods suffer from the fact that once a step (merge or split) is done, it can never be undone.
- **Limitations**: Hierarchical clustering methods can be sensitive to the choice of proximity method and can be computationally expensive for large datasets.

---

## Time and Space Complexity
- **Time Complexity**: The time complexity of hierarchical clustering methods depends on the proximity method used and the number of data points.
  - **Example**: The time complexity of hierarchical clustering using Euclidean distance is O(n^2), where n is the number of data points.
- **Space Complexity**: The space complexity of hierarchical clustering methods depends on the number of data points and the number of clusters.
  - **Example**: The space complexity of hierarchical clustering using Euclidean distance is O(n), where n is the number of data points.

---

## Limitations of Hierarchical Clustering
- **Limitations**: Hierarchical clustering methods have several limitations, including:
  - **Rigidity**: Hierarchical methods suffer from the fact that once a step (merge or split) is done, it can never be undone.
  - **Sensitivity to Proximity Method**: Hierarchical clustering methods can be sensitive to the choice of proximity method.
  - **Computational Expense**: Hierarchical clustering methods can be computationally expensive for large datasets.

Hierarchical Clustering Techniques
# Hierarchical Clustering Techniques

## Outline
- **Hierarchical Clustering Techniques**
  - **Agglomerative and Divisive Techniques**
  - **Dendrograms**
- **Proximity Methods**
- **Advantages and Limitations**
- **Time and Space Complexity**
- **Limitations of Hierarchical Clustering**

---

## Agglomerative vs Divisive Hierarchical Clustering
- **Agglomerative Hierarchical Clustering**
  - **Bottom-up strategy**: Start with individual objects and iteratively merge clusters
  - **Merging step**: Find closest clusters and combine to form a new cluster
  - **Termination conditions**: All objects in a single cluster or specified conditions met
  - **Root of hierarchy**: Single cluster containing all objects
  - **Number of iterations**: At most n iterations, where n is the number of objects
- **Divisive Hierarchical Clustering**
  - **Top-down strategy**: Start with all objects in a single cluster and iteratively split clusters
  - **Splitting step**: Split a cluster into smaller clusters based on some similarity measure

---

## Space and Time Complexity of Hierarchical Clustering Technique
- **Space Complexity**
  - **High space requirement**: Store similarity matrix in RAM
  - **Space complexity**: O(n²) where n is the number of data points
- **Time Complexity**
  - **High time complexity**: Perform n iterations and update similarity matrix
  - **Time complexity**: O(n³) where n is the number of data points

---

## Advantages and Limitations of Hierarchical Clustering
- **Advantages**
  - _Easy to implement and visualize_
  - _Handles varying densities and shapes of clusters_
- **Limitations**
  - _Sensitive to noise and outliers_
  - _Difficult to determine optimal number of clusters_
  - _Computationally expensive for large datasets_

---

Hierarchical Clustering Techniques
# Hierarchical Clustering Techniques

## Table of Contents
- [Hierarchical Clustering Overview](#hierarchical-clustering-overview)
- [Agglomerative and Divisive Clustering](#agglomerative-and-divisive-clustering)
- [Dendrograms](#dendrograms)
- [Agglomerative Clustering](#agglomerative-clustering)
- [Divisive Clustering](#divisive-clustering)
- [Proximity Methods](#proximity-methods)
- [Advantages and Limitations](#advantages-and-limitations)
- [Time and Space Complexity](#time-and-space-complexity)
- [Limitations of Hierarchical Clustering](#limitations-of-hierarchical-clustering)

## Hierarchical Clustering Overview
- **Definition**: *Hierarchical clustering is separating data into groups based on some measure of similarity, finding a way to measure how they’re alike and different, and further narrowing down the data.*
  - It forms tree like structure called as **Dendrogram**.
  - While partitioning methods meet the basic clustering requirement of organizing a set of objects into a number of exclusive groups, in some situations we may want to partition our data into groups at different levels such as in a hierarchy.
- **Hierarchical Clustering Method**: A hierarchical clustering method works by grouping data objects into a hierarchy or “tree” of clusters.
  - Representing data objects in the form of a hierarchy is useful for data summarization and visualization.

---

## Agglomerative and Divisive Clustering
- **Agglomerative Clustering**: This approach starts with each data point in its own cluster and merges them based on similarity until only one cluster remains.
- **Divisive Clustering**: This approach starts with all data points in one cluster and splits them based on dissimilarity until each data point is in its own cluster.

---

## Dendrograms
- A **Dendrogram** is a tree-like diagram that shows the hierarchical relationship between clusters.
  - Each merge or split is represented by a node in the dendrogram.
  - The height of the node represents the distance between the clusters.

---

## Agglomerative Clustering
- **Steps**:
  1. Start with each data point in its own cluster.
  2. Calculate the similarity between each pair of clusters.
  3. Merge the two most similar clusters.
  4. Repeat steps 2 and 3 until only one cluster remains.
- **Example**:
  ```
  Cluster 1: [1, 2, 3]
  Cluster 2: [4, 5, 6]
  Similarity: 0.8
  Merge: [1, 2, 3, 4, 5, 6]
  ```

---

## Divisive Clustering
- **Steps**:
  1. Start with all data points in one cluster.
  2. Calculate the dissimilarity between each pair of data points.
  3. Split the cluster into two sub-clusters based on the most dissimilar data points.
  4. Repeat steps 2 and 3 until each data point is in its own cluster.
- **Example**:
  ```
  Cluster: [1, 2, 3, 4, 5, 6]
  Dissimilarity: 0.5
  Split: [1, 2, 3] and [4, 5, 6]
  ```

---

## Proximity Methods
- **Definition**: *Proximity methods are used to measure the similarity or dissimilarity between data points.*
  - Common proximity methods include Euclidean distance, Manhattan distance, and cosine similarity.
- **Example**:
  ```
  Data Point 1: [1, 2, 3]
  Data Point 2: [4, 5, 6]
  Euclidean Distance: 5.196
  ```

---

## Advantages and Limitations
- **Advantages**:
  - Hierarchical clustering can handle complex data structures.
  - It can be used for data summarization and visualization.
- **Limitations**:
  - Hierarchical clustering can be computationally expensive.
  - It can be sensitive to noise and outliers.

---

## Time and Space Complexity
- **Time Complexity**: O(n^2), where n is the number of data points.
- **Space Complexity**: O(n), where n is the number of data points.

---

## Limitations of Hierarchical Clustering
- **Rigidity**: Once a merge or split is made, it cannot be undone.
- **Error Propagation**: Errors in the early stages of the algorithm can propagate to the later stages.
- **Scalability**: Hierarchical clustering can be computationally expensive for large datasets.
