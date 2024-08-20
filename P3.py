import random
import matplotlib.pyplot as plt
import numpy as np
import os

def readDataFromFile(filePath):
    try:
        with open(filePath, 'r') as file:
            return file.readlines()
    except FileNotFoundError as e:
        print(f"Error reading data: {e}")
        return None

def getCentroids(num_clusters, dim, clusters):
    
    centroids = [[0] * dim for _ in range(num_clusters)]

    try:
        for cluster_index in range(num_clusters):
            for _, point in enumerate(clusters[cluster_index]):
                for dimension_index in range(dim):
                    centroids[cluster_index][dimension_index] += point[dimension_index]
    except Exception as e:
        print(f"ERROR: {e}")

    return centroids

def getInitialCentroids(numberOfClusters, data):
    try:
        dataPoints = [list(map(float, line.strip().split()[:-1])) for line in data]
        
        # Choose K distinct data points as initial centroids
        centroids = random.sample(dataPoints, numberOfClusters)
        
        clusters = [[] for _ in range(numberOfClusters)]

        # Assign each data point to its nearest initial centroid
        for dataPoint in dataPoints:
            distances = [getEuclideanDistance(dataPoint, centroid) for centroid in centroids]
            index = np.argmin(distances)
            clusters[index].append(dataPoint)
        
        return clusters, dataPoints
    
    except Exception as e:
        print(f"ERROR: {e}")
        return None, None, None

def updateCentroids(K, numberOfDimensions, clusters):
    try:
        centroids = [[0] * numberOfDimensions for _ in range(K)]
        
        for index in range(K):
            if len(clusters[index]) > 0:
                for dimension in range(numberOfDimensions):
                    centroids[index][dimension] = 0
                    
                for pointIndex in range(len(clusters[index])):
                    for dimension in range(numberOfDimensions):
                        centroids[index][dimension] += clusters[index][pointIndex][dimension]
        
        for index in range(K):
            if len(clusters[index]) > 0:
                for dimension in range(numberOfDimensions):
                    centroids[index][dimension] /= len(clusters[index])
    except Exception as e:
        print(f"ERROR: {e}") 
    
    return centroids

def getSSE(num_clusters, dataPoints, centroids):
    clusters = [[] for _ in range(num_clusters)]
    total_error = 0

    try:
        for point in dataPoints:
            distances = [getEuclideanDistance(point, centroid) for centroid in centroids]
            index = np.argmin(distances)
            clusters[index].append(point)
            total_error += min(distances)
    except Exception as e:
        print(f"ERROR: {e}")
    
    return clusters, total_error

def getEuclideanDistance(point1, point2):
    differences = [point1[x] - point2[x] for x in range(len(point1))]
    square_addition = sum([difference ** 2 for difference in differences])
    return square_addition ** 0.5

# Plotting the results
def plotGraph(rangeOfClusters, errors):
    plt.plot(list(rangeOfClusters), errors, marker='o', color='red', linestyle='-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title(f"K-Means Clustering: SSE vs Numbers of Clusters")
    plt.grid(True)
    plt.show()


def main():
    try:
        random.seed(0) 
        filePath = input("Enter the file path: ")
        
        if not os.path.exists(filePath):
            print(f"Error: File '{filePath}' not found.")
            return

        data = readDataFromFile(filePath)
        if not data:
            return

        errors = []
        rangeOfClusters = range(2, 11)
        numberOfIterations = 20

        for K in rangeOfClusters:
            clusters, dataPoints = getInitialCentroids(K, data)
            numberOfDimensions = len(dataPoints[0])
            centroids = getCentroids(K, numberOfDimensions, clusters)
            centroids = updateCentroids(K, numberOfDimensions, clusters)

            for _ in range(numberOfIterations ):
                clusters, sse = getSSE(K, dataPoints, centroids)
                centroids = getCentroids(K, numberOfDimensions, clusters)
                centroids = updateCentroids(K, numberOfDimensions, clusters)

            print(f"For k = {K} After {numberOfIterations} iterations: Error = {sse:.4f}")
            errors.append(sse)

        plotGraph(rangeOfClusters, errors)

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
