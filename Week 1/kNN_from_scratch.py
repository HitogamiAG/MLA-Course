import numpy as np

class kNearestNeighbors:
    """
    kNearestNeighbors algorithm realization from scratch
    """
    def __init__(self,
                 k: int,
                 distance_metric: str,
                 ) -> None:
        """Initialize kNN model

        Args:
            k (int): Number of neighbors
            distance_metric (str): Name of distance metric
        """
        self.k = k
        
        assert distance_metric in ['euclidean', 'cosine'], 'Specified metric does not realized'
        self.distance_func = self.euclidean_func if distance_metric == 'euclidean' else self.cosine_func
        
        self.x_train = None
        self.y_train = None
        
    def euclidean_func(self, data_matrix: np.array, data_vector: np.array) -> np.array:
        """Measure euclidean distance between train set and new vector

        Args:
            data_matrix (np.array): Train set
            data_vector (np.array): New vector

        Returns:
            np.array: Distances to train vectors
        """
        assert data_matrix.shape[1] == data_vector.shape[0], "Number of features in fitted data and new vector didn't match"
        
        distance = np.sqrt( np.power(data_matrix - data_vector, 2).sum(axis = 1))
        return distance
    
    def cosine_func(self, data_matrix: np.array, data_vector: np.array) -> np.array:
        """Measure cosine distance between train set and new vector

        Args:
            data_matrix (np.array): Train set
            data_vector (np.array): New vector

        Returns:
            np.array: Distances to train vectors
        """
        assert data_matrix.shape[1] == data_vector.shape[0], "Number of features in fitted data and new vector didn't match"
        
        distance = 1 - (np.dot(data_matrix, data_vector) / (np.linalg.norm(data_matrix, axis=1) * np.linalg.norm(data_vector)))
        return distance
    
    def fit(self, x_train:np.array, y_train:np.array):
        """Load train set into the model

        Args:
            x_train (np.array): Independent features
            y_train (np.array): Classes
        """
        assert x_train.shape[0] == y_train.shape[0], "y_train size doesn't match with x_train records"
        assert not np.count_nonzero(np.isnan(x_train)) or not np.count_nonzero(np.isnan(y_train)), "NaN values in x_train or y_train"
        
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self, x_test):
        """Predict class for new vectors

        Args:
            x_test (np.array): New vectors (in form of matrix)

        Returns:
            int: vector of classes
        """
        assert self.x_train is not None and self.y_train is not None, 'Use fit() function to train model'
        assert self.x_train.shape[1] == x_test.shape[1], 'Mismatch in number of features between train and test'
        
        preds = np.zeros((len(x_test)))
        for idx, test_vector in enumerate(x_test):
            distance_vector = self.distance_func(self.x_train, test_vector)
            classes = self.y_train[np.argpartition(distance_vector, self.k)[:self.k]]
            classes, counts = np.unique(classes, return_counts=True)
            preds[idx] = classes[np.argmax(counts)]
            
        return preds
        
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    x, y = make_classification(n_samples = 200
                           ,n_features = 2
                           ,n_informative = 2
                           ,n_redundant = 0
                           ,n_clusters_per_class = 1
                           ,flip_y = 0
                           ,class_sep = 2
                           ,random_state = 7
                           )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    knn_clf = kNearestNeighbors(3, 'euclidean')
    knn_clf.fit(x_train, y_train)
    preds = knn_clf.predict(x_test)
    
    print(f'Accuracy: {sum(preds == y_test) / len(preds) * 100}%')
    
    plt.style.use('fivethirtyeight')
    sns.scatterplot(x = x_train[:, 0], y = x_train[:, 1], hue=y_train, palette=['green', 'orange'])
    sns.scatterplot(x = x_test[:, 0], y = x_test[:, 1], hue = preds, palette=['red', 'blue'])
    plt.show()
    
    # uniform_matrix = np.random.uniform(-5, 5, size=(500, 2))
    # preds = knn_clf.predict(uniform_matrix)
    
    # plt.style.use('fivethirtyeight')
    # sns.scatterplot(x = x_train[:, 0], y = x_train[:, 1], hue=y_train, palette=['green', 'orange'])
    # sns.scatterplot(x = uniform_matrix[:, 0], y = uniform_matrix[:, 1], hue = preds, palette=['red', 'blue'])
    # plt.show()
