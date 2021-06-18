import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import pickle as pkl

import tensorflow as tf


class Dataset:
    """The base class for all datasets.
    
    Every dataset class should inherit from Dataset 
    and load the data. Dataset only declaires the attributes.
    
    Attributes:
        train_data: A numpy array with data that can be labelled.
        train_labels: A numpy array with labels of train_data.
        test_data: A numpy array with data that will be used for testing.
        test_labels: A numpy array with labels of test_data.
        n_state_estimation: An integer indicating #datapoints reserved for state representation estimation.
        distances: A numpy array with pairwise Eucledian distances between all train_data.
    """
    
    def __init__(self, n_state_estimation):
        """Inits the Dataset object and initialises the attributes with given or empty values."""
        self.train_data = np.array([[]])
        self.train_labels = np.array([[]])
        self.test_data = np.array([[]])
        self.test_labels = np.array([[]])
        self.n_state_estimation = n_state_estimation
        self.regenerate()
        
    def regenerate(self):
        """The function for generating a dataset with new parameters."""
        pass
        
    def _scale_data(self):
        """Scales train data to 0 mean and unit variance. Test data is scaled with parameters of train data."""
        self.train_data = self.train_data/255.0
        self.test_data = self.test_data/255.0

        
    def _keep_state_data(self):
        """self.n_state_estimation samples in training data are reserved for estimating the state."""
        self.train_data, self.state_data, self.train_labels, self.state_labels = train_test_split(
            self.train_data, self.train_labels, test_size=self.n_state_estimation)
        


class MNIST_train(Dataset):      
    """
    Added by Seungbo
    
    Attributes:
        possible_names: A list indicating the dataset names that can be used.
        subset: An integer indicating what subset of data to use. 0: even, 1: odd, -1: all datapoints. 
        size: An integer indicating the size of training dataset to sample, if -1 use all data.
    """
    
    def __init__(self, n_state_estimation, subset, size=-1):
        """Inits a few attributes and the attributes of Dataset object."""
        self.subset = subset
        self.size = size
        Dataset.__init__(self, n_state_estimation) 
    
    def regenerate(self):
        """Loads the data and split it into train and test."""
        # load data
        mnist = tf.keras.datasets.mnist
        (X, y), _ = mnist.load_data()
        X = X.reshape((len(X), 28, 28, 1))
        #X = X.reshape(len(X), -1)
        dtst_size = len(y)
        
        # even datapoints subset
        if self.subset == 0:
            valid_indeces = list(range(0, dtst_size, 2))
        # odd datapoints subset
        elif self.subset == 1:
            valid_indeces = list(range(1, dtst_size, 2))
        # all datapoints
        elif self.subset == -1:
            valid_indeces = list(range(dtst_size))
        else:
            print('Incorrect subset attribute value!')
        
        # try to split data into training and test subsets while insuring that 
        # all classes from test data are present in train data 
        
        # get a part of dataset according to subset (even, odd or all)
        train_test_data = X[valid_indeces,:]
        train_test_labels = y[valid_indeces]
        # use a random half/half split for train and test data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        
        for train_index, test_index in sss.split(train_test_data, train_test_labels):
            self.train_data,self.test_data = X[train_index], X[test_index]
            self.train_labels, self.test_labels = y[train_index], y[test_index]

        self._scale_data()
        self._keep_state_data()
        #self._compute_distances()

        # keep only a part of data for training
        self.train_data = self.train_data[:self.size,:]
        self.train_labels = self.train_labels[:self.size]
        #self.train_labels = to_categorical(self.train_labels, 10)
        # this is needed to insure that some of the classes are missing in train or test data

class MNIST_test(Dataset):      
    """
    Added by Seungbo
    
    Attributes:
        possible_names: A list indicating the dataset names that can be used.
        subset: An integer indicating what subset of data to use. 0: even, 1: odd, -1: all datapoints. 
        size: An integer indicating the size of training dataset to sample, if -1 use all data.
    """
    
    def __init__(self, n_state_estimation, subset, size=-1):
        """Inits a few attributes and the attributes of Dataset object."""
        self.subset = subset
        self.size = size
        Dataset.__init__(self, n_state_estimation) 
    
    def regenerate(self):
        """Loads the data and split it into train and test."""
        # load data
        mnist = tf.keras.datasets.mnist
        _, (X, y) = mnist.load_data()
        X = X.reshape((len(X), 28, 28, 1))
        #X = X.reshape(len(X), -1)
        dtst_size = len(y)
        
        # even datapoints subset
        if self.subset == 0:
            valid_indeces = list(range(0, dtst_size, 2))
        # odd datapoints subset
        elif self.subset == 1:
            valid_indeces = list(range(1, dtst_size, 2))
        # all datapoints
        elif self.subset == -1:
            valid_indeces = list(range(dtst_size))
        else:
            print('Incorrect subset attribute value!')
        
        # try to split data into training and test subsets while insuring that 
        # all classes from test data are present in train data 
        
        # get a part of dataset according to subset (even, odd or all)
        train_test_data = X[valid_indeces,:]
        train_test_labels = y[valid_indeces]
        # use a random half/half split for train and test data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        
        for train_index, test_index in sss.split(train_test_data, train_test_labels):
            self.train_data,self.test_data = X[train_index], X[test_index]
            self.train_labels, self.test_labels = y[train_index], y[test_index]

        self._scale_data()
        self._keep_state_data()
        #self._compute_distances()

        # keep only a part of data for training
        self.train_data = self.train_data[:self.size,:]
        self.train_labels = self.train_labels[:self.size]
        #self.train_labels = to_categorical(self.train_labels, 10)        
        # this is needed to insure that some of the classes are missing in train or test data
