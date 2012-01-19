#! /usr/bin/python

import csv
import random
from numpy import *
import scipy

class perceptron:
    
    learning_rate = 0.2
    features = 64
    training_set = []
    training_class = []
    weights = []

    def __init__(self, learning_rate, feature_count, file_name):
        random.seed()

        self.learning_rate = learning_rate
        self.feature = feature_count
        self.training_set = self.get_training_data(file_name)
        self.weights = [random.random() for i in range(0,feature_count+1)]
     
    def get_training_data(self, file_name):
        file_reader = csv.reader(open(file_name, 'r'))
        temp_val = []
        for i in file_reader:
            temp_val.append(i)

        temp_ints = [map(int, i) for i in temp_val]
        
        self.training_set = temp_ints
        for i in temp_ints:
            self.training_class.append(i[-1])
            i[-1] = 1 
        self.training_set = array(self.training_set) 
    def cost_function(self, x, y):
        return 1/2 * (x-y) * (x-y)
    
    def gradient_step(self, cost):
        pass

    def make_array(self, x, y):
        return array(training_set)
    
    def compute():
        pass 






    
if __name__ == "__main__":
    per = perceptron(0.2, 64, "optdigits.tra")
