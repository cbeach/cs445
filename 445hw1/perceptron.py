#! /usr/bin/python

import csv
import random
from numpy import *
import scipy

class perceptron:
    
    learning_rate = 0.2
    features = 64
    training_set = None
    training_class = None
    weights = []
    target_digit = 2

    def __init__(self, learning_rate, feature_count, file_name):
        random.seed()

        self.learning_rate = learning_rate
        self.feature = feature_count
        self.get_training_data(file_name)
        self.weights = [random.random() for i in range(0,feature_count+1)]
        self.weights = array(self.weights)
         
    def get_training_data(self, file_name):
        file_reader = csv.reader(open(file_name, 'r'))
        temp_val = []
        for i in file_reader:
            temp_val.append(i)

        temp_ints = [map(int, i) for i in temp_val]
        
        self.training_set = temp_ints

        temp_class_list = []
        for i in temp_ints:
            temp_class_list.append(i[-1])
            i[-1] = 1 

    def cost_function(self, x, y):
        return 1/2 * (x-y) * (x-y)
    
    def gradient_step(self, cost):
        pass

    def make_array(self, x, y):
        return array(training_set)
    
    def compute(self, instance):
        prediction = 0
        total = 0
        for i in range(0, self.features + 1):
            total = total + (instance[i] * self.weights[i])
        return total

    def train(self):
        result = 0
        for i in self.training_set:
            result = self.compute(i)
            if( result > 0):
                pass
                #self.weights = [i-( self.learning_rate * (self.training_class))]
            elif( result < 0):
               pass 
            self.weights = [] 

    
if __name__ == "__main__":
    per = perceptron(0.2, 64, "optdigits.tra")
    per.train()
