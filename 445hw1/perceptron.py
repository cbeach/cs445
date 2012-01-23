#! /usr/bin/python

import csv
import random
from numpy import *
import scipy

class perceptron:
    
    learning_rate = 0.2
    features = 64
    weights = []
    target_digit = 2

    def __init__(self, learning_rate, feature_count, file_name):
        random.seed()

        self.learning_rate = learning_rate
        self.feature = feature_count
        self.get_training_data(file_name)
        for i in range(10):
            self.weights.append([(random.random() * 2) - 1 for i in range(0,feature_count+1)])
         
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
        self.training_class = temp_class_list

    def cost_function(self, x, y):
        return 1/2 * (x-y) * (x-y)
    
    def gradient_step(self, cost):
        pass

    def make_array(self, x, y):
        return array(training_set)
    
    def compute(self, instance, digit_value):
        return sum([instance[i] * self.weights[digit_value][i] for i in range(65)])

    def train(self):
        
        true_positive = [0 for i in range(10)]
        true_negative = 0  #should I keep trank of all of these with a list?
        false_positive = [0 for i in range(10)]
        false_negative = 0

        for i in range(100): #range(len(self.training_set)):
            #print("Training on example ", i)
            #import pdb; pdb.set_trace()
            if self.training_class[i] == self.target_digit:  #itterate over all examples
                for j in range(10):  #itterate over all perceptrons
                    if j == self.target_digit:  #toss out 2 vs. 2
                        continue
                    result = self.compute(self.training_set[i],j)
                    if result < 0:
                        self.adjust_weights(j, 1, result, self.training_set[i])
                        false_positive[j] = false_positive[j] + 1
                    else:
                        true_positive[j] = true_positive[j] + 1
            else:
                result = self.compute(self.training_set[i], self.training_class[i])
                if result > 0:
                    self.adjust_weights(self.training_class[i], 1, result, self.training_set[i])
                    false_negative = false_negative + 1
                else:
                    true_negative = true_negative + 1 
        return (true_positive, true_negative, false_positive, false_negative)

    def adjust_weights(self, digit_class, expected_output, actual_output, training_example):
        for i in self.weights[digit_class]:
            i = [self.learning_rate * (expected_output - actual_output) * training_example[j] for j in range(65)]

    
if __name__ == "__main__":
    per = perceptron(0.2, 64, "optdigits.tra")
    print(per.train())
