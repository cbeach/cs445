#! /usr/bin/python

import csv
import random
from numpy import *
import scipy

class perceptron:
    
    learning_rate = 0.2
    features = 64
    target_digit = 2
    positive_examples = []
    negative_examples = []

    def __init__(self, learning_rate, feature_count, compared_digit):
        random.seed()
        self.compared_digit = compared_digit
        self.learning_rate = learning_rate
        self.feature = feature_count

        self.weights = [(random.random() * 2) - 1 for i in range(0,feature_count+1)]
         
    def set_training_data(self, training_data):
        for i in training_data:
            
            if i[-1] == self.target_digit:
                i[-1] = 1
                self.positive_examples.append(i)
            elif i[-1] == self.compared_digit:
                i[-1] = 1
                self.negative_examples.append(i)

    def compute(self, instance):
        total = 0
        for i in range(len(instance)):
            total = total + instance[i] * self.weights[i]
        return total

    def train(self):
        for i in range(100):
            self.epoch()

    def epoch(self):
        confusion = [0,0,0,0] #tp,tn,fp,fn
        for i in self.positive_examples:
            result = self.compute(i)
            if result < 0:
                confusion[3] = confusion[3] + 1
                self.adjust_weights(1, -1, i)
            else:
                confusion[0] = confusion[0] + 1
        
        for i in self.negative_examples:
            result = self.compute(i)
            if result > 0:
                confusion[2] = confusion[2] + 1
                self.adjust_weights(-1, 1, i)
            else:
                confusion[1] = confusion[1] + 1
        print(confusion)
            

    def adjust_weights(self, expected_output, actual_output, training_example):
        for i in range(len(self.weights)):
            delta = (self.learning_rate * (expected_output - actual_output) * training_example[i])
            self.weights[i] = self.weights[i] + delta

    def test_compute(self):
        for i in range(len(self.training_set)):
            print(self.compute(self.training_set[i], self.training_class[i]))

#End class perceptron  **********************************

def get_testing_data(self, file_name):
    file_reader = csv.reader(open(file_name, 'r'))
    temp_val = []
    for i in file_reader:
        temp_val.append(i)

    temp_ints = [map(int, i) for i in temp_val]
    self.testing_set = temp_ints

    temp_class_list = []
    for i in temp_ints:
        temp_class_list.append(i[-1])
        i[-1] = 1 
    self.testing_class = temp_class_list

def get_training_data(file_name):
    file_reader = csv.reader(open(file_name, 'r'))
    temp_val = []
    
    for i in file_reader:
        temp_val.append(i)

    temp_training_data = [map(float, i) for i in temp_val]
    return temp_training_data


if __name__ == "__main__":
    #per = [perceptron(0.2, 64, i) for i in range(10)]
    training_data = get_training_data("optdigits.tra")

    per = perceptron(0.0002,64,1)
    per.set_training_data(training_data)
    per.train()
    #for i in per:
    #    i.train()
    #results = per.test()

    #print(results)
    #print(sum(sum(i) for i in results))
