#! /usr/bin/python

import csv
import random
from numpy import *
import scipy

class perceptron:
    
    learning_rate = 0.2
    feature_count = 64
    target_digit = 0 
    compaired_digit = 0
    positive_examples = []
    negative_examples = []
    positive_tests = []
    negative_tests = []

    def __init__(self):
        random.seed()

        self.weights = [(random.random() * 2) - 1 for i in range(0,self.feature_count+1)]
         
    def set_testing_data(self, training_data):
        self.positive_tests = []
        self.negative_tests = []

        for i in testing_data[:]:
            
            if i[-1] == self.target_digit:
                self.positive_tests.append(i[:])
                self.positive_tests[-1][-1] = 1
            elif i[-1] == self.compaired_digit:
                self.negative_tests.append(i[:])
                self.negative_tests[-1][-1] = 1
        
    def set_training_data(self, training_data):
        self.positive_examples = []
        self.negative_examples = []
        for i in training_data[:]:
            
            if i[-1] == self.target_digit:
                self.positive_examples.append(i[:])
                self.positive_examples[-1][-1] = 1
            elif i[-1] == self.compaired_digit:
                self.negative_examples.append(i[:])
                self.negative_examples[-1][-1] = 1

    def compute(self, instance):
        total = 0
        for i in range(len(instance)):
            total = total + instance[i] * self.weights[i]
        return total

    def train(self):
        percent_correct = 0
        percent_correct_last_epoch = 0
        change_in_accuracy = 10
        count = 0

        while change_in_accuracy > .5:
            count += 1
            result = self.epoch()
            percent_correct_last_epoch = percent_correct
            try:
                percent_correct = 100 * (float(result[-2]) / (float(result[-1]) + float(result[-2])))
            except ZeroDivisionError:
                pass
            change_in_accuracy = percent_correct - percent_correct_last_epoch
        print("Perceptron %(digit)d trained in %(count)d epochs" % {'count':count,'digit': self.compaired_digit})

    def epoch(self):
        confusion = [0,0,0,0,0,0] #tp,tn,fp,fn,r,w
        
        for i in self.positive_examples:
            result = self.compute(i)
            if result < 0:
                confusion[3] = confusion[3] + 1
                confusion[5] = confusion[5] + 1
                self.adjust_weights(1, -1, i)
            else:
                confusion[0] = confusion[0] + 1
                confusion[4] = confusion[4] + 1
        
        for i in self.negative_examples:
            result = self.compute(i)
            if result > 0:
                confusion[2] = confusion[2] + 1
                self.adjust_weights(-1, 1, i)
                confusion[5] = confusion[5] + 1
            else:
                confusion[1] = confusion[1] + 1
                confusion[4] = confusion[4] + 1
        return confusion 
    def test(self):
        return self.testing_epoch()

    def testing_epoch(self):
        confusion = [0,0,0,0,0,0] #tp,tn,fp,fn,r,w
        
        for i in self.positive_tests:
            result = self.compute(i)
            if result < 0:
                confusion[3] = confusion[3] + 1
                confusion[5] = confusion[5] + 1
            else:
                confusion[0] = confusion[0] + 1
                confusion[4] = confusion[4] + 1
        
        for i in self.negative_tests:
            result = self.compute(i)
            if result > 0:
                confusion[2] = confusion[2] + 1
                confusion[5] = confusion[5] + 1
            else:
                confusion[1] = confusion[1] + 1
                confusion[4] = confusion[4] + 1
        return confusion 

    def adjust_weights(self, expected_output, actual_output, training_example):
        for i in range(len(self.weights)):
            delta = (self.learning_rate * (expected_output - actual_output) * training_example[i])
            self.weights[i] = self.weights[i] + delta

    def set_compaired_digit(self, compair):
        self.compaired_digit = compair

#End class perceptron  **********************************

def get_data(file_name):
    file_reader = csv.reader(open(file_name, 'r'))
    temp_val = []
    
    for i in file_reader:
        temp_val.append(i)

    temp_training_data = [map(float, i) for i in temp_val]
    return temp_training_data


if __name__ == "__main__":
    training_data = get_data("optdigits.tra")
    testing_data = get_data("optdigits.tes")

    per = perceptron()
    
    count = 0
    for i in range(10):
        print("\nTraining perceptron %d" % i)
        per.set_compaired_digit(i)
        per.set_training_data(training_data)
        per.set_testing_data(testing_data)
        per.train()

        print("\nTesting perceptron %d" % i)
        test = per.test()
        print(test, 100 * (float(test[-2]) / (float(test[-2]) + float(test[-1]))))

    #results = per.test()

    #print(results)
    #print(sum(sum(i) for i in results))
