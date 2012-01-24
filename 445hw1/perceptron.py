#! /usr/bin/python

import csv
import random
from numpy import *
import scipy

class perceptron:
    
    learning_rate = 0.1
    feature_count = 64
    target_digit = 9 
    compaired_digit = 0
    positive_examples = []
    negative_examples = []
    positive_tests = []
    negative_tests = []

    def __init__(self):
        random.seed()

        self.weights = [(random.random() * 2) - 1 for i in range(0,self.feature_count+1)]
    #*****************
    #Cherry pick needed values out of the set of testing examples
    #*****************
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
    #more or less the same thing as set_testing_data
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
    
    #***************
    #Compute a value
    #***************
    def compute(self, instance):
        total = 0
        for i in range(len(instance)):
            total = total + instance[i] * self.weights[i]
        return total

    #************************
    #Kinda self explanitory
    #************************
    def train(self):
        percent_correct = 0
        percent_correct_last_epoch = 0
        change_in_accuracy = 10
        count = 0
        
        #stop training when the accuracy stops improving
        while change_in_accuracy > .1:
            count += 1
            result = self.epoch()
            percent_correct_last_epoch = percent_correct
            try:
                percent_correct = 100 * (float(result[-2]) / (float(result[-1]) + float(result[-2])))
            except ZeroDivisionError:
                pass
            change_in_accuracy = percent_correct - percent_correct_last_epoch
        return count

    #******************************
    #run one training epoch
    #******************************
    def epoch(self):
        confusion = [0,0,0,0,0,0] #tp,tn,fp,fn,correct, incorrect
        
        #randomly select instances from either training set to avoid weird results
        while len(self.positive_examples) > 0 or len(self.negative_examples) > 0:
            if random.random() > .5 and len(self.positive_examples) > 0:
                i = self.positive_examples.pop()
                expected = 1
            elif len(self.negative_examples) > 0:
                i = self.negative_examples.pop()
                expected = -1
            else:
                break

            result = self.compute(i)
            if self.sign(result) != expected:
                confusion[3] = confusion[3] + 1
                confusion[5] = confusion[5] + 1
                self.adjust_weights(expected, self.sign(result), i)
            else:
                confusion[0] = confusion[0] + 1
                confusion[4] = confusion[4] + 1
        
        return confusion 

    def test(self):
        return self.testing_epoch()

    def testing_epoch(self):
        confusion = [0,0,0,0,0,0] #tp,tn,fp,fn,correct, 
        
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

    #********************************************
    #Use the gradien decent algorithm to adjust the weights
    #********************************************
    def adjust_weights(self, expected_output, actual_output, training_example):
        for i in range(len(self.weights)):
            delta = (self.learning_rate * (expected_output - actual_output) * training_example[i])
            self.weights[i] = self.weights[i] + delta

    def set_compaired_digit(self, compair):
        self.compaired_digit = compair
    #************************
    #Return the sign of a number
    #************************
    def sign(self, i):
        if i >= 0:
            return 1
        elif i < 0:
            return -1
#End class perceptron  **********************************


#****************************
#Read and parse the csv files
#****************************
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
    test_results = []
    
    #all of this confussing stuff is for outputing latex compatible information
    #Basically my write up was more or less automatically generated ^_^
    
    print("\t\t\\begin{tabular}{ | c || r | r | r |}")
    print("\t\t\t\\hline")
    print("\t\t\t Perceptron & Epochs & Testing Accuracy \\\ " )
    for i in range(10):
        #print("\nTraining perceptron %d" % i)
        per.set_compaired_digit(i)
        per.set_training_data(training_data)
        per.set_testing_data(testing_data)
        epoch = per.train()
        #print("\nTesting perceptron %d" % i)
        test_instance = per.test()
        test_results.append(test_instance)

        print("\t\t\t 9 vs %d & %d & %d \\\ " % (count, epoch, (100.0 * (float(test_instance[-2]) / (float(test_instance[-1])+ float(test_instance[-2]))))))
        print("\t\t\t\\hline")
        count += 1
    print("\t\t\\end{tabular}")
    print("\n")
    print("\t\t\\begin{multicols}{3}")
    count = 0
    for test in test_results:
        print("\t\t\t\\begin{tabular}{| c | c | c |}")
        print("\t\t\t\t\\hline")
        print("\t\t\t\t& 9 & %d \\\ " % count)
        print("\t\t\t\t\\hline")
        print("\t\t\t\t9 & %d & %d \\\ " % (test[0], test[2]))
        print("\t\t\t\t\\hline")
        print("\t\t\t\t%d & %d & %d \\\ " % (count, test[3], test[1]))
        print("\t\t\t\t\\hline")
        print("\t\t\t\\end{tabular}")
        print("\t\t\t\\newline")
        print("\n")
        count += 1
    print("\\end{multicols}")
    print("\\end{document}")
    #results = per.test()

    #print(results)
    #print(sum(sum(i) for i in results))
