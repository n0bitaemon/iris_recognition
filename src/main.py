from createDB import createDB
from verifyDB import verify
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random
import pickle
import os

dataset_dir = 'iris_recognition\\CASIA1\\*'
dataset_far_dir = 'iris_recognition\\sample_for_far\\*'
template_dir = 'iris_recognition\\src\\templates\\CASIA1'
db_filename = 'iris_recognition\\src\\database.pkl'

frr_dir = 'iris_recognition\\src\\frr_test'
far_dir = 'iris_recognition\\src\\far_test'
database = None

# Load database
if(os.path.isfile(db_filename)):
    with open(db_filename, 'rb') as db_file:
        database = pickle.load(db_file)
else:
    database = createDB(dataset_dir, template_dir, db_filename)

def convertPath(path):
    return int(os.path.basename(path)[:3])

def compare(filepath, match):
    evaluated_sample = convertPath(filepath)
    result_sample = convertPath(match) if match != None else -1
    result_bool = evaluated_sample == result_sample

    return result_bool

def get_random_samples(dir, n):
    files = glob(os.path.join(dir, "*"))
    if(len(files) <= n):
        return files
    return random.sample(files, n)


def start(threshold):
    while(True):
        print('----------------------')
        print("Input your file path: ")
        filepath = input()

        try:
            match = verify(filepath, database, threshold)
            if(match == None):
                print('No match')
            else:
                print('Found: {}'.format(convertPath(match)))
                print('Result:', 'Correct' if compare(filepath, match) else 'Incorrect')

        except Exception as error:
            print('Error:', error)

# start_dir contains samples that are in database already
def calc_frr_arr(start_dir, threshold):
    files = glob(os.path.join(start_dir, "*"))
    n_files = len(files)
    if n_files == 0:
        return 0
    
    n_match = 0
    for file in files:
        try:
            match = verify(file, database, threshold)
            if(match != None):
                n_match += 1
        except Exception as error:
            print('Error:', error)

    print('n_correct = {}, n_files = {}'.format(n_match, n_files))
    return (1 - n_match/n_files)*100

# start_dir contains samples that are not in database yet
def calc_far_arr(start_dir, threshold):
    files = glob(os.path.join(start_dir, "*"))
    n_files = len(files)
    if n_files == 0:
        return 0
    
    n_match = 0
    for file in files:
        try:
            match = verify(file, database, threshold)
            if(match != None):
                n_match += 1
        except Exception as error:
            print('Error:', error)

    print('n_match = {}, n_files = {}'.format(n_match, n_files))
    return (n_match/n_files)*100

def calc_accuracy(start_dir, threshold, n):
    # files = glob(os.path.join(start_dir, "*"))
    files = get_random_samples(start_dir, n)
    # print(files)
    n_files = len(files)
    if n_files == 0:
        return 0
    
    n_correct = 0
    for file in files:
        try:
            match = verify(file, database, threshold)
            if(compare(file, match)):
                n_correct += 1
        except Exception as error:
            print('Error:', error)

    print('n_correct = {}, n_files = {}'.format(n_correct, n_files))
    return (n_correct/n_files)*100

start(0.37)

##### GENERATE THRESHOLD ARRAY #####
threshold_arr = np.arange(0.1, 0.5, 0.02)
print('threshold_arr =', threshold_arr)

##### CALCULATE FRR #####
# Create test set for calculating frr
# random_samples = get_random_samples(dataset_dir, 100)
# for sample in random_samples:
#     shutil.copy(sample, frr_dir)

# frr_arr = []
# for threshold in threshold_arr:
#     frr = calc_frr_arr('iris_recognition\\src\\frr_test', threshold)
#     print('Threshold = {}, frr = {}%'.format(threshold, frr))
#     frr_arr.append(frr)
# print(frr_arr)


##### CALCULATE FAR #####
# Create test set for calculating far
# random_samples = get_random_samples(dataset_far_dir, 100)
# for sample in random_samples:
#     shutil.copy(sample, far_dir)

# far_arr = []
# for threshold in threshold_arr:
#     far = calc_far_arr('iris_recognition\\src\\far_test', threshold)
#     print('Threshold = {}, far = {}%'.format(threshold, far))
#     far_arr.append(far)
# print(far_arr)

##### DRAW FRR, FAR DIAGRAM #####
# FRR, FAR obtained from previous step
# far_arr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 8.0, 39.0, 75.0, 96.0, 100.0, 100.0, 100.0]
# frr_arr = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 98.80478087649402, 97.21115537848605, 90.0398406374502, 77.68924302788844, 64.5418326693227, 49.40239043824701, 32.66932270916335, 19.123505976095622, 9.16334661354582, 3.1872509960159334, 1.195219123505975, 0.3984063745019917, 0.3984063745019917, 0.0]

# fig, ax = plt.subplots()
# ax.plot(threshold_arr, far_arr, 'r--', label='FAR')
# ax.plot(threshold_arr, frr_arr, 'g--', label='FRR')
# plt.xlabel('Threshold')

# legend = ax.legend(loc='upper center', fontsize='x-large')
# plt.show()


##### ACCURACY EVALUATION #####
# Calculating Accuracy
# n_samples = [20, 40, 60, 80, 100]
# accuracy_arr = []
# for n in n_samples:
#     print('Calculating accuracy for {} samples'.format(n))
#     accuracy = calc_accuracy(frr_dir, 0.37, n)
#     accuracy_arr.append(accuracy)
# Draw accuracy
# accuracy_arr = [90.0, 92.5, 73.33, 81.25, 79.0]
# plt.plot(n_samples, accuracy_arr)
# plt.xlabel('n_samples')
# plt.show()
