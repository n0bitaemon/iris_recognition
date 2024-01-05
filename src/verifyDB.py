import os
from glob import glob
import numpy as np
from utils.extractandenconding import matchingTemplate, extractFeature


if __name__ == '__main__':
    filename = 'iris_recognition\\src\\tests\\002_1_1.jpg'
    template_dir = 'iris_recognition\\src\\templates\\CASIA1'
    threshold = 0.15
    
    print('\tStart verifying {}\n'.format(filename))

    if not os.path.exists(template_dir):
        print('No match')
        exit()

    files = glob(os.path.join(template_dir, "*_1_*.jpg.npy"))
    n_files = len(files)
    print("N# of files which we are extracting features", n_files)

    match_arr = []
    for file in files:
        print('Verifyting {}...'.format(file))
        code1 = np.load(file)
        code2 = extractFeature(filename)
        if(matchingTemplate(code1, code2, threshold=0.125)):
            match_arr.append(file)

    print(match_arr)