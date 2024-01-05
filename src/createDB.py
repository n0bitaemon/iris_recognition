import os
import numpy as np
from glob import glob
from utils.extractandenconding import extractFeature

if __name__ == '__main__':
    dataset_dir = 'iris_recognition\\CASIA1\\*'
    template_dir = 'iris_recognition\\src\\templates\\CASIA1'

    if not os.path.exists(template_dir):
        print("makedirs", template_dir)
        os.makedirs(template_dir)

    files = glob(os.path.join(dataset_dir, "*_1_*.jpg"))
    n_files = len(files)
    print("N# of files which we are extracting features", n_files)

    n_success = 0
    for idx, file in enumerate(files):
        print("#{}. Process file {}... => ".format(idx, file), end='')
        try:
            code = extractFeature(file)
            basename = os.path.basename(file)
            out_file = os.path.join(template_dir, "%s.npy" % (basename))
            np.save(out_file, code)
            print('Success')
            n_success += 1
        except Exception as error:
            print('Error:', error)
            pass
    print("N# of succession:", n_success)