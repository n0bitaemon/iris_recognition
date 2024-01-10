import os
from glob import glob
import numpy as np
from utils.extractandenconding import extractFeature
from utils.imgutils import hamming_distance


def verify(filename, database, threshold):
    # print('\tStart verifying {}\n'.format(filename))

    code = extractFeature(filename)
    for record in database.items():
        # print('Verifyting {}...'.format(record[0]))
        code_in_db = record[1]
        hdis = hamming_distance(code, code_in_db)
        if(hdis != 0.0 and hdis < threshold):
            # print('hdis = {}'.format(hdis))
            return record[0]
    return None

    


