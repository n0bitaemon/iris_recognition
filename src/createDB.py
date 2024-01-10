import os
import pickle
from glob import glob
from utils.extractandenconding import extractFeature

def createDB(dataset_dir, template_dir, db_filename):
    database = {}

    files = glob(os.path.join(dataset_dir, "*.jpg"))
    n_files = len(files)
    print("N# of files which we are extracting features", n_files)

    n_success = 0
    for idx, file in enumerate(files):
        basename = os.path.basename(file)
        print("#{}. Process file {}... => ".format(idx, basename), end='')
        try:
            code = extractFeature(file)
            # out_file = os.path.join(template_dir, "%s.npy" % (basename))
            # np.save(out_file, code)
            database[file] = code
            print('Success')
            n_success += 1
        except Exception as error:
            os.remove(file)
            print('Error:', error)
            print('Removed {}'.format(file))
            pass
    print("N# of succession:", n_success)

    # Save to file

    with open(db_filename, 'wb') as db_file:
        pickle.dump(database, db_file)
        print('Saved database to file {}'.format(db_filename))

    return database