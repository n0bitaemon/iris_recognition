from createDB import createDB
from verifyDB import verify
import pickle
import os

dataset_dir = 'iris_recognition\\CASIA1\\*'
template_dir = 'iris_recognition\\src\\templates\\CASIA1'
db_filename = 'iris_recognition\\src\\database.pkl'
database = None

if(os.path.isfile(db_filename)):
    with open(db_filename, 'rb') as db_file:
        database = pickle.load(db_file)
else:
    database = createDB(dataset_dir, template_dir, db_filename)

def convertPath(path):
    return int(os.path.basename(path)[:3])

while(True):
    print('----------------------')
    print("Input your file path: ")
    filepath = input()
    threshold = 0.40

    try:
        match_arr = verify(filepath, database, threshold)

        print("##### RESULT #####")
        closest = ['', 999999]
        for i in match_arr:
            # print("Sample: {}, distance: {}".format(i[0], i[1]))
            if(i[1] != 0.0 and i[1] < closest[1]):
                closest = i

        evaluated_number = convertPath(filepath)
        result_number = convertPath(closest[0]) if closest[0] != '' else -1
        result_bool = evaluated_number == result_number
        print('Evaluated entity: {}'.format(evaluated_number))
        print('Result: {} => {}'.format(result_number, 'Correct' if result_bool else 'Wrong'))
    except Exception as error:
        print('Error:', error)
