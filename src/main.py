from createDB import createDB
from verifyDB import verify

dataset_dir = 'iris_recognition\\CASIA1\\*'
template_dir = 'iris_recognition\\src\\templates\\CASIA1'

database = createDB(dataset_dir, template_dir)

while(True):
    print("Input your file path: ")
    filepath = input()
    threshold = 0.40

    try:
        match_arr = verify(filepath, database, threshold)

        print("##### RESULT #####")
        for i in match_arr:
            print("Sample: {}, distance: {}".format(i[0], i[1]))
    except Exception as error:
        print('Error:', error)