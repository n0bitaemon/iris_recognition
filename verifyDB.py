from createDB import extractFeature, matchingTemplate
from time import time

template_dir = 'templates/CASIA1/'
filename ='tests/017_2_4.jpg'
threshold = 0.37

print('\tStart verifying {}\n'.format(filename))
template, mask, filename = extractFeature(filename)
result = matchingTemplate(template, mask, template_dir, threshold)

# results 
if result == -1:
    print('\tNo registered sample.')
elif result == 0:
    print('\tNo sample found.')
else:
    print('\tsamples found (desc order of reliability):'.format(len(result)))
    for res in result:
        print("\t", res)

