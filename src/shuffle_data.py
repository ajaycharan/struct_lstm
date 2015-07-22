from sklearn.cross_validation import train_test_split as _split
import numpy as np
import sys

'''
This script takes in the list of input data list
and randomly splits the data into train and test
in the proportion 4:1, and writes train and test
file lists.

You can provide the input data list filename
as command line argument.

'''

trainlist = open('trainlist', 'w')
testlist = open('testlist', 'w')

if len(sys.argv) == 2:
    file_list = sys.argv[1]
else:
    file_list = open('img_list')

files = []

for line in file_list: files.append(line.split('\n')[0])

files = map(lambda x: x.split(' '), files)
classes = map(lambda x: int(x[1]), files)
class_set = list(set(classes))

dummy_data = np.arange(len(classes)).reshape((len(classes), 1))
train_data, test_data, train_label, test_label = \
        _split(dummy_data, classes, test_size=0.2, random_state=42)

train_list = [files[i] for i in train_data]
test_list = [files[i] for i in test_data]

train_list = map(lambda x: ' '.join(x) + '\n', train_list)
test_list = map(lambda x: ' '.join(x) + '\n', test_list)

for x in train_list:
    trainlist.write(x)

for x in test_list: testlist.write(x)

trainlist.close()
testlist.close()
