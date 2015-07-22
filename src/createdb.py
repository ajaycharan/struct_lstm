import h5py as h5
import numpy as np
import cv2
import re
import caffe.io as _datum

__author__ = 'Srikanth Muralidharan'

'''

This code creates database compatible with caffe.
It has two functions: processh5 that creates h5
library, and processdb that creates db library.

Both the files receive as input, a text file
that has 'train' or 'test' in as substring
its name. These files contain the list of images
(with the paths) to be processed, along with
their labels.

These files write to the corresponding leveldb/h5
outputs.

'''

def processh5(filename):

    f = open(filename)
    count = 0
    x = re.compile('train')
    y = re.compile('test')

    img_batch = np.zeros((227, 227, 3, 0), dtype=np.float32)
    count = 0
    labels = []
    for line in f:
        line = line.split('\n')[0]
        line = line.split(' ')
        img = cv2.imread(line[0])
        img = cv2.resize(img, (227, 227))
        img_batch = np.concatenate((img_batch, img[..., np.newaxis]), axis=3)
        count += 1
        labels.append(int(line[1]))
        if count % 20 == 0:
            if x.search(filename):
                db_f = h5.File('trainh5/train_' + str(count/20)+ '.h5', 'w')
            elif y.search(filename):
                db_f = h5.File('testh5/test_' + str(count/20) + '.h5', 'w')

            db_f['data'] = img_batch
            db_f['label'] = labels
            db_f.close()
            img_batch = np.zeros((227, 227, 3, 0), dtype=np.float32)
            labels = []

    if count % 20 != 0:
        if x.search(filename):
            db_f = h5.File('trainh5/train_' + str(count/20 + 1)+ '.h5', 'w')
        elif y.search(filename):
            db_f = h5.File('testh5/test_' + str(count/20 + 1) + '.h5', 'w')

        db_f['data'] = img_batch
        db_f['label'] = labels
        db_f.close()

    return

def processdb(filename):

    f = open(filename)
    count = 0
    x = re.compile('train')
    y = re.compile('test')
    if re.search(filename, x):
        db_f = plyvel.DB('train.db')
    elif re.search(filename, y):
        db_f = plyvel.DB('test.db')

    db = db_f.write_batch()

    for line in filename:

        line = line.split('\n')[0]

        line = line.split(' ')
        img = cv2.imread(line[0])
        datum = _datum(img, int(line[1]))

        db.put('%08d' % (count, datum.SerializeToString()))
        count += 1
        if count % 20 == 0:

            db.write()
            del db
            db = db_f.write_batch()

        if count % 20 != 0:
            db.write()

    return

if __name__ == '__main__':

    filename = ['trainlist', 'testlist']

    for x in filename:
        processh5(x)
