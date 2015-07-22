import glob
import numpy as np
import os
import hickle as hkl
import caffe
import cv2
from multiprocessing import Pool
from operator import itemgetter
from itertools import groupby


__author__ = 'Srikanth Muralidharan'

'''
This file has two useful helper functions:

The first file extract frames, and also has
several optional inputs that include:

1. videoname(required): Input Video with its path
2. int_: Start-Stop interval(Defualt: (0. np.inf))
3. sample: sampling rate(Default: 1)
4. prefix: prefix path of frames to be written(Default: "frames_v")
5. reshape: Reshape to [227, 227](Default: False)
6. write_frames: Write frames to prefix path(Default:False)

The second useful function is caffenet

It takes no input argument
It provides the Alexnet and also the transformer

We need to provide the path to caffe model, so that it
loads and produces the net successfully.

'''

in_prefix = 'E001/'
out_prefix = 'E001_fc7/'
try:
    os.makedirs(out_prefix)
except:
    pass

videos = glob.glob(in_prefix + '*.mp4')
#print videos


def extract_frame(videoname, int_=None, sample=0, prefix='frames_v', reshape=False, write_frames=False):

    video = cv2.VideoCapture(videoname)
    count = 0
    _, temp = video.read()
    video.release()
    width, height, _ = temp.shape
    video = cv2.VideoCapture(videoname)
    if reshape is False:
        frames = np.zeros((width, height, 3, 0), dtype=np.uint8)
    else:
        frames = np.zeros((227, 227, 3, 0), dtype=np.uint8)
    if sample == 0:
        sample = 1
    if int_ is None:
        start = 0
        end = np.inf
    else:
        start = int_[0]
        end = int_[1]
    while(True):
        x, temp = video.read()
        if x:
            if count < start:
                count += 1
                continue
            elif count >= end:
                break
            count += 1
            if count % sample == 0:
                if reshape is True:
                    temp = cv2.resize(temp, (227, 227))
                if write_frames is True:
                    img_name = prefix + '/' + str(count) + '.jpg'
                    cv2.imwrite(img_name, temp)
                else:
                    frames = np.concatenate((frames, temp[..., np.newaxis]), axis=3)
            if count % 100 == 0:
                print count
        else:
            break
    return frames

class vid_batch(object):

    def __init__(self, video):

        if video.shape[-1] % 15 != 0:
            rem = video.shape[-1] % 15
            l_frame = video[..., -1]
            n_tiles = 15 - rem
            self.video = np.concatenate((video, np.tile(l_frame[..., np.newaxis], [n_tiles])), axis=3)
            self.padded = n_tiles
        else:
            self.video = video
            self.padded = 0


        self.nbatch = self.video.shape[-1]/15


def caffenet():

    caffe_prefix = '/home/sportlogiq/Downloads/caffe-master/models/bvlc_alexnet/'
    net = caffe.Net(caffe_prefix + 'deploy.prototxt', caffe_prefix + 'bvlc_alexnet.caffemodel', caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(caffe_prefix.replace('models/bvlc_alexnet', 'python') + 'caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_raw_scale('data', 255)
    transformer.set_transpose('data', (2, 0, 1))

    return net, transformer

def process_video(filename):

    print filename
    video = extract_frame(filename)
    video = vid_batch(video)
    print 'Video Loaded!'
    net, transformer = caffenet()
    feats = np.zeros((4096, 0), dtype=np.float32)

    net.blobs['data'].reshape(15, 3, 227, 227)
    for x in xrange(video.nbatch):
        frames = video.video[..., 15*x: 15*(x+1)]
        #cur_frames = np.zeros((227, 227, 3, 0), dtype=np.uint8)
        for i in xrange(frames.shape[-1]):
            cur_frame = frames[..., i]
            net.blobs['data'].data[i] = transformer.preprocess('data', cur_frame)
        out = net.forward()
        cur_data = net.blobs['fc7'].data.T
        if x == video.nbatch - 1:
            if video.padded != 0:
                cur_data = cur_data[...,:(15 - video.padded)]
        feats = np.concatenate((feats, cur_data), axis=1)
        print feats.shape

    out_file = filename.replace('E001', 'E001_fc7')
    out_file = out_file.replace('mp4', 'hkl')

    hkl.dump({'feats':feats}, out_file)

    return

if __name__ == '__main__':

    pool = Pool()
    #videos = []
    for x in videos:process_video(x)
    #pool.map(process_video, videos)
