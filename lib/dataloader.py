import tensorflow as tf
from lib.ops import *

import cv2 as cv
import collections, os, math
import numpy as np
from scipy import signal

# The inference data loader. 
# should be a png sequence
def inference_data_loader(FLAGS):
    filedir = FLAGS.input_dir_LR
    downSP = False
    if (FLAGS.input_dir_LR is None) or (not os.path.exists(FLAGS.input_dir_LR)):
        if (FLAGS.input_dir_HR is None) or (not os.path.exists(FLAGS.input_dir_HR)):
            raise ValueError('Input directory not found')
        filedir = FLAGS.input_dir_HR
        downSP = True
        
    image_list_LR_temp = os.listdir(filedir)
    image_list_LR_temp = [_ for _ in image_list_LR_temp if _.endswith(".png")] 
    image_list_LR_temp = sorted(image_list_LR_temp) # first sort according to abc, then sort according to 123
    image_list_LR_temp.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
    if FLAGS.input_dir_len > 0:
        image_list_LR_temp = image_list_LR_temp[:FLAGS.input_dir_len]
        
    image_list_LR = [os.path.join(filedir, _) for _ in image_list_LR_temp]

    # Read in and preprocess the images
    def preprocess_test(name):
        im = cv.imread(name,3).astype(np.float32)[:,:,::-1]
        
        if downSP:
            icol_blur = cv.GaussianBlur(im, (0,0), sigmaX=1.5)
            im = icol_blur[::4,::4,::]
        im = im / 255.0 #np.max(im)
        return im

    image_LR = [preprocess_test(_) for _ in image_list_LR]
    
    if True: # a hard-coded symmetric padding
        image_list_LR = image_list_LR[5:0:-1] + image_list_LR
        image_LR = image_LR[5:0:-1] + image_LR

    Data = collections.namedtuple('Data', 'paths_LR, inputs')
    return Data(
        paths_LR=image_list_LR,
        inputs=image_LR
    )

def loadHR_batch(FLAGS, tar_size):
    # file processing on CPU
    with tf.device('/cpu:0'):
        # Read the file names
        filenames = tf.train.match_filenames_once(os.path.join(FLAGS.input_video_dir, FLAGS.input_video_pre + '*'))
        filename_queue = tf.train.string_input_producer(filenames, shuffle=True, capacity=FLAGS.name_video_queue_capacity)
        
        # Read the video files
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)
        video = tf.image.decode_video(value)
        
        # Process the video frames
        video = tf.cast(video, tf.float32) / 255.0
        video = tf.image.resize(video, [tar_size, tar_size], method=tf.image.ResizeMethod.BILINEAR)
        
        # Create batches
        batch = tf.train.batch([video], batch_size=FLAGS.batch_size, capacity=FLAGS.video_queue_capacity)
        
        return batch

def loadHR(FLAGS, tar_size):
    # a k_w_border margin should be in tar_size for Gaussian blur
    with tf.device('/cpu:0'):
        # Read the file names
        filenames = tf.train.match_filenames_once(os.path.join(FLAGS.input_video_dir, FLAGS.input_video_pre + '*'))
        filename_queue = tf.train.string_input_producer(filenames, shuffle=True, capacity=FLAGS.name_video_queue_capacity)
        
        # Read the video files
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)
        video = tf.image.decode_video(value)
        
        # Process the video frames
        video = tf.cast(video, tf.float32) / 255.0
        video = tf.image.resize(video, [tar_size, tar_size], method=tf.image.ResizeMethod.BILINEAR)
        
        return video

def frvsr_gpu_data_loader(FLAGS, useValData_ph):
    # useValData_ph, tf bool placeholder, whether to use validationdata
    with tf.device('/cpu:0'):
        # Read the file names
        filenames = tf.train.match_filenames_once(os.path.join(FLAGS.input_video_dir, FLAGS.input_video_pre + '*'))
        filename_queue = tf.train.string_input_producer(filenames, shuffle=True, capacity=FLAGS.name_video_queue_capacity)
        
        # Read the video files
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)
        video = tf.image.decode_video(value)
        
        # Process the video frames
        video = tf.cast(video, tf.float32) / 255.0
        video = tf.image.resize(video, [FLAGS.crop_size, FLAGS.crop_size], method=tf.image.ResizeMethod.BILINEAR)
        
        # Create batches
        batch = tf.train.batch([video], batch_size=FLAGS.batch_size, capacity=FLAGS.video_queue_capacity)
        
        return batch
    
