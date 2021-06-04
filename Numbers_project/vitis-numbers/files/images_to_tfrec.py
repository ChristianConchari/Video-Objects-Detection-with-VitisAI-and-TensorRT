import os
import argparse
import zipfile
import random
import shutil
from tqdm import tqdm

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf


DIVIDER = '-----------------------------------------'


def _bytes_feature(value):
  '''Returns a bytes_list from a string / byte'''
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  '''Returns a float_list from a float / double'''
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  ''' Returns an int64_list from a bool / enum / int / uint '''
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _calc_num_shards(img_list, img_shard):
  ''' calculate number of shards'''
  last_shard =  len(img_list) % img_shard
  if last_shard != 0:
    num_shards =  (len(img_list) // img_shard) + 1
  else:
    num_shards =  (len(img_list) // img_shard)
  return last_shard, num_shards



def write_tfrec(tfrec_filename, image_dir, img_list):
  ''' write TFRecord file '''

  with tf.io.TFRecordWriter(tfrec_filename) as writer:

    for img in img_list:

      class_name,_ = img.split('_',1)
      if class_name == 'zero':
        label = 0
      else:
        label = 1
      filePath = os.path.join(image_dir, img)

      # read the JPEG source file into a tf.string
      image = tf.io.read_file(filePath)

      # get the shape of the image from the JPEG file header
      image_shape = tf.io.extract_jpeg_shape(image)

      # features dictionary
      feature_dict = {
        'label' : _int64_feature(label),
        'height': _int64_feature(image_shape[0]),
        'width' : _int64_feature(image_shape[1]),
        'chans' : _int64_feature(image_shape[2]),
        'image' : _bytes_feature(image)
      }

      # Create Features object
      features = tf.train.Features(feature = feature_dict)

      # create Example object
      tf_example = tf.train.Example(features=features)

      # serialize Example object into TFRecord file
      writer.write(tf_example.SerializeToString())

  return




def make_tfrec(dataset_dir,tfrec_dir,img_shard):

  # remove any previous data
  shutil.rmtree(dataset_dir, ignore_errors=True)    
  shutil.rmtree(tfrec_dir, ignore_errors=True)
  os.makedirs(dataset_dir)   
  os.makedirs(tfrec_dir)

  # unzip the zeros-vs-twos archive that was downloaded from Kaggle
  zip_ref = zipfile.ZipFile('numbers.zip', 'r')
  zip_ref.extractall(dataset_dir)
  zip_ref.close()
  
  # make a list of all images
  imageList = os.listdir(os.path.join(dataset_dir,'train'))

  # make lists of images according to their class
  twoImages=[]
  zeroImages=[]
  for img in imageList:
    class_name,_ = img.split('_',1)
    if class_name == 'two':
        twoImages.append(img)
    else:
        zeroImages.append(img)
  assert(len(twoImages)==len(zeroImages)), 'Number of images in each class do not match'

  # define Train/test split as 80:20
  split = int(len(zeroImages) * 0.2)

  testImages = zeroImages[:split] + twoImages[:split]
  trainImages = zeroImages[split:] + twoImages[split:]
  random.shuffle(testImages)
  random.shuffle(trainImages)


  ''' Test TFRecords '''
  print('Creating test TFRecord files...')

  # how many TFRecord files?
  last_shard, num_shards = _calc_num_shards(testImages, img_shard)
  print (num_shards,'TFRecord files will be created.')

  if (last_shard>0):
    print ('Last TFRecord file will have',last_shard,'images.')
  
  # create TFRecord files (shards)
  start = 0
  for i in tqdm(range(num_shards)):    
    tfrec_filename = 'test_'+str(i)+'.tfrecord'
    write_path = os.path.join(tfrec_dir, tfrec_filename)
    if (i == num_shards-1):
      write_tfrec(write_path, dataset_dir+'/train', testImages[start:])
    else:
      end = start + img_shard
      write_tfrec(write_path, dataset_dir+'/train', testImages[start:end])
      start = end

  # move test images to a separate folder for later use on target board
  shutil.rmtree(dataset_dir+'/test', ignore_errors=True)    
  os.makedirs(dataset_dir+'/test')
  for img in testImages:
    shutil.move( os.path.join(dataset_dir,'train',img),  os.path.join(dataset_dir,'test',img) )


  ''' Training TFRecords '''
  print('Creating training TFRecord files...')

  # how many TFRecord files?
  last_shard, num_shards = _calc_num_shards(trainImages, img_shard)
  print (num_shards,'TFRecord files will be created.')
  if (last_shard>0):
    print ('Last TFRecord file will have',last_shard,'images.')
  
  # create TFRecord files (shards)
  start = 0
  for i in tqdm(range(num_shards)):    
    tfrec_filename = 'train_'+str(i)+'.tfrecord'
    write_path = os.path.join(tfrec_dir, tfrec_filename)
    if (i == num_shards-1):
      write_tfrec(write_path, dataset_dir+'/train', trainImages[start:])
    else:
      end = start + img_shard
      write_tfrec(write_path, dataset_dir+'/train', trainImages[start:end])
      start = end

  
  # delete training images to save space
  # test images are  not deleted as they will be used on the target board
  shutil.rmtree(dataset_dir+'/train')   

  
  print('\nDATASET PREPARATION COMPLETED')
  print(DIVIDER,'\n')

  return
    


def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d', '--dataset_dir', type=str, default='dataset', help='path to dataset images')
  ap.add_argument('-t', '--tfrec_dir',   type=str, default='tfrecords', help='path to TFRecord files')
  ap.add_argument('-s', '--img_shard',   type=int, default=2000,  help='Number of images per shard. Default is 1000') 
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('DATASET PREPARATION STARTED..')
  print('Command line options:')
  print (' --dataset_dir  : ',args.dataset_dir)
  print (' --tfrec_dir    : ',args.tfrec_dir)
  print (' --img_shard    : ',args.img_shard)

  make_tfrec(args.dataset_dir,args.tfrec_dir,args.img_shard)

if __name__ == '__main__':
    run_main()
