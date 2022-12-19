import os
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np


def extract_data(zip_url, project_name, data_dir='../../../data/'):
  """
      Extracts data from `zip_url` to the location identified by `data_dir` and `project_name` parameters.

      Parameters
      ----------
      zip_url: string
          Indicates the external url
      project_name: string
          Indicates the path to the sub-directory at the location identified by the parameter `data_dir`
      data_dir: string
          Indicates the path to the directory where the data should be saved.
      Returns
      -------

  """
  zip_path = os.path.join(data_dir, project_name + '.zip')

  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print("Created new directory {}".format(data_dir))

  if (os.path.exists(zip_path)):
    print("Zip file was downloaded and extracted before!")
  else:
    if (os.path.exists(os.path.join(data_dir, project_name, 'download/'))):
      pass
    else:
      os.makedirs(os.path.join(data_dir, project_name, 'download/'))
      urllib.request.urlretrieve(zip_url, zip_path)
      print("Downloaded data as {}".format(zip_path))
      with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
      print("Unzipped data to {}".format(os.path.join(data_dir, project_name, 'download/')))


def invert_one_hot(image):
  """
      Inverts a one-hot label mask.

      Parameters
      ----------
      image : numpy array (I x H x W)
          Label mask present in one-hot fashion (i.e. with 0s and 1s and multiple z slices)
          here `I` is the number of GT or predicted objects
      Returns
      -------
      numpy array (H x W)
          A flattened label mask with objects labelled from 1 ... I
  """
  instance = np.zeros((image.shape[1], image.shape[2]), dtype="uint16")
  for z in range(image.shape[0]):
    instance = np.where(image[z] > 0, instance + z + 1, instance)
    # TODO - Alternate ways of inverting one-hot label masks would exist !!
  return instance


def visualize(image, ground_truth, new_cmp):
  """
      Visualizes 2 x 2 grid with Top-Left (Image), Top-Right (Ground Truth), Bottom-Left (Seed),
      Bottom-Right (Instance Segmentation Prediction)

      Parameters
      -------

      image: Numpy Array (YXC)
          Raw RGB style image
      ground_truth: Numpy Array (YX)
          GT Label Mask
      new_cmp: Color Map

      Returns
      -------

      """

  font = {'family': 'serif',
          'color': 'white',
          'weight': 'bold',
          'size': 16,
          }
  plt.figure(figsize=(15, 15))
  img_show = image if image.ndim == 2 else image[..., 0]
  plt.subplot(121);
  plt.imshow(img_show, cmap='magma');
  plt.text(30, 30, "IM", fontdict=font)
  plt.xlabel('Image')
  plt.axis('off')
  if (ground_truth is not None):
    plt.subplot(122);
    plt.axis('off')
    plt.imshow(ground_truth, cmap=new_cmp, interpolation='None')
    plt.text(30, 30, "GT", fontdict=font)
    plt.xlabel('Ground Truth')
  plt.tight_layout()
  plt.show()
