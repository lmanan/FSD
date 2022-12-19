import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tifffile
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from FSD.utils import invert_one_hot


def cart2pol(x, y):
  """
  Converts the cartesian coordinates to polar coordinates
  Parameters
  -------

  x: x coordinate of pixel

  y: y coordinate of pixel

  Returns
  -------
  (rho, phi): tuple

    """
  rho = np.sqrt(x ** 2 + y ** 2)
  phi = np.arctan2(y, x)  # domain is (-pi, pi)
  phi = np.where(phi < 0, 2 * np.pi + phi, phi)
  return (rho, phi)


def get_starting_index(y, x, y_interior, x_interior):
  """
  Finds the index of the boundary pixel which is at theta = 0 w.r.t the interior pixel (y_interior, x_interior)
  and has the least radial distance

  Parameters
  -------

  x: x coordinates of boundary pixels

  y: y coordinates of boundary pixels

  y_interior: y coordinate of one interior pixel

  x_interior : x coordinate of one interior pixel
    For simplicity, this interior pixel is assumed to be the object centroid


  Returns
  -------
  index: index of that pixel within the set of boundary pixels
      which is at theta = 0 degrees with respect to the interior pixel and also has the least radial distance
      in case of multiple boundary pixel candidates at theta = 0

  """

  rho, phi = cart2pol(x - x_interior, y - y_interior)
  # sort first by phi ascending, then by r
  indices = np.lexsort((rho, phi))
  return indices[0]


# https://github.com/machine-shop/deepwings/blob/6526066a08843e1dc4c16063bf820e3975d8171a/deepwings/method_features_extraction/image_processing.py#L156-L245
def moore_neighborhood(current, backtrack):  # y, x
  """Returns clockwise list of pixels from the moore neighborhood of current\
  pixel:
  The first element is the coordinates of the backtrack pixel.
  The following elements are the coordinates of the neighboring pixels in
  clockwise order.
  Parameters
  ----------
  current ([y, x]): Coordinates of the current pixel
  backtrack ([y, x]): Coordinates of the backtrack pixel
  Returns
  -------
  List of coordinates of the moore neighborood pixels, or 0 if the backtrack
  pixel is not a current pixel neighbor
  """

  operations = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1],
                         [0, -1], [-1, -1]])
  neighbors = (current + operations).astype(int)

  for i, point in enumerate(neighbors):
    if np.all(point == backtrack):
      # we return the sorted neighborhood
      return np.concatenate((neighbors[i:], neighbors[:i]))
  return 0


def make_contour_from_point_coordinates(y, x, y_interior, x_interior):
  """
  Returns an array which contains the boundary pixel coordinates in x and y arranged sequentially


  Parameters
  -------

  x: x coordinates of boundary pixels

  y: y coordinates of boundary pixels

  y_interior: y coordinate of one interior pixel

  x_interior : x coordinate of one interior pixel
    For simplicity, this interior pixel is assumed to be the object centroid


  Returns
  -------
  numpy array : contains the y and x coordinates of the boundary pixels arraneged sequentially in a CW fashion

  """

  index_start = get_starting_index(y, x, y_interior, x_interior)

  coords = np.array(list(zip(y, x)))
  maxs = np.amax(coords, axis=0)
  binary = np.zeros((maxs[0] + 2, maxs[1] + 2))
  binary[tuple([y, x])] = 1

  while True:  # asserting that the starting point is not isolated
    start = [y[index_start], x[index_start]]
    focus_start = binary[start[0] - 1:start[0] + 2, start[1] - 1:start[1] + 2]
    if np.sum(focus_start) > 1:
      break
    index_start += 1

  # Determining backtrack pixel for the first element
  if (binary[start[0] + 1, start[1]] == 0 and
    binary[start[0] + 1, start[1] - 1] == 0):
    backtrack_start = [start[0] + 1, start[1]]
  else:
    backtrack_start = [start[0], start[1] - 1]

  current = start
  backtrack = backtrack_start
  boundary = []
  counter = 0

  while True:
    neighbors_current = moore_neighborhood(current, backtrack)
    y = neighbors_current[:, 0]
    x = neighbors_current[:, 1]
    idx = np.argmax(binary[tuple([y, x])])
    boundary.append(current)
    backtrack = neighbors_current[idx - 1]
    current = neighbors_current[idx]
    counter += 1

    if (np.all(current == start) and np.all(backtrack == backtrack_start)):
      break

  return np.array(boundary)[:, 0], np.array(boundary)[:, 1]


def get_sequential_boundary_pixels(instance, y_interior, x_interior, mode='outer'):
  boundary = find_boundaries(instance, mode=mode)
  y_b, x_b = np.where(boundary)
  y_b_seq, x_b_seq = make_contour_from_point_coordinates(y_b, x_b, y_interior, x_interior)
  return y_b_seq, x_b_seq


def process(dir_name, fsd_dir_name='fsd/'):
  """
  Processes a complete dataset containing one-hot encoded GT label masks


  Parameters
  -------

  dir_name: str
      path to the the directories `images` and `masks'
      For example, for the BBBC020 dataset, this should be set equal to '../../../data/BBBC020'
  fsd_dir_name : str
      path to a new directory where the frequency components are saved



  Returns
  -------

  """
  if not os.path.exists(dir_name + '/' + fsd_dir_name):
    os.makedirs(os.path.dirname(dir_name + '/' + fsd_dir_name))
    print("Created new directory : {}".format(dir_name + '/' + fsd_dir_name))

  label_mask_names = sorted(glob(os.path.join(dir_name, 'masks', '*.tif')))
  for label_mask_name in tqdm(label_mask_names):
    label_mask = tifffile.imread(label_mask_name)
    for z in range(label_mask.shape[0]):
      im = label_mask[z] == 255
      y_in, x_in = np.where(im)
      y_m, x_m = np.mean(y_in), np.mean(x_in)
      y, x = get_sequential_boundary_pixels(im, int(y_m), int(x_m))
      X = 1 / len(x) * scipy.fft.fft(x)
      Y = 1 / len(y) * scipy.fft.fft(y)
      np.savez(
        os.path.join(dir_name, fsd_dir_name) + os.path.basename(label_mask_name)[:-4] + '+' + str(z).zfill(2) + '.npz',
        X=X, Y=Y)


def perform_idft(X, num_frequencies, number_of_points=100):
  x_recon = []
  for i in range(number_of_points):
    temp = 0
    for l in range(-num_frequencies // 2, num_frequencies // 2):
      temp += X[l] * np.exp(1j * 2 * np.pi * i * l / number_of_points)
    x_recon.append(temp)
  return x_recon


def reconstruct_label_mask(im_path, label_path, fsd_dir_name='fsd/', num_frequencies=10, new_cmp='magma'):
  """
    Visualizes 2 x 2 grid with Top-Left (Image), Top-Right (Ground Truth), Bottom-Right (Reconstruction),
    Parameters
    -------
    image: str
        Path to the raw Image
    label_path: str
        Path to the GT label masks
    num_frequencies: int
        Default = 10
    new_cmp: Color Map
        Default = 'magma'
    Returns
    -------
  """

  font = {'family': 'serif',
          'color': 'white',
          'weight': 'bold',
          'size': 16,
          }
  plt.figure(figsize=(15, 15))
  image = tifffile.imread(im_path)
  img_show = image if image.ndim == 2 else image[..., 0]
  plt.subplot(122);
  plt.axis('off')
  ground_truth = tifffile.imread(label_path)
  plt.imshow(invert_one_hot(ground_truth), cmap=new_cmp, interpolation='None')
  plt.text(30, 30, "GT", fontdict=font)
  plt.xlabel('Ground Truth')
  npz_file_names = sorted(glob(os.path.join(fsd_dir_name, os.path.basename(label_path)[:-4] + '*')))

  plt.subplot(121);
  plt.imshow(img_show, cmap='magma', alpha=0.5);
  for z in range(ground_truth.shape[0]):
    npzfile = np.load(npz_file_names[z])
    X = npzfile['X']
    Y = npzfile['Y']
    x_recon = perform_idft(X, num_frequencies=num_frequencies)
    y_recon = perform_idft(Y, num_frequencies=num_frequencies)
    plt.plot(np.real(x_recon), np.real(y_recon), 'r-.')
  plt.axis('off')
  plt.text(30, 30, "Reconstruction", fontdict=font)
  plt.xlabel('Reconstruction')
  plt.tight_layout()
  plt.show()
