#!/usr/bin/python3

import math;
import numpy as np;
import cv2;
import tensorflow as tf;

class Predictor(object):

  def __init__(self):

    # number of distinct limb is 19.
    # list of heatmap id pair. each pair of distinct keypoints define a distinct limb.
    # libSeq.shape = (19,2)
    self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
        [1, 16], [16, 18], [3, 17], [6, 18]];
    # list of paf channel id pair. each pair of paf channel define a vector field (one for x one for y) of a distinct limb.
    # mapIdx.shape = (19,2)
    self.mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
        [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
        [55, 56], [37, 38], [45, 46]];
    # visualize
    self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
        [0, 255, 0], \
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
        [85, 0, 255], \
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]];
    # load model
    self.model = tf.keras.models.load_model('savedmodel');

  def predict(self, oriImg):

    multiplier = [x * 368 / oriImg.shape[0] for x in [0.5, 1, 1.5, 2]]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    for m in range(len(multiplier)):
      scale = multiplier[m]

      imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
      imageToTest_padded, pad = self.padRightDownCorner(imageToTest, 8, 128)

      input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

      # predict with openpose network
      output_blobs = self.model.signatures['serving_default'](tf.cast(input_img, dtype = tf.float32));
      output_blobs = (output_blobs['Mconv7_stage6_L1'], output_blobs['Mconv7_stage6_L2']);
      # extract outputs, resize, and remove padding
      heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
      heatmap = cv2.resize(heatmap, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
      heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
      heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

      paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
      paf = cv2.resize(paf, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
      paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
      paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

      heatmap_avg = heatmap_avg + heatmap / len(multiplier)
      paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
      map_ori = heatmap_avg[:, :, part]
      map = cv2.GaussianBlur(map_ori, (0,0), sigmaX = 3, sigmaY = 3);
      #map = gaussian_filter(map_ori, sigma=3)

      map_left = np.zeros(map.shape)
      map_left[1:, :] = map[:-1, :]
      map_right = np.zeros(map.shape)
      map_right[:-1, :] = map[1:, :]
      map_up = np.zeros(map.shape)
      map_up[:, 1:] = map[:, :-1]
      map_down = np.zeros(map.shape)
      map_down[:, :-1] = map[:, 1:]

      peaks_binary = np.logical_and.reduce(
        (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > 0.1))
      peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
      peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
      id = range(peak_counter, peak_counter + len(peaks))
      peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

      all_peaks.append(peaks_with_score_and_id)
      peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(self.mapIdx)):
      score_mid = paf_avg[:, :, [x - 19 for x in self.mapIdx[k]]]
      candA = all_peaks[self.limbSeq[k][0] - 1]
      candB = all_peaks[self.limbSeq[k][1] - 1]
      nA = len(candA)
      nB = len(candB)
      indexA, indexB = self.limbSeq[k]
      if (nA != 0 and nB != 0):
        connection_candidate = []
        for i in range(nA):
          for j in range(nB):
            vec = np.subtract(candB[j][:2], candA[i][:2])
            norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
            # failure case when 2 body parts overlaps
            if norm == 0:
              continue
            vec = np.divide(vec, norm)

            startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                  np.linspace(candA[i][1], candB[j][1], num=mid_num)))

            vec_x = np.array(
              [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
              for I in range(len(startend))])
            vec_y = np.array(
              [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
              for I in range(len(startend))])

            score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
            score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
              0.5 * oriImg.shape[0] / norm - 1, 0)
            criterion1 = len(np.nonzero(score_midpts > 0.05)[0]) > 0.8 * len(
              score_midpts)
            criterion2 = score_with_dist_prior > 0
            if criterion1 and criterion2:
              connection_candidate.append([i, j, score_with_dist_prior,
                            score_with_dist_prior + candA[i][2] + candB[j][2]])

        connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
        connection = np.zeros((0, 5))
        for c in range(len(connection_candidate)):
          i, j, s = connection_candidate[c][0:3]
          if (i not in connection[:, 3] and j not in connection[:, 4]):
            connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
            if (len(connection) >= min(nA, nB)):
              break

        connection_all.append(connection)
      else:
        special_k.append(k)
        connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(self.mapIdx)):
      if k not in special_k:
        partAs = connection_all[k][:, 0]
        partBs = connection_all[k][:, 1]
        indexA, indexB = np.array(self.limbSeq[k]) - 1

        for i in range(len(connection_all[k])):  # = 1:size(temp,1)
          found = 0
          subset_idx = [-1, -1]
          for j in range(len(subset)):  # 1:size(subset,1):
            if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
              subset_idx[found] = j
              found += 1

          if found == 1:
            j = subset_idx[0]
            if (subset[j][indexB] != partBs[i]):
              subset[j][indexB] = partBs[i]
              subset[j][-1] += 1
              subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
          elif found == 2:  # if found 2 and disjoint, merge them
            j1, j2 = subset_idx
            membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
            if len(np.nonzero(membership == 2)[0]) == 0:  # merge
              subset[j1][:-2] += (subset[j2][:-2] + 1)
              subset[j1][-2:] += subset[j2][-2:]
              subset[j1][-2] += connection_all[k][i][2]
              subset = np.delete(subset, j2, 0)
            else:  # as like found == 1
              subset[j1][indexB] = partBs[i]
              subset[j1][-1] += 1
              subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

          # if find no partA in the subset, create a new subset
          elif not found and k < 17:
            row = -1 * np.ones(20)
            row[indexA] = partAs[i]
            row[indexB] = partBs[i]
            row[-1] = 2
            row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                connection_all[k][i][2]
            subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
      if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
        deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    
    return all_peaks, candidate, subset;

  def sketch(self, canvas, all_peaks, candidate, subset):

    # all_peaks.shape = (18, obj_num, 4): all detected keypoints
    # candidate.shape = (obj_num, 4): 
    # subset.shape = (obj_num, 20): index of point
    for i in range(18):
      for j in range(len(all_peaks[i])):
        cv2.circle(canvas, all_peaks[i][j][0:2], 4, self.colors[i], thickness=-1)

    for i in range(17):
      # for every limb (pair of specific key points)
      for n in range(len(subset)):
        index = subset[n][np.array(self.limbSeq[i]) - 1] # index.shape = (2)
        if -1 in index:
          continue
        cur_canvas = canvas.copy()
        X = candidate[index.astype(int), 0].astype(int) # X.shape = (2)
        Y = candidate[index.astype(int), 1].astype(int) # Y.shape = (2)
        cv2.line(canvas, (X[0], Y[0]), (X[1], Y[1]), self.colors[i], thickness = 1);

    return canvas;

  def visualize(self, canvas, all_peaks, candidate, subset):
    
    for i in range(18):
      for j in range(len(all_peaks[i])):
        cv2.circle(canvas, all_peaks[i][j][0:2], 4, self.colors[i], thickness=-1)
        
    stickwidth = 4

    for i in range(17):
      # for every limb (pair of specific key points)
      for n in range(len(subset)):
        index = subset[n][np.array(self.limbSeq[i]) - 1] # index.shape = (2)
        if -1 in index:
          continue
        cur_canvas = canvas.copy()
        Y = candidate[index.astype(int), 0] # Y.shape = (2)
        X = candidate[index.astype(int), 1] # X.shape = (2)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

  def padRightDownCorner(self, img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

if __name__ == "__main__":
  
  predictor = Predictor();
  img = cv2.imread('test/square_dance.jpg');
  all_peaks, candidate, subset = predictor.predict(img);
  canvas = predictor.sketch(img, all_peaks, candidate, subset);
  cv2.imshow('pose',canvas);
  cv2.waitKey();
