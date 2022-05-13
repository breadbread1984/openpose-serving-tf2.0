#!/usr/bin/python3

import math;
import numpy as np;
import cv2;
import tensorflow as tf;

class OpenPose(object):
  # number of distinct limb is 19.
  # list of heatmap id pair. each pair of distinct keypoints define a distinct limb.
  # libSeq.shape = (19,2)
  limbSeq = [[2, 3],   [2, 6],   [3, 4],   [4, 5],   [6, 7], [7, 8],  [2, 9],   [9, 10], \
             [10, 11], [2, 12],  [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], \
             [16, 18], [3, 17],  [6, 18]]
  # list of paf channel id pair. each pair of paf channel define a vector field (one for x one for y) of a distinct limb.
  # mapIdx.shape = (19,2)
  mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
            [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
            [55, 56], [37, 38], [45, 46]]
  # visualize
  colors = [[255, 0, 0],   [255, 85, 0],  [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85],  [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],  [0, 0, 255],  [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
  def __init__(self, model):
    self.model = tf.keras.models.load_model(model);
  def preprocess(self, img):
    processed = list()
    scales = [x * 368 / img.shape[0] for x in [0.5, 1, 1.5, 2]]
    for scale in scales:
      resized = cv2.resize(img, dsize = None, fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
      h, w, _ = resized.shape
      # pad to make the image shape divisible by stride(8)
      pad = [[0, 0 if h % 8 == 0 else 8 - (h % 8)], [0, 0 if w % 8 == 0 else 8 - (w % 8)], [0, 0]]
      padded = np.pad(resized, pad, mode = 'constant', constant_values = 128)
      data = np.expand_dims(padded, axis = 0).astype(np.float32) # data.shape = (b,c,h,w)
      processed.append((data, pad))
    return processed
  def predict(self, img):
    processed = self.preprocess(img)
    # 1) get heatmaps and pafs of different scales
    heatmaps = list()
    pafs = list()
    for inputs, pad in processed:
      paf, heatmap = self.model.signatures['serving_default'](inputs);
      heatmap = np.squeeze(heatmap, axis = 0)
      heatmap = cv2.resize(heatmap, dsize = None, fx = 8, fy = 8, interpolation = cv2.INTER_CUBIC)
      heatmap = heatmap[pad[0][0]:heatmap.shape[0] - pad[0][1],pad[1][0]:heatmap.shape[1] - pad[1][1],:]
      heatmap = cv2.resize(heatmap, dsize = (img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
      heatmaps.append(heatmap)
      paf = np.squeeze(paf, axis = 0)
      paf = cv2.resize(paf, dsize = None, fx = 8, fy = 8, interpolation = cv2.INTER_CUBIC)
      paf = paf[pad[0][0]:paf.shape[0] - pad[0][1],pad[1][0]:paf.shape[1] - pad[1][1],:]
      paf = cv2.resize(paf, dsize = (img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
      pafs.append(paf)
    heatmap_avg = np.mean(np.stack(heatmaps), axis = 0)
    paf_avg = np.mean(np.stack(pafs), axis = 0)
    # 2) find heatmap peak coordinates and corresponding scores of 18 keypoints
    all_peaks = list()
    for part in range(18):
      heatmap = heatmap_avg[...,part]
      heatmap = cv2.GaussianBlur(heatmap, ksize = None, sigmaX = 3, sigmaY = 3)
      up = np.roll(heatmap, shift = 1, axis = 0)
      down = np.roll(heatmap, shift = -1, axis = 0)
      left = np.roll(heatmap, shift = 1, axis = 1)
      right = np.roll(heatmap, shift = -1, axis = 1)
      peaks = np.logical_and.reduce([heatmap >= up, heatmap >= down, heatmap >= left, heatmap >= right, heatmap > 0.1], axis = 0)
      y,x = np.where(peaks);
      all_peaks.append([(n,m,heatmap[n,m],i) for i,(n,m) in enumerate(zip(y,x))]) # (y,x,score,index in this keypoint set)
    # 3) find candidate limbs of every kind, and score them
    connection_all = list()
    for k in range(len(self.mapIdx)):
      score_mid = paf_avg[...,[x - 19 for x in self.mapIdx[k]]]
      # get the 2 candidate keypoint sets that can serve as the 2 ends of this limb
      candA = all_peaks[self.limbSeq[k][0] - 1] # candidate keypointA set
      candB = all_peaks[self.limbSeq[k][1] - 1] # candidate keypointB set
      if len(candA) and len(candB):
        connection_candidate = list()
        for i, A in enumerate(candA):
          for j, B in enumerate(candB):
            Apos = A[:2]; Bpos = B[:2]
            dist = np.linalg.norm(np.array(Bpos) - np.array(Apos),2)
            # skip cases in which two keypoints overlap
            if dist == 0: continue
            # sample paf over the line between these two keypoints
            samples_y = np.linspace(A[0], B[0], num = 10)
            samples_x = np.linspace(A[1], B[1], num = 10)
            vec_x = score_mid[np.round(samples_y).astype(np.int32), np.round(samples_x).astype(np.int32), 0] # vec_x.shape = (10,)
            vec_y = score_mid[np.round(samples_y).astype(np.int32), np.round(samples_x).astype(np.int32), 1] # vec_y.shape = (10,)
            weights = (np.array(Bpos) - np.array(Apos)) / np.maximum(dist, 1e-6)
            score_midpts = vec_x * weights[1] + vec_y * weights[0] # score_midpts.shape = (10,)
            score_with_dist_prior = np.sum(score_midpts) / score_midpts.shape[0] + np.minimum(0.5 * img.shape[0] / dist - 1, 0)
            criterion1 = np.sum((score_midpts > 0.05).astype(np.int32)) > 0.8 * score_midpts.shape[0]
            criterion2 = score_with_dist_prior > 0
            if criterion1 and criterion2:
              # save candidate limb which meets 2 criterions
              connection_candidate.append((i, j, score_with_dist_prior, score_with_dist_prior + A[2] + B[2]))
        # nms
        connection_candidate = sorted(connection_candidate, key = lambda x: x[2], reverse = True)
        connection, I, J = list(), set(), set()
        for candidate in connection_candidate:
          i,j,s = candidate[:3]
          if i not in I and j not in J:
            # any keypoint cannot be shared between two limb of a same kind
            connection.append([candA[i][3], candB[j][3], s]) # (index of keypointA in the heatmap it belongs, index of keypointB in the heat map it belongs, score)
            I.add(i); J.add(j)
            if len(connection) >= min(len(candA), len(candB)): break
        connection = np.stack(connection) if len(connection) else np.zeros((0, 3)) # connection.shape = (limb_candidate_num, 3)
        connection_all.append(connection)
      else:
        # cannot find any candidate of this limb, because one of the end keypoint is missing
        connection_all.append(None)
    # 4) generate skeleton from limbs
    skeletons = list() # skeleton[heatmap idx]->keypoint idx
    for k in range(len(self.mapIdx)):
      # for limb of every kind
      if connection_all[k] is not None:
        # this limb has candidates
        indexA, indexB = np.array(self.limbSeq[k]) - 1 # indices of heatmaps which keypointA and keypointB belongs
        for cand_limb in connection_all[k]:
          # cand_limb = (index of keypointA in its heatmap, index of keypointB in its heatmap, score)
          share_end_limbs = [(idx, skeleton) for idx, skeleton in enumerate(skeletons) if (skeleton[indexA] == int(cand_limb[0])) or \
                                                                                          (skeleton[indexB] == int(cand_limb[1]))]
          if len(share_end_limbs) == 0 and k < 17:
            # if this limb has not been used by other skeletons
            # create a new skeleton and put the limb to this skeleton
            skeleton = np.array([None for i in range(20)])
            skeleton[indexA] = int(cand_limb[0])
            skeleton[indexB] = int(cand_limb[1])
            skeleton[-1] = 2 # counter of how many keypoints of this skeleton have been found
            skeleton[-2] = all_peaks[indexA][int(cand_limb[0])][2] + all_peaks[indexB][int(cand_limb[1])][2] + cand_limb[2] # score of the skeleton (total score of keypoints and limbs)
            skeletons.append(skeleton)
          elif len(share_end_limbs) == 1:
            # if one skeleton has a existing limb shares keypoint with this limb
            # add current limb to this skeleton
            idx, matched = share_end_limbs[0]
            if matched[indexB] != int(cand_limb[1]):
              matched[indexB] = int(cand_limb[1])
              matched[-1] += 1
              matched[-2] += all_peaks[indexB][int(cand_limb[1])][2] + cand_limb[2]
          elif len(share_end_limbs) == 2:
            # if two skeletons each sharing a distinct keypoint with this limb
            idx1, matched1 = share_end_limbs[0]; kpset1 = set(np.where(matched1[:-2] != None)[0])
            idx2, matched2 = share_end_limbs[1]; kpset2 = set(np.where(matched2[:-2] != None)[0])
            # if the two skeletons own disjoint keypoints, merge the two skeletons together
            if kpset1.isdisjoint(kpset2):
              matched1[np.where(matched2[:-2] != None)[0]] = matched2[np.where(matched2[:-2] != None)[0]]
              matched1[-2:] += matched2[-2:]
              matched1[-2] += cand_limb[2]
              skeletons.pop(idx2)
            # if the two skeletons own common keypoints, add the limb to both of them
            else:
              matched1[indexB] = int(cand_limb[1])
              matched1[-1] += 1
              matched1[-2] += all_peaks[indexB][int(cand_limb[1])][2] + cand_limb[2]
    # 5) delete skeletons with only a few limbs
    skeletons = [skeleton for skeleton in skeletons if skeleton[-1] >= 4 and skeleton[-2] / skeleton[-1] >= 0.4]
    results = list()
    for skeleton in skeletons:
      result = list()
      for heatmap_idx, keypoint_idx in enumerate(skeleton[:-2]):
        if keypoint_idx is None: result.append(None)
        else: result.append(all_peaks[heatmap_idx][keypoint_idx][:2][::-1])
      results.append((result, skeleton[-2]))
    return results
  def visualize(self, img, results):
    for skeleton, score in results:
      # visualize joint
      for heatmap_idx, joint in enumerate(skeleton):
        if joint is None: continue
        cv2.circle(img, (joint[0], joint[1]), 4, self.colors[heatmap_idx], thickness = -1)
      # visualize limb
      for limb_idx, limb in enumerate(self.limbSeq):
        if limb_idx >= 17: break
        if skeleton[limb[0] - 1] is None or skeleton[limb[1] - 1] is None: continue
        joint1 = skeleton[limb[0] - 1]
        joint2 = skeleton[limb[1] - 1]
        meanX = np.mean([joint1[0], joint2[0]])
        meanY = np.mean([joint1[1], joint2[1]])
        length = ((joint1[0] - joint2[0]) ** 2 + (joint1[1] - joint2[1]) ** 2) ** .5
        angle = np.degrees(np.arctan2(joint1[1] - joint2[1], joint1[0] - joint2[0]))
        polygon = cv2.ellipse2Poly((int(meanX), int(meanY)), (int(length / 2), 4), int(angle), 0, 360, 1)
        cur_img = img.copy()
        cv2.fillConvexPoly(cur_img, polygon, self.colors[limb_idx])
        img = cv2.addWeighted(img, .4, cur_img, .6, 0)
    return img

if __name__ == "__main__":
  
  openpose = OpenPose('savedmodel');
  img = cv2.imread('test/square_dance.jpg');
  results = openpose.predict(img);
  img = openpose.visualize(img, results);
  cv2.imshow('results',img);
  cv2.waitKey();
