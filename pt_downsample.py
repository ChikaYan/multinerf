import sklearn.neighbors as skln
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()

parser.add_argument('--pt_path', default='')
parser.add_argument('--downsample_density', type=float, default=0.001)
args = parser.parse_args()

def downsample(pts):
  nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=args.downsample_density, algorithm='kd_tree', n_jobs=-1)
  nn_engine.fit(pts)
  rnn_idxs = nn_engine.radius_neighbors(pts, radius=args.downsample_density, return_distance=False)
  mask = np.ones(pts.shape[0], dtype=np.bool_)
  for curr, idxs in enumerate(rnn_idxs):
      if mask[curr]:
          mask[idxs] = 0
          mask[curr] = 1
  pts = pts[mask]
  return pts


pts = np.load(args.pt_path)
pts = downsample(pts)
print(f'Down sampled points {args.pt_path}')
os.remove(args.pt_path)
pts = np.save(args.pt_path, pts)

  