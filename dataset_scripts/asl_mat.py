import os
import pickle

import numpy as np
import scipy.io
import sparse
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Source data folder: <label>/<label>_NNNN.mat  (official ASL-DVS ICCV2019 dataset,
# downloaded via the OpenI mirror since the original Dropbox link is dead)
path_dataset = '../datasets/ICCV2019_DVS_dataset/'
# Target data folder
path_dataset_dst = '../datasets/ICCV2019_DVS_dataset/clean_dataset_frames_2000/'

chunk_len_ms = 2
chunk_len_us = chunk_len_ms * 1000
width = 240
height = 180

class_mapping = {l: i for i, l in enumerate('a b c d e f g h i k l m n o p q r s t u v w x y'.split())}

# Only look at the 24 official class folders (skips clean_dataset_frames_2000/ itself
# and any other stray files/dirs under path_dataset).
labels = sorted([d for d in os.listdir(path_dataset)
                  if d in class_mapping and os.path.isdir(os.path.join(path_dataset, d))])
total_samples = [s for d in labels for s in os.listdir(os.path.join(path_dataset, d))
                  if s.endswith('.mat')]
total_labels = [s.split('_')[0] for s in total_samples]

train_samples, test_samples = train_test_split(
    total_samples, test_size=0.2, random_state=0, stratify=total_labels)
train_samples_set = set(train_samples)


def process_file_sample(path_dataset, label, f, train_samples_set):
    mode = 'train' if f in train_samples_set else 'test'
    stem = f[:-4]   # e.g. 'a_0001'
    filename_dst = os.path.join(path_dataset_dst, mode, '{}.pckl'.format(stem))

    # Resume support: a file only counts as "done" if it actually has content.
    if os.path.isfile(filename_dst) and os.path.getsize(filename_dst) > 0:
        return

    os.makedirs(os.path.dirname(filename_dst), exist_ok=True)

    mat = scipy.io.loadmat(os.path.join(path_dataset, label, f))
    total_events = np.array([mat['x'], mat['y'], mat['ts'], mat['pol']]).transpose()[0].astype('int64')

    if total_events.shape[0] == 0:
        print('Empty file:', f)
        return

    # Vectorized chunking (see EventTransformer/dataset_scripts/asl.py for details):
    # groups events into <=chunk_len_us trailing windows, walking backward from the
    # most recent event, using a single np.searchsorted call instead of an O(N^2)
    # np.where + re-slice loop.
    t = total_events[:, 2]
    n = t.shape[0]
    left = np.searchsorted(t, t - chunk_len_us, side='left').tolist()

    chunk_bounds = []
    end_idx = n
    while end_idx > 0:
        start_idx = left[end_idx - 1]
        if end_idx - start_idx > 4:
            chunk_bounds.append((start_idx, end_idx))
        end_idx = start_idx - 1 if start_idx > 1 else 0
    if len(chunk_bounds) == 0:
        print('No chunks:', f)
        return
    chunk_bounds.reverse()
    total_chunks = [total_events[s:e] for s, e in chunk_bounds]

    total_frames = []
    for chunk in total_chunks:
        frame = sparse.COO(
            chunk[:, [1, 0, 3]].transpose().astype('int32'),
            np.ones(chunk.shape[0]).astype('int32'),
            (height, width, 2)
        )
        total_frames.append(frame)
    total_frames = sparse.stack(total_frames)

    total_frames = np.clip(total_frames, a_min=0, a_max=255)
    total_frames = total_frames.astype('uint8')

    # Write atomically: dump to a temp file first, then rename into place.
    tmp_filename_dst = filename_dst + '.tmp'
    with open(tmp_filename_dst, 'wb') as fh:
        pickle.dump(total_frames, fh)
    os.replace(tmp_filename_dst, filename_dst)


if __name__ == '__main__':
    Parallel(n_jobs=8)(
        delayed(process_file_sample)(path_dataset, f.split('_')[0], f, train_samples_set)
        for f in tqdm(total_samples)
    )

    # Append class index suffix to filenames, e.g. 'a_0001.pckl' -> 'a_0001_0.pckl'
    for split in ('train', 'test'):
        split_dir = os.path.join(path_dataset_dst, split)
        if not os.path.isdir(split_dir):
            continue
        for f in os.listdir(split_dir):
            if not f.endswith('.pckl') or f.endswith('.tmp'):
                continue
            label_char = f.split('_')[0]
            if label_char not in class_mapping:
                continue
            new_name = f.replace('.pckl', '_{}.pckl'.format(class_mapping[label_char]))
            os.rename(os.path.join(split_dir, f), os.path.join(split_dir, new_name))

    print('Done.')
