import os
from tqdm import tqdm
import sparse
import numpy as np
import pickle

from joblib import Parallel, delayed
from aermanager.parsers import parse_header_from_file, get_aer_events_from_file


# Source data folder  (subject1/, subject2/, ... each containing *.aedat files)
path_dataset = '../datasets/ASL_DVS/'
# Target data folder
path_dataset_dst = '../datasets/ASL_DVS/clean_dataset_frames_2000/'

chunk_len_ms = 2
chunk_len_us = chunk_len_ms * 1000
width = 240
height = 180


def parse_davis240c(filename):
    """
    Parse AEDAT 2.0 events from a DAVIS240C sensor (240x180).
    Bit layout (big-endian address):
        x   = bits [19:12]  (0-239)
        y   = bits [29:22]  (0-179)
        pol = bit  [11]
    """
    data_version, data_start = parse_header_from_file(filename)
    all_events = get_aer_events_from_file(filename, data_version, data_start)
    all_addr = all_events['address']
    t = all_events['timeStamp']

    x = (all_addr >> 12) & 0xFF
    y = (all_addr >> 22) & 0xFF
    p = (all_addr >> 11) & 0x1

    # Filter out APS/IMU events that decode to out-of-range coordinates
    valid = (x <= 239) & (y <= 179)
    return np.column_stack([x[valid], y[valid], t[valid], p[valid]]).astype('int32')


# Collect all samples as (subject_dir, filename) tuples
subjects = sorted([d for d in os.listdir(path_dataset)
                   if os.path.isdir(os.path.join(path_dataset, d))])
total_samples = [(d, f)
                 for d in subjects
                 for f in os.listdir(os.path.join(path_dataset, d))
                 if f.endswith('.aedat')]

# Subject-based split: last subject → test, rest → train
# This is more meaningful for generalization across subjects
test_subject = subjects[-1]   # e.g. 'subject5'
train_samples = [(d, f) for d, f in total_samples if d != test_subject]
test_samples  = [(d, f) for d, f in total_samples if d == test_subject]


def process_file_sample(path_dataset, subject, filename, train_samples):
    mode = 'train' if (subject, filename) in set(train_samples) else 'test'
    stem = '{}_{}'.format(subject, filename[:-6])   # e.g. 'subject1_a'
    filename_dst = os.path.join(path_dataset_dst, mode, '{}.pckl'.format(stem))
    if os.path.isfile(filename_dst):
        return

    os.makedirs(os.path.dirname(filename_dst), exist_ok=True)

    filepath = os.path.join(path_dataset, subject, filename)
    total_events = parse_davis240c(filepath)

    if total_events.shape[0] == 0:
        print('Empty file:', filepath)
        return

    total_chunks = []
    while total_events.shape[0] > 0:
        end_t = total_events[-1][2]
        chunk_inds = np.where(total_events[:, 2] >= end_t - chunk_len_us)[0]
        if len(chunk_inds) <= 4:
            pass
        else:
            total_chunks.append(total_events[chunk_inds])
        total_events = total_events[:max(1, chunk_inds.min()) - 1]
    if len(total_chunks) == 0:
        print('No chunks:', filepath)
        return
    total_chunks = total_chunks[::-1]

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

    pickle.dump(total_frames, open(filename_dst, 'wb'))


Parallel(n_jobs=8)(
    delayed(process_file_sample)(path_dataset, subject, filename, train_samples)
    for subject, filename in tqdm(total_samples)
)


# %%  Rename files to append class index suffix

class_mapping = {l: i for i, l in enumerate('a b c d e f g h i k l m n o p q r s t u v w x y'.split())}

for split in ('train', 'test'):
    split_dir = os.path.join(path_dataset_dst, split)
    if not os.path.isdir(split_dir):
        continue
    for f in os.listdir(split_dir):
        if not f.endswith('.pckl'):
            continue
        label_char = f.split('_')[-1][0]   # last segment's first char, e.g. 'a' from 'subject1_a.pckl'
        if label_char not in class_mapping:
            continue
        new_name = f.replace('.pckl', '_{}.pckl'.format(class_mapping[label_char]))
        os.rename(
            os.path.join(split_dir, f),
            os.path.join(split_dir, new_name)
        )
