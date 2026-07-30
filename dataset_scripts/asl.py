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
    # NOTE: timestamps are read as uint32 and can range up to ~4.29e9 (us),
    # i.e. recordings longer than ~71.5 minutes. Casting the whole stacked
    # array to int32 (as before) silently overflows into negative values
    # for such timestamps, which breaks the "events are time-sorted"
    # assumption the chunking logic below relies on and can collapse an
    # entire long recording into a handful of garbage frames. Use int64
    # for the timestamp column to avoid this.
    return np.column_stack([x[valid], y[valid], t[valid], p[valid]]).astype('int64')


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
    # Resume support: a file only counts as "done" if it actually has content.
    # (A previous interrupted run can leave 0-byte files behind.)
    if os.path.isfile(filename_dst) and os.path.getsize(filename_dst) > 0:
        return

    os.makedirs(os.path.dirname(filename_dst), exist_ok=True)

    filepath = os.path.join(path_dataset, subject, filename)
    total_events = parse_davis240c(filepath)

    if total_events.shape[0] == 0:
        print('Empty file:', filepath)
        return

    # Sanity check: the chunking below assumes events are time-sorted
    # (required for np.searchsorted to behave, and to guarantee the
    # backward-walking loop terminates). Raw AEDAT timestamps are a 32-bit
    # microsecond counter that wraps around after ~71.58 minutes; a
    # recording left running past that point (or otherwise corrupted) can
    # have a large fraction of out-of-order events. Detect that up front
    # and skip the file instead of silently producing garbage frames or
    # hanging in an infinite loop.
    t_check = total_events[:, 2]
    out_of_order = np.count_nonzero(np.diff(t_check) < 0)
    if out_of_order > 0:
        frac = out_of_order / max(1, len(t_check) - 1)
        print('Skipping (non-monotonic/corrupted timestamps, likely 32-bit '
              'overflow from an overlong recording): {} '
              '({} / {} events out of order, {:.1%})'.format(
                  filepath, out_of_order, len(t_check), frac))
        return

    # Group events into <=chunk_len_us windows, walking backward from the
    # most recent event (matches the original semantics: each window is the
    # trailing chunk_len_us slice, followed by a 1-event gap before the next
    # window starts).
    #
    # This is done with a single vectorized `np.searchsorted` call instead of
    # calling `np.where` (O(remaining size)) and re-slicing the array
    # (another O(remaining size) copy) on every loop iteration. For the
    # small samples this dataset was originally designed for, the
    # naive version was fine; but here individual .aedat files can contain
    # tens or hundreds of millions of events, which made the old O(N^2)-ish
    # loop effectively never finish (hours, sometimes much more).
    t = total_events[:, 2]
    n = t.shape[0]
    # left[i] = leftmost index j such that t[j] >= t[i] - chunk_len_us
    left = np.searchsorted(t, t - chunk_len_us, side='left').tolist()

    chunk_bounds = []
    end_idx = n
    while end_idx > 0:
        start_idx = left[end_idx - 1]
        if end_idx - start_idx > 4:
            chunk_bounds.append((start_idx, end_idx))
        end_idx = start_idx - 1 if start_idx > 1 else 0
    if len(chunk_bounds) == 0:
        print('No chunks:', filepath)
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
    # This guarantees that if the process gets killed mid-write, the
    # destination file either doesn't exist or is fully written -- never
    # a truncated/0-byte file that a resumed run would mistake for "done".
    tmp_filename_dst = filename_dst + '.tmp'
    with open(tmp_filename_dst, 'wb') as f:
        pickle.dump(total_frames, f)
    os.replace(tmp_filename_dst, filename_dst)


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
