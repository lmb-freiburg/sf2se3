# dataset structure flyingthings3d subset dispnet:
# SUBSET/MODALITY[/left] [/right] [/into_future][/into_past]/XXXXXXX.ZZZ
#
# VAL/MODALITY/XXXXXXX:
#   disparity:      0000000 - 0004247 (total 4248)
#   optical flow (into_future):   0000000 - 0004246 (total 3824)
#   optical flow (into_future):   0000001 - 0004247 (total 3824)
# TRAIN/MODALITY/XXXXXXX:
#   disparity:      0000000 - 0021817 (total 21818)

import os
import glob
import shutil
import pickle
dataset_path_source = '' # path to source
dataset_path_target = '' # path to target

dataset_parts_subdirs = ['val', 'train']
dataset_parts_subdirs_map_src_to_target = {'val' : 'TEST', 'train' : 'TRAIN'}
#dataset_data_subdirs = ['image_clean', 'flow', 'flow_occlusions', 'disparity', 'disparity_change', 'disparity_occlusions', 'object_ids', 'depth_boundaries']
dataset_data_subdirs = ['depth_boundaries']

dataset_data_subdirs_map_src_to_target = {
    'image_clean' : 'rgb_clean',
    'flow' : 'optical_flow',
    'flow_occlusions' : 'optical_flow_occlusions',
    'disparity' : 'disparity',
    'disparity_change' : 'disparity_change',
    'disparity_occlusions' : 'disparity_occlusions',
    'object_ids' : 'object_ids',
    'depth_boundaries' : 'depth_boundaries'
}
dataset_data_dep_subdirs = {'image_clean' : ['left', 'right'],
                            'flow' : ['left/into_future', 'right/into_future', 'left/into_past', 'right/into_past'],
                            'flow_occlusions' : ['left/into_future', 'right/into_future'],#, 'left/into_past', 'right/into_past'],
                            'disparity' : ['left', 'right'],
                            'disparity_change' : ['left/into_future', 'right/into_future'],#, 'left/into_past', 'right/into_past'],
                            'disparity_occlusions' : ['left', 'right'],
                            'object_ids' : ['left', 'right'],
                            'depth_boundaries' : ['left', 'right']}
dataset_data_1file_per_seq = {'image_clean' : 'False',
                              'flow' : 'False',
                              'flow_occlusions' : 'False',
                              'disparity' : 'False',
                              'disparity_change' : 'False',
                              'disparity_occlusions' : 'False',
                              'object_ids' : 'False',
                              'depth_boundaries' : 'False'}

fpath_val_seqs_lens = 'datasets/ft3d_meta/dispnet_val_seq_lengths.txt'
fpath_train_seqs_lens = 'datasets/ft3d_meta/dispnet_train_seq_lengths.txt'

print("WD", os.getcwd())
import sys
sys.path.append(os.getcwd())
from m4_io import m4 as m4io
test_seqs_lens = m4io.readlines_txt(fpath_val_seqs_lens)
test_seqs_lens = [int(seq_len) for seq_len in test_seqs_lens]
test_els = sum(test_seqs_lens)
# number sequences: 425 :  els: 4248 flow-els: 4248-425=3823
# problem: flow: 3824
train_seqs_lens = m4io.readlines_txt(fpath_train_seqs_lens)
train_seqs_lens = [int(seq_len) for seq_len in train_seqs_lens]
train_els = sum(train_seqs_lens)
train_seqs_lens = [int(seq_len) for seq_len in train_seqs_lens]
# number sequences: 2183 :  els: 21818 flow-els: 21818-2183=19635
# problem: flow: into-future: 19640 / into-past: 19642
#target_test_seqs_dirs_abs = sorted(glob.glob(dataset_path_target+'/'+'TEST'+'/'+'/[ABC]/[0-9][0-9][0-9][0-9]'))
#target_train_seqs_dirs_abs = sorted(glob.glob(dataset_path_target+'/'+'TRAIN'+'/'+'/[ABC]/[0-9][0-9][0-9][0-9]'))
#target_test_seqs_el_ids = [[fname.split('.')[0] for fname in sorted(os.listdir(os.path.join(seq_dir, 'frames_cleanpass_left')))] for seq_dir in target_test_seqs_dirs_abs]
#target_train_seqs_el_ids = [[fname.split('.')[0] for fname in sorted(os.listdir(os.path.join(seq_dir, 'frames_cleanpass_left')))] for seq_dir in target_train_seqs_dirs_abs]
#test_seqs_lens_v2 = [len(seq_el_ids) for seq_el_ids in target_test_seqs_el_ids]
#train_seqs_lens_v2 = [len(seq_el_ids) for seq_el_ids in target_train_seqs_el_ids]
file = open('datasets/ft3d_meta/seq_index_val.pickle', 'rb')
test_map_el_to_seq_el = pickle.load(file)
# id -> tuple: ('A', 0, 6)
file = open('datasets/ft3d_meta/seq_index_train.pickle', 'rb')
train_map_el_to_seq_el = pickle.load(file)

# 'frames_finalpass' : left / right
# 'frames_cleanpass' : left /right
# 'optical_flow' : left / right
# 'disparity' : into_future / into_past , left / right
# 'object_index' : left / right
# A,B,C / XXXX

# data_dirs = ['frames_finalpass', 'frames_cleanpass', 'optical_flow', 'disparity',
# sequence regex: /[A,B,C]/[0-9]

# for each dataset part
#     for each data type
#       1. find sequences

for dataset_part_subdir in dataset_parts_subdirs:
    print('part', dataset_part_subdir)
    for dataset_data_subdir in dataset_data_subdirs:
        print('data', dataset_data_subdir)
        src_part_data_dir = os.path.join(dataset_path_source, dataset_part_subdir, dataset_data_subdir)
        if dataset_part_subdir == 'val':
            map_el_to_seq_el_id = test_map_el_to_seq_el
            seq_lens = test_seqs_lens
        elif dataset_part_subdir == 'train':
            map_el_to_seq_el_id = train_map_el_to_seq_el
            seq_lens = train_seqs_lens
        else:
            print('error: unknown dataset_part_subdir', dataset_part_subdir)

        if os.path.exists(dataset_part_subdir):
            shutil.rmtree(dataset_part_subdir, ignore_errors=True)
        if not os.path.exists(dataset_part_subdir):
            os.makedirs(dataset_part_subdir)

        print("src", src_part_data_dir)
        for dataset_data_dep_subdir in dataset_data_dep_subdirs[dataset_data_subdir]:
            src_part_data_dep_dir = os.path.join(src_part_data_dir, dataset_data_dep_subdir)
            print("src", src_part_data_dep_dir, len(os.listdir(src_part_data_dep_dir)))
            src_part_data_dep_fnames = sorted(os.listdir(src_part_data_dep_dir))
            print("src: #fnames", len(src_part_data_dep_fnames))
            for i in range(len(src_part_data_dep_fnames)):
                source_fname = src_part_data_dep_fnames[i]
                src_el_id = int(source_fname.split('.')[0])

                target_seq_letter, target_seq_letter_id, target_seq_letter_id_el_id = map_el_to_seq_el_id[src_el_id]
                target_part_subdir = dataset_parts_subdirs_map_src_to_target[dataset_part_subdir]
                target_data_subdir = dataset_data_subdirs_map_src_to_target[dataset_data_subdir] + "_" + dataset_data_dep_subdir.replace('/', '_')
                target_seq_subdir = os.path.join(target_seq_letter, f'{target_seq_letter_id:04d}')
                target_seq_data_dir = os.path.join(dataset_path_target, target_part_subdir, target_seq_subdir, target_data_subdir)
                if not os.path.exists(target_seq_data_dir):
                    os.makedirs(target_seq_data_dir)

                source_path = os.path.join(src_part_data_dep_dir, source_fname)
                target_path = os.path.join(target_seq_data_dir,
                                           f'{target_seq_letter_id_el_id:04d}' + '.' +
                                           source_fname.split('.')[-1])
                print("src", source_path)
                print("target", target_path)
                if os.path.exists(target_path):
                    os.unlink(target_path)
                os.symlink(source_path, target_path)