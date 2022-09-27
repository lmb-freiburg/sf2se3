import os
import glob

dataset_path_source = '' # path to source
dataset_path_target = '' # path to target
dataset_parts_subdirs = ['TEST', 'TRAIN']
dataset_data_subdirs = ['frames_finalpass', 'frames_cleanpass', 'optical_flow', 'disparity', 'object_index', 'camera_data']
dataset_data_dep_subdirs = {'frames_finalpass' : ['left', 'right'],
                            'frames_cleanpass' : ['left', 'right'],
                            'optical_flow' : ['into_future/left', 'into_future/right', 'into_past/left', 'into_past/right'],
                            'disparity' : ['left', 'right'],
                            'object_index' : ['left', 'right'],
                            'camera_data' : ['']}
dataset_data_1file_per_seq = {'frames_finalpass' : 'False',
                              'frames_cleanpass' : 'False',
                              'optical_flow' : 'False',
                              'disparity' : 'False',
                              'object_index' : 'False',
                              'camera_data' : 'camera_data.txt'}

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

import random
random.seed(0)
flownet3d_train_seqs_dirs_abs = glob.glob(os.path.join(dataset_path_source, 'disparity/TRAIN/*/*/*/'))
flownet3d_test_seqs_dirs_abs = glob.glob(os.path.join(dataset_path_source, 'disparity/TEST/*/*/*/'))
flownet3d_seqs_dirs_abs = random.sample(flownet3d_train_seqs_dirs_abs, 2223) + random.sample(flownet3d_test_seqs_dirs_abs, 223)
flownet3d_seqs_dirs_rel = [os.path.join(*seq_dir_abs.split('/')[-5:-2]) for seq_dir_abs in flownet3d_seqs_dirs_abs]
for dataset_data_subdir in dataset_data_subdirs:
    print('data', dataset_data_subdir)
    for dataset_part_subdir in dataset_parts_subdirs:
        print('part', dataset_part_subdir)
        seqs_dirs_abs = glob.glob(dataset_path_source+'/'+dataset_data_subdir+'/'+dataset_part_subdir+'/[ABC]/[0-9][0-9][0-9][0-9]')
        print('found seqs dirs abs', len(seqs_dirs_abs))
        seqs_dirs_rel = [os.path.join(*seq_dir_abs.split('/')[-2:]) for seq_dir_abs in seqs_dirs_abs]
        for seq_dir_rel in seqs_dirs_rel:
            if os.path.join(dataset_part_subdir, seq_dir_rel) not in flownet3d_seqs_dirs_rel:
                continue
            for dataset_data_dep_subdir in dataset_data_dep_subdirs[dataset_data_subdir]:
                data_ext = dataset_data_dep_subdir.replace('/', '_')
                if len(data_ext) > 0:
                    data_ext = '_' + data_ext
                if len(dataset_data_dep_subdir) > 0:
                    source_dir = os.path.join(dataset_path_source, dataset_data_subdir, dataset_part_subdir, seq_dir_rel, dataset_data_dep_subdir)
                else:
                    source_dir = os.path.join(dataset_path_source, dataset_data_subdir, dataset_part_subdir, seq_dir_rel)
                if dataset_data_1file_per_seq[dataset_data_subdir] != 'False':
                    source_path = os.path.join(source_dir, dataset_data_1file_per_seq[dataset_data_subdir])
                    target_dir = os.path.join(dataset_path_target, dataset_part_subdir, seq_dir_rel)
                    target_path = os.path.join(target_dir, dataset_data_1file_per_seq[dataset_data_subdir])
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    os.symlink(source_path, target_path)
                else:
                    target_dir = os.path.join(dataset_path_target, dataset_part_subdir, seq_dir_rel,
                                              dataset_data_subdir + data_ext)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    for file in os.listdir(source_dir):
                        source_path = os.path.join(source_dir, file)
                        target_path = os.path.join(target_dir, file)
                        os.symlink(source_path, target_path)

#       1.1 filter sequences for flownet3d
#       2. create sequences directory

#       3. create data_key_dirs
#       4. link files




#walk_info = [x for x in os.walk(path_ft3d_orig)]
#subdirs = [x[0] for x in walk_info]
#subdirs_fnames = [sorted(x[2]) for x in walk_info]