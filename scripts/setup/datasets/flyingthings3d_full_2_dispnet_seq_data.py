import os
import glob
import re

dataset_path_source = '' # path to source
dataset_path_target = '' # path to target

dataset_parts_subdirs = ['TEST', 'TRAIN']
#dataset_data_subdirs = ['frames_finalpass', 'frames_cleanpass', 'optical_flow', 'disparity', 'disparity_change', 'object_index', 'camera_data']
dataset_data_subdirs = ['camera_data']
dataset_data_dep_subdirs = {'frames_finalpass' : ['left', 'right'],
                            'frames_cleanpass' : ['left', 'right'],
                            'optical_flow' : ['into_future/left', 'into_future/right', 'into_past/left', 'into_past/right'],
                            'disparity' : ['left', 'right'],
                            'disparity_change' : ['into_future/left', 'into_future/right', 'into_past/left', 'into_past/right'],
                            'object_index' : ['left', 'right'],
                            'camera_data' : ['']}
dataset_data_1file_per_seq = {'frames_finalpass' : 'False',
                              'frames_cleanpass' : 'False',
                              'optical_flow' : 'False',
                              'disparity' : 'False',
                              'disparity_change' : 'False',
                              'object_index' : 'False',
                              'camera_data' : 'camera_data.txt'}

blacklist_path = 'datasets/ft3d_meta/dispnet_all_unused_files.txt'
fpath = blacklist_path
a_file = open(fpath, "r")
blacklist = a_file.read().splitlines()
blacklist_seqs = [os.path.join(*fpath.split("/")[:3]) for fpath in blacklist]
blacklist_fnames = [fpath.split("/")[-1].split('.')[0] for fpath in blacklist]

blacklist_seqs_fnames = {}
for i, fname in enumerate(blacklist_fnames):
    key = blacklist_seqs[i]
    if  key not in blacklist_seqs_fnames.keys():
        blacklist_seqs_fnames[key] = []
    blacklist_seqs_fnames[key].append(fname)

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

for dataset_data_subdir in dataset_data_subdirs:
    print('data', dataset_data_subdir) # frames_finalpass, etc.
    for dataset_part_subdir in dataset_parts_subdirs: # TRAIN, TEST
        print('part', dataset_part_subdir)
        seqs_dirs_abs = glob.glob(dataset_path_source+'/'+dataset_data_subdir+'/'+dataset_part_subdir+'/[ABC]/[0-9][0-9][0-9][0-9]')
        print('found seqs dirs abs', len(seqs_dirs_abs))
        seqs_dirs_rel = [os.path.join(*seq_dir_abs.split('/')[-2:]) for seq_dir_abs in seqs_dirs_abs]
        # [ABC]/[0-9][0-9][0-9][0-9]
        for seq_dir_rel in seqs_dirs_rel:
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
                    # only if the other files created target dir copy the single files per sequence
                    if os.path.exists(target_dir):
                        os.symlink(source_path, target_path)
                else:

                    target_dir = os.path.join(dataset_path_target, dataset_part_subdir, seq_dir_rel,
                                              dataset_data_subdir + data_ext)

                    source_fnames = []
                    for fname in os.listdir(source_dir):
                        if os.path.join(dataset_part_subdir, seq_dir_rel) not in blacklist_seqs_fnames.keys():
                            #print("not in blacklist:", os.path.join(dataset_part_subdir, seq_dir_rel))
                            source_fnames.append(fname)
                        else:
                            blacklist_seq_fnames = blacklist_seqs_fnames[os.path.join(dataset_part_subdir, seq_dir_rel)]
                            #print("blacklist", blacklist_seq_fnames)
                            if re.findall('[0-9]+', fname)[0] not in blacklist_seq_fnames:
                                source_fnames.append(fname)

                    #print("source_fnames", source_fnames)
                    if len(source_fnames) > 0:
                        if not os.path.exists(target_dir):
                            os.makedirs(target_dir)
                        for file in source_fnames:
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