import numpy as np
import os
import glob
import re
import torch
from torch.utils.data import Dataset as PytorchDataset
import torchvision
import m4_io.m4 as m4_io

class SeqDataset(PytorchDataset):

    def __init__(self,
        data_load_root,
        dtype=None,
        dev = None,
        width=640,
        height=640,
        directory_structure="DATA-SEQ",
        directory_structure_seq_depth=1,
        seqs_dirs_filter_tags=[],
        timestamp_inter_time_min_diff = None,
        timestamp_inter_data_max_diff = 0.02, # 0.02
        data_seq_load_files=None,
        data_load_dirs={},
        data_load_offsets={},
        data_load_types={},
        data_fix={},
        index_shift=0,
        meta_use=True,
        meta_recalc=False):

        # data_load_dirs:
        #   DATA-SEQ / JOINT: absolute path
        #   SEQ-DATA: relative path of each sequence
        #    -> requires data_load_root

        self.meta_use = meta_use
        self.meta_recalc = meta_recalc
        self.directory_structure = directory_structure
        self.directory_structure_seq_depth = directory_structure_seq_depth
        self.seqs_dirs_filter_tags = seqs_dirs_filter_tags
        self.seq_dirs = None
        self.seq_tags = None
        self.data_load_root = data_load_root
        self.index_shift = index_shift
        if dtype == None:
            self.dtype = torch.float32
        else:
            self.dtype = dtype

        self.dtype_np = m4_io.dtype_torch2numpy(self.dtype)

        self.transform_to_tensor = torchvision.transforms.ToTensor()

        if dev == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = dev

        self.width = width
        self.height = height

        self.data_fix_keys = data_fix.keys()
        self.data_fix = data_fix

        self.data_load_keys = data_load_dirs.keys()


        self.data_seqs_fnames = {}
        self.data_seqs_dirs = {}
        self.data_seqs_lens = {}
        #self.data_seqs_lens_min = []
        self.data_load_offsets = data_load_offsets
        self.data_load_types = data_load_types

        self.meta_dir = os.path.join(self.data_load_root, 'meta')

        print('dataset: retrieve sequence directories and filenames...')
        if os.path.exists(self.meta_dir) and self.meta_use and not self.meta_recalc:
            print('dataset: retrieve by loading existing meta information...')
            self.data_seqs_dirs = m4_io.read_json(os.path.join(self.meta_dir, "data_seqs_dirs.json"))
            self.data_seqs_fnames = m4_io.read_json(os.path.join(self.meta_dir, "data_seqs_fnames.json"))
            self.data_seqs_lens = m4_io.read_json(os.path.join(self.meta_dir, "data_seqs_lens.json"))
            self.seq_tags = m4_io.read_json(os.path.join(self.meta_dir, "seq_tags.json"))
            self.seq_dirs = m4_io.read_json(os.path.join(self.meta_dir, "seq_dirs.json"))


        else:
            print('dataset: retrieve by browsing through file directories...')
            for data_load_key, data_load_dir in data_load_dirs.items():
                print('dataset: key', data_load_key, 'dir', data_load_dir)
                seq_dirs, seq_tags, seq_fnames = self.get_seq_dirs_and_fnames(data_load_dir, data_load_key)

                if self.seq_tags is None:
                    self.seq_tags = seq_tags

                seq_lens = [len(seq_fnames[seq_id]) for seq_id in range(len(seq_fnames))]

                self.data_seqs_fnames[data_load_key] = seq_fnames
                self.data_seqs_dirs[data_load_key] = seq_dirs
                self.data_seqs_lens[data_load_key] = seq_lens

                #if len(self.data_seqs_lens_min) == 0:
                #    self.data_seqs_lens_min = seq_lens
                #else:
                #    for seq_id in range(len(seq_lens)):
                #        if self.data_seqs_lens_min[seq_id] > seq_lens[seq_id]:
                #            self.data_seqs_lens_min[seq_id] = seq_lens[seq_id]

            if self.meta_use:
                if not os.path.exists(self.meta_dir):
                    os.makedirs(self.meta_dir)
                m4_io.save_json(os.path.join(self.meta_dir, "data_seqs_dirs.json"), self.data_seqs_dirs)
                m4_io.save_json(os.path.join(self.meta_dir, "data_seqs_fnames.json"), self.data_seqs_fnames)
                m4_io.save_json(os.path.join(self.meta_dir, "data_seqs_lens.json"), self.data_seqs_lens)
                m4_io.save_json(os.path.join(self.meta_dir, "seq_tags.json"), self.seq_tags)
                m4_io.save_json(os.path.join(self.meta_dir, "seq_dirs.json"), self.seq_dirs)


        if len(self.seqs_dirs_filter_tags) > 0:
            filtered_seq_ids = [i for i in range(len(self.seq_tags)) if self.seq_tags[i] in self.seqs_dirs_filter_tags]
            self.seq_tags = [self.seq_tags[filtered_seq_id] for filtered_seq_id in filtered_seq_ids]
            if self.seq_dirs is not None:
                self.seq_dirs = [self.seq_dirs[filtered_seq_id] for filtered_seq_id in filtered_seq_ids]
            for data_load_key, data_load_dir in data_load_dirs.items():
                self.data_seqs_dirs[data_load_key] = [self.data_seqs_dirs[data_load_key][filtered_seq_id] for
                                                      filtered_seq_id in filtered_seq_ids]
                self.data_seqs_fnames[data_load_key] = [self.data_seqs_fnames[data_load_key][filtered_seq_id] for
                                                      filtered_seq_id in filtered_seq_ids]
                self.data_seqs_lens[data_load_key] = [self.data_seqs_lens[data_load_key][filtered_seq_id] for
                                                      filtered_seq_id in filtered_seq_ids]


        print('dataset: retrieving timestamps...')
        self.data_seqs_tstamps = {}
        for data_load_key in data_load_dirs.keys():

            self.data_seqs_tstamps[data_load_key] = []
            seqs_fnames = self.data_seqs_fnames[data_load_key]
            for seq_fnames in seqs_fnames:
                seq_timestamps = [float(re.findall('[0-9]+\\.?[0-9]*', seq_fname)[0]) for seq_fname in seq_fnames]
                #seq_timestamps = [float(os.path.splitext(seq_fname)[0]) for seq_fname in seq_fnames]
                self.data_seqs_tstamps[data_load_key].append(seq_timestamps)
            #print(seq_fnames)
            #pass

        print('dataset: preloading files for each sequence...')
        self.data_preloaded = {}
        if self.seq_dirs is not None:
            for data_seq_load_key in data_seq_load_files.keys():
                self.data_preloaded[data_seq_load_key] = []
                self.data_seqs_lens[data_seq_load_key] = []
                self.data_seqs_tstamps[data_seq_load_key] = []
                for seq_dir in self.seq_dirs:
                    fpath = os.path.join(self.data_load_root, seq_dir, data_seq_load_files[data_seq_load_key])
                    data = self.read_data(fpath, ftype=self.data_load_types[data_seq_load_key], dkey=data_seq_load_key)
                    for key in data.keys():
                        if key.endswith("_tstamp"):
                            self.data_seqs_tstamps[data_seq_load_key].append(data[key].detach().cpu().numpy())
                        else:
                            self.data_preloaded[data_seq_load_key].append(data[key])
                    self.data_seqs_lens[data_seq_load_key].append(len(self.data_seqs_tstamps[data_seq_load_key][-1]))

        print('dataset: checking if seqs lens match for each data key and sequence')
        # check if number of sequences for each data key is equal
        for data_key1, seq_lens1 in self.data_seqs_lens.items():
            for data_key2, seq_lens2 in self.data_seqs_lens.items():
                if len(seq_lens1) != len(seq_lens2):
                    print('error: number of sequences differ', data_key1, data_key2)
                self.data_num_seqs = len(seq_lens1)

        print('dataset: match time stamps')
        self.data_first_key = next(iter(data_load_dirs))
        self.data_num_seqs = len(self.data_seqs_lens[self.data_first_key])
        self.data_seqs_last_tstamps_ids = {}

        for data_load_key in self.data_seqs_tstamps.keys():
            self.data_seqs_last_tstamps_ids[data_load_key] = []
            for seq_id in range(self.data_num_seqs):
                self.data_seqs_last_tstamps_ids[data_load_key].append(-1)

        self.data_seqs_el_ids = {}

        for seq_id in range(self.data_num_seqs):
            for data_load_key in self.data_seqs_tstamps.keys():
                if data_load_key not in self.data_seqs_el_ids.keys():
                    self.data_seqs_el_ids[data_load_key] = []
                self.data_seqs_el_ids[data_load_key].append([])

        for seq_id in range(self.data_num_seqs):
            for el_id in range(self.data_seqs_lens[self.data_first_key][seq_id]):
                data_nearest_id = {}
                data_nearest_id_tstamp_diff = {}
                tstamp_ref = self.data_seqs_tstamps[self.data_first_key][seq_id][el_id]

                if timestamp_inter_time_min_diff is not None:
                    if self.data_seqs_last_tstamps_ids[self.data_first_key][seq_id] >= 0:
                        last_tstamp_ref = self.data_seqs_tstamps[self.data_first_key][seq_id][self.data_seqs_last_tstamps_ids[self.data_first_key][seq_id]]
                        if (tstamp_ref - last_tstamp_ref) < timestamp_inter_time_min_diff:
                            continue
                for data_load_key in self.data_seqs_tstamps.keys():
                    last_tstamp_id = self.data_seqs_last_tstamps_ids[data_load_key][seq_id]
                    #print('tstamp ref', tstamp_ref)
                    #print(data_load_key)
                    #print(seq_id, self.data_seqs_dirs[data_load_key][seq_id])
                    remaining_tstamps_diffs = [abs(tstamp-tstamp_ref) for tstamp in self.data_seqs_tstamps[data_load_key][seq_id][last_tstamp_id+1:]]
                    if len(remaining_tstamps_diffs) > 0:
                        #print(remaining_tstamps_diffs)
                        data_nearest_id[data_load_key] = last_tstamp_id + 1 + np.argmin(remaining_tstamps_diffs)
                        data_nearest_id_tstamp_diff[data_load_key] = remaining_tstamps_diffs[data_nearest_id[data_load_key]- last_tstamp_id - 1]
                    else:
                        data_nearest_id_tstamp_diff[data_load_key] = 9999999.

                #print(data_nearest_id.keys())
                #print(data_nearest_id_tstamp_diff.values())
                if (np.array(list(data_nearest_id_tstamp_diff.values())) < timestamp_inter_data_max_diff).all():
                    for data_load_key in self.data_seqs_tstamps.keys():
                        self.data_seqs_el_ids[data_load_key][seq_id].append(data_nearest_id[data_load_key])
                        self.data_seqs_last_tstamps_ids[data_load_key][seq_id] = data_nearest_id[data_load_key]
        #for data_load_key in self.data_seqs_tstamps.keys():
        #    print(data_load_key, self.data_seqs_el_ids[data_load_key])
        #    print(data_load_key, self.data_seqs_last_tstamps_ids[data_load_key])

        self.data_seqs_offset_max = {}
        self.data_seqs_lens_min = []
        for data_load_key in self.data_seqs_el_ids.keys():
            self.data_seqs_offset_max[data_load_key] = []
            for seq_id in range(self.data_num_seqs):
                if data_load_key == self.data_first_key:
                    self.data_seqs_lens_min.append(len(self.data_seqs_el_ids[data_load_key][seq_id]))
                #self.data_load_offsets
                remaining_ids = list(range(self.data_seqs_last_tstamps_ids[data_load_key][seq_id] + 1, self.data_seqs_lens[data_load_key][seq_id]))
                self.data_seqs_el_ids[data_load_key][seq_id] += remaining_ids
                self.data_seqs_offset_max[data_load_key].append(len(remaining_ids))

                self.data_seqs_lens[data_load_key][seq_id] = len(self.data_seqs_el_ids[data_load_key][seq_id])
                if data_load_key in self.data_seqs_fnames.keys():
                    self.data_seqs_fnames[data_load_key][seq_id] = [self.data_seqs_fnames[data_load_key][seq_id][el_id] for el_id in self.data_seqs_el_ids[data_load_key][seq_id]]
                if data_load_key in self.data_preloaded.keys():
                    self.data_preloaded[data_load_key][seq_id] = [self.data_preloaded[data_load_key][seq_id][el_id]
                                                                    for el_id in
                                                                    self.data_seqs_el_ids[data_load_key][seq_id]]
                    # bug: self.data_preloaded len must change somewhere
                #self.data_seqs_fnames[]

        for data_load_key in self.data_load_offsets.keys():
            for seq_id in range(self.data_num_seqs):
                if self.data_seqs_lens[data_load_key][seq_id]-self.data_seqs_lens_min[seq_id] < np.max(self.data_load_offsets[data_load_key]):
                    self.data_seqs_lens_min[seq_id] = self.data_seqs_lens[data_load_key][seq_id] - np.max(self.data_load_offsets[data_load_key])
        # data_load_key
        # 1. for each key last-timestamp-id: -1
        # 2. search for latest 1st timestamp ( use this key)
        # 3.1 go through all timestamps for the key of 1st timestamp
        # 3.2 find neighbors among timestamps larger than last-timestamp to current timestamp
        # 3.3 if no neighbors is found for one key dismiss timestamp
        # 3.4 else set neighbors and set latest timestamp for each key
        # 4. set the sequence len to the number of found matches in timestamps
        # 5. (optional) add timestamps after latest timestmap for each key
        #    -> enables adding additional data for future offsets

        self.data_num_els = sum(self.data_seqs_lens_min)

        print('info: found ', self.__len__(), ' number of elements')
        print('info: fix keys', self.data_fix_keys)
        print('info: load keys', self.data_load_keys)
        self.data_els_ids = list(range(self.data_num_els))
        self.data_seqs_ids = list(range(self.data_num_seqs))
        self.data_els_ids_2_seqs_ids = []
        for seq_id in range(self.data_num_seqs):
            self.data_els_ids_2_seqs_ids += [self.data_seqs_ids[seq_id]] * self.data_seqs_lens_min[seq_id]

        self.data_els_ids_2_seqs_els_id = []
        for seq_id in range(self.data_num_seqs):
            self.data_els_ids_2_seqs_els_id += list(range(self.data_seqs_lens_min[seq_id]))

        #for seq_dir in self.data_seqs_dirs[self.data_first_key]:
        #    print(seq_dir)


    def __len__(self):
        return self.data_num_els

    def __getitem__(self, data_el_id):
        data_el_id += self.index_shift
        data_el_id %= self.__len__()
        return self.get_item_without_shift(data_el_id)

    def get_item_without_shift(self, data_el_id):
        #print(data_el_id)
        seq_el_id = self.data_els_ids_2_seqs_els_id[data_el_id]
        seq_id = self.data_els_ids_2_seqs_ids[data_el_id]

        data_el = {}

        for data_key in self.data_fix_keys:
            data_el[data_key] = torch.from_numpy(np.array(self.data_fix[data_key], dtype=self.dtype_np)).to(self.device)
            #data_el += [self.data_fix[data_key]]

        for data_key in self.data_load_keys:
            if data_key in self.data_load_offsets:
                offsets = self.data_load_offsets[data_key]
            else:
                offsets = [0]
            for offset in offsets:
                data_el.update(self.read_data(os.path.join(self.data_seqs_dirs[data_key][seq_id],
                                                           self.data_seqs_fnames[data_key][seq_id][seq_el_id+offset]),
                                              ftype=self.data_load_types[data_key],
                                              dkey=data_key,
                                              offset=offset))

        for data_key in self.data_preloaded.keys():
            if data_key in self.data_load_offsets:
                offsets = self.data_load_offsets[data_key]
            else:
                offsets = [0]
            for offset in offsets:
                data_preload = {}
                data_preload[data_key + '_' + str(offset)] = self.data_preloaded[data_key][seq_id][seq_el_id+offset]
                data_el.update(data_preload)
            #data_el += [self.read_data(os.path.join(self.data_seqs_dirs[data_key][seq_id], self.data_seqs_fnames[data_key][seq_id][seq_el_id]), ftype=self.data_types[data_key])]

        data_el['seq_el_id'] = seq_el_id
        data_el['seq_len'] = self.data_seqs_lens_min[seq_id]
        data_el['seq_tag'] = self.seq_tags[seq_id]

        return data_el

    def read_data(self, fpath, ftype, dkey, offset=0):
        print('read ', fpath )
        print('type ', ftype)
        print('dkey ', dkey)
        return {}

    def get_seq_dirs_and_fnames(self, dir, data_load_key):

        if self.directory_structure == 'DATA-SEQ':
            # x[0] dirs, x[1]: subdirs in dirs, x[2]: files  in dirs
            walk_info = [x for x in os.walk(dir)]
            subdirs = [x[0] for x in walk_info]
            subdirs_fnames = [sorted(x[2]) for x in walk_info]

            subdirs, subdirs_fnames = self.filter_subdirs_not_empty(subdirs, subdirs_fnames)

            num_subdirs = len(subdirs)
            seq_dirs = []
            seq_tags = []
            seq_fnames = []

            for subdir_id in range(num_subdirs):
                num_subdir_fnames = len(subdirs_fnames[subdir_id])

                subdir_fname_curr_time_id = 0
                for fname_id in range(num_subdir_fnames):
                    subdir_fname_last_time_id = subdir_fname_curr_time_id
                    subdir_fname_curr_time_id = self.get_time_id_from_fname(subdirs_fnames[subdir_id][fname_id])
                    if fname_id == 0 or (subdir_fname_last_time_id + 1) != subdir_fname_curr_time_id:
                        seq_dirs.append(subdirs[subdir_id])
                        #seq_tags.append(os.path.basename(os.path.normpath(subdirs[subdir_id])))
                        seq_tags.append(subdirs[subdir_id][len(dir)+1:])
                        seq_fnames.append([])
                    seq_fnames[-1].append(subdirs_fnames[subdir_id][fname_id])

        elif self.directory_structure == 'SEQ-DATA':
            # x[0] dirs, x[1]: subdirs in dirs, x[2]: files  in dirs
            if self.seq_dirs is None:
                self.seq_dirs = glob.glob(self.data_load_root + '/*' * self.directory_structure_seq_depth)
                # only extract dirs
                self.seq_dirs = [seq_dir for seq_dir in self.seq_dirs if os.path.isdir(seq_dir)]
                # extract only subdirs relative to data_load_root
                self.seq_dirs = [seq_dir[len(self.data_load_root)+1:] for seq_dir in self.seq_dirs]
                # dont use meta information as seq_dir
                self.seq_dirs = [seq_dir for seq_dir in self.seq_dirs if 'meta' not in seq_dir]
                #for seqs_dirs_filter_tag in self.seqs_dirs_filter_tags:
                #    self.seq_dirs, _, _ = self.filter_subdirs_key(self.seq_dirs, None, None, seqs_dirs_filter_tag)
            seq_dirs = []
            seq_tags = []
            seq_fnames = []
            for i, seq_dir in enumerate(self.seq_dirs):
                seq_fnames.append(sorted(os.listdir(os.path.join(self.data_load_root, seq_dir, dir))))
                seq_dirs.append(os.path.join(self.data_load_root, seq_dir, dir))
                seq_tags.append(seq_dir)

        elif self.directory_structure == 'BROX':
            #/media/driveD/datasets/Brox_FlyingThings3D_part/train/flow

            walk_info = [x for x in os.walk(dir)]
            subdirs = [x[0] for x in walk_info]
            subdirs_fnames = [sorted(x[2]) for x in walk_info]

            subdirs, subdirs_fnames = self.filter_subdirs_not_empty(subdirs, subdirs_fnames)

            if '_l_' in data_load_key or data_load_key.endswith('_l'):
                subdirs, _, subdirs_fnames = self.filter_subdirs_key(subdirs, None, subdirs_fnames, 'left')
            if '_r_' in data_load_key or data_load_key.endswith('_r'):
                subdirs, _, subdirs_fnames = self.filter_subdirs_key(subdirs, None, subdirs_fnames, 'right')
            if '_01_' in data_load_key or data_load_key.endswith('_01'):
                subdirs, _, subdirs_fnames = self.filter_subdirs_key(subdirs, None, subdirs_fnames, 'into_future')
            if '_10_' in data_load_key or data_load_key.endswith('_10'):
                subdirs, _, subdirs_fnames = self.filter_subdirs_key(subdirs, None, subdirs_fnames, 'into_past')

            num_subdirs = len(subdirs)
            seq_dirs = []
            seq_tags = []
            seq_fnames = []

            for subdir_id in range(num_subdirs):
                num_subdir_fnames = len(subdirs_fnames[subdir_id])

                subdir_fname_curr_time_id = 0
                for fname_id in range(num_subdir_fnames):
                    subdir_fname_last_time_id = subdir_fname_curr_time_id
                    subdir_fname_curr_time_id = self.get_time_id_from_fname(subdirs_fnames[subdir_id][fname_id])
                    if fname_id == 0 or (subdir_fname_last_time_id + 1) != subdir_fname_curr_time_id:
                        seq_dirs.append(subdirs[subdir_id])
                        seq_tags.append(re.findall('[ABC]+/[0-9]+\\.?[0-9]*', subdirs[subdir_id])[-1])
                        seq_fnames.append([])
                    seq_fnames[-1].append(subdirs_fnames[subdir_id][fname_id])

        else:
            print('error: directory structure not implemented ', self.directory_structure)

        #for seqs_dirs_filter_tag in self.seqs_dirs_filter_tags:
        #    seq_dirs, seq_tags, seq_fnames = self.filter_subdirs_key(seq_dirs, seq_tags, seq_fnames, seqs_dirs_filter_tag)

        return seq_dirs, seq_tags, seq_fnames

    def filter_subdirs_key(self, subdirs, subdirs_tags, subdirs_fnames, key):
        subdirs_ids_filtered = [i for i in range(len(subdirs)) if key in subdirs[i]]
        subdirs = [subdirs[id] for id in range(len(subdirs)) if id in subdirs_ids_filtered]
        if subdirs_fnames is not None:
            subdirs_fnames = [subdirs_fnames[id] for id in range(len(subdirs_fnames)) if id in subdirs_ids_filtered]
        if subdirs_tags is not None:
             subdirs_tags = [subdirs_tags[id] for id in range(len(subdirs_tags)) if id in subdirs_ids_filtered]
        return subdirs, subdirs_tags, subdirs_fnames

    def filter_subdirs_not_empty(self, subdirs, subdirs_fnames):
        subdirs_ids_not_empty = [i for i in range(len(subdirs)) if len(subdirs_fnames[i]) > 0]
        subdirs = [subdirs[id] for id in range(len(subdirs)) if id in subdirs_ids_not_empty]
        subdirs_fnames = [subdirs_fnames[id] for id in range(len(subdirs_fnames)) if id in subdirs_ids_not_empty]
        return subdirs, subdirs_fnames

    def get_time_id_from_fname(self, fname):
        if "." in fname:
            fname = fname.split(".")[0]
        if "_" in fname:
            fname = fname.split("_")[1]
        return int(fname)

