import os


def get_imgpairs_from_filepath(filepath, type="monodepth2"):

    # Using readlines()
    file = open(filepath, "r")
    lines = file.readlines()
    file.close()
    imgpairs = []

    if type == "monodepth2":
        for line in lines:
            drive_id, img_id, l = line.split("/")[1].split(" ")
            # print(drive_id, img_id, l)
            imgpairs.append(drive_id + " " + img_id.zfill(10))
            # imgpairs.append(drive_id + ' ' + str(int(img_id)+1).zfill(10))

    elif type == "smsf":
        for line in lines:
            drive_id, img_id = line.split(" ")
            img_id = img_id.rstrip("\n")
            print(drive_id, img_id)
            imgpairs.append(drive_id + " " + img_id.zfill(10))
            # imgpairs.append(drive_id + ' ' + str(int(img_id) + 1).zfill(10))
    return imgpairs


monodepth2_imgpairs = []

dir = "eigen_zhou"
monodepth2_imgpairs += get_imgpairs_from_filepath(os.path.join(dir, "train_files.txt"))
monodepth2_imgpairs += get_imgpairs_from_filepath(os.path.join(dir, "val_files.txt"))

monodepth2_imgpairs = list(set(monodepth2_imgpairs))
monodepth2_imgpairs = sorted(monodepth2_imgpairs)
# print(imgpairs)
print("found", len(monodepth2_imgpairs), "imgpairs")

smsf_filename = "train_files.txt"
# smsf_filename = 'val_files.txt'
smsf_imgpairs = []
smsf_dir = "smsf"
smsf_imgpairs += get_imgpairs_from_filepath(
    os.path.join(smsf_dir, smsf_filename), type="smsf"
)
# smsf_imgpairs += get_imgpairs_from_filepath(os.path.join(smsf_dir, 'val_files.txt'), type='smsf')
smsf_imgpairs = list(set(smsf_imgpairs))
smsf_imgpairs = sorted(smsf_imgpairs)

smsf_imgpairs_reduced = set(smsf_imgpairs) - (
    set(smsf_imgpairs) - set(monodepth2_imgpairs)
)
smsf_imgpairs_reduced = sorted(list(smsf_imgpairs_reduced))

"""
img_id_recent = -1
last_indices = []
for i, line in enumerate(sorted(smsf_imgpairs_reduced)):
    drive_id, img_id = line.split(' ')
    img_id = int(img_id.rstrip("\n"))

    if img_id_recent != -1 and img_id_recent+1 != img_id:
        last_indices.append(i-1)
    img_id_recent = img_id

for index in sorted(last_indices, reverse=True):
    del smsf_imgpairs_reduced[index]
"""

print("found", len(smsf_imgpairs_reduced), "imgpairs")

smsf_eigen_zhou_dir = "smsf_eigen_zhou"

file = open(os.path.join(smsf_eigen_zhou_dir, smsf_filename), "w")
for line in sorted(smsf_imgpairs_reduced):
    file.write(line)
    file.write("\n")
file.close()
