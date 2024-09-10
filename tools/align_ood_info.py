import os
import json
import argparse
from tqdm import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dataroot', type=str, default="DATASET_DIR")
args = arg_parser.parse_args()

# VOC
print()
print("================> VOC as ID dataset <================")
info_path = os.path.join(args.dataroot, 'SyncOOD_VOC/info_raw.json')
save_path = os.path.join(args.dataroot, 'SyncOOD_VOC/info.json')
# import pdb; pdb.set_trace()
print(" + Checking synthetic OOD data annotation file ...")
if not os.path.exists(info_path):
    print(" + + Fail! :( Please check your data path!")
else:
    with open(info_path, 'r') as file:
        info_data = json.load(file)
    print(" + + Success! :) There are " + str(len(info_data)) + " image annotations.")

    print(" + Processing the data path...")
    save_info = []
    for curr_info in tqdm(info_data):
        curr_name = curr_info['file_name']
        curr_info['file_name'] = os.path.join(args.dataroot, 'SyncOOD_VOC/images/' + curr_name)
        save_info.append(curr_info)
    with open(save_path, 'w') as f:
        json.dump(save_info, f)
    print(" + + Finish! The processed file is saved as: " + save_path)


# BDD
print()
print("================> BDD as ID dataset <================")
info_path = os.path.join(args.dataroot, 'SyncOOD_BDD/info_raw.json')
save_path = os.path.join(args.dataroot, 'SyncOOD_BDD/info.json')
# import pdb; pdb.set_trace()
print(" + Checking synthetic OOD data annotation file ...")
if not os.path.exists(info_path):
    print(" + + Fail! :( Please check your data path!")
else:
    with open(info_path, 'r') as file:
        info_data = json.load(file)
    print(" + + Success! :) There are " + str(len(info_data)) + " image annotations.")

    print(" + Processing the data path...")
    save_info = []
    for curr_info in tqdm(info_data):
        curr_name = curr_info['file_name']
        curr_info['file_name'] = os.path.join(args.dataroot, 'SyncOOD_BDD/images/' + curr_name)
        save_info.append(curr_info)
    with open(save_path, 'w') as f:
        json.dump(save_info, f)
    print(" + + Finish! The processed file is saved as: " + save_path)