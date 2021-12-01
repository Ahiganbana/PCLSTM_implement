import os
import numpy as np
import shutil

dir_path = r'D:\Learning\data\211128\compress'
train_path = r'D:\Learning\data\211128\train'
test_path = r'D:\Learning\data\211128\test'
valid_path = r'D:\Learning\data\211128\valid'
ratio = [0.8, 0.1, 0.1]
dir_list = os.listdir(dir_path)

for dir_item in dir_list:
    files_path = os.path.join(dir_path, dir_item)
    train_new_path = os.path.join(train_path, dir_item)
    valid_new_path = os.path.join(valid_path, dir_item)
    test_new_path = os.path.join(test_path, dir_item)
    if not os.path.exists(train_new_path):
        os.makedirs(train_new_path)
    if not os.path.exists(valid_new_path):
        os.makedirs(valid_new_path)
    if not os.path.exists(test_new_path):
        os.makedirs(test_new_path)
    files_list = os.listdir(files_path)
    files_count = len(files_list)
    random_list = np.random.permutation(files_count)
    train_random_list = random_list[0: int(files_count * ratio[0])]
    valid_random_list = random_list[int(files_count * ratio[0]): int(files_count * ratio[0]) + int(files_count * ratio[1])]
    test_random_list = random_list[int(files_count * ratio[0]) + int(files_count * ratio[2]):]
    for item in train_random_list:
         move_train_file_path = os.path.join(files_path, files_list[item])
         shutil.move(move_train_file_path, train_new_path)
    for item in valid_random_list:
        move_valid_file_path = os.path.join(files_path, files_list[item])
        shutil.move(move_valid_file_path, valid_new_path)
    for item in test_random_list:
        move_test_file_path = os.path.join(files_path, files_list[item])
        shutil.move(move_test_file_path, test_new_path)
