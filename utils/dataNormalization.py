import os
import pandas as pd

dir_path = 'D:/Learning/data/211128/pclstm_data'
new_dir_path = 'D:/Learning/data/211128/normalization'
max_set = []
min_set = []

def minmaxscaler(data, new_file_path):
    save_dict = {}
    i = 0
    for index_name, each_colum in data.iteritems():
        if index_name == 0 or index_name == 15:
            save_dict[i] = each_colum
            i += 1
            continue
        each_colum = (each_colum - min_set[index_name - 1]) / (max_set[index_name - 1] - min_set[index_name - 1])
        save_dict[i] = each_colum
        i += 1
    data_for_save = pd.DataFrame(save_dict)
    data_for_save.to_csv(new_file_path, encoding='gbk', header=None, index = False)

def find_max_and_min(data):
    for index, each_colnum in data.iteritems():
        if index == 0 or index == 15:
            continue
        max_data = each_colnum.max()
        min_data = each_colnum.min()
        if len(max_set) <= 13:
            max_set.append(max_data)
        else:
            if max_set[index - 1] < max_data:
                max_set[index - 1] = max_data
        if len(min_set) <= 13:
            min_set.append(min_data)
        else:
            if min_set[index - 1] > min_data:
                min_set[index - 1] = min_data
        



if not os.path.exists(new_dir_path):
    os.mkdir(new_dir_path)

pathDir = os.listdir(dir_path)

for dir_item in pathDir:
    full_dir_item = os.path.join('%s/%s' % (dir_path, dir_item))
    full_new_dir_path = os.path.join('%s/%s' % (new_dir_path, dir_item))
    # print(full_new_dir_path)
    if not os.path.exists(full_new_dir_path):
        os.mkdir(full_new_dir_path)
    sub_dir_list = os.listdir(full_dir_item)
    for sub_dir_item in sub_dir_list:
        full_sub_dir_item = os.path.join('%s/%s' % (full_dir_item, sub_dir_item))
        full_sub_new_dir_item = os.path.join('%s/%s' % (full_new_dir_path, sub_dir_item))
        if not os.path.exists(full_sub_new_dir_item):
            os.mkdir(full_sub_new_dir_item)
        # print(full_sub_dir_item)
        # print(full_sub_new_dir_item)
        for _, _, files in os.walk(full_sub_dir_item):
            for file in files:
                full_file_path = os.path.join('%s/%s' % (full_sub_dir_item, file))
                full_new_file_path = os.path.join('%s/%s' % (full_sub_new_dir_item, 'n' + file))
                # print(full_new_file_path)
                if os.path.splitext(file)[1] == '.csv':
                    data_row = pd.read_csv(full_file_path, encoding='gb2312', header=None)
                    find_max_and_min(data_row)
                    # minmaxscaler(data_row, full_new_file_path)
print(max_set)
print(min_set)

for dir_item in pathDir:
    full_dir_item = os.path.join('%s/%s' % (dir_path, dir_item))
    full_new_dir_path = os.path.join('%s/%s' % (new_dir_path, dir_item))
    # print(full_new_dir_path)
    if not os.path.exists(full_new_dir_path):
        os.mkdir(full_new_dir_path)
    sub_dir_list = os.listdir(full_dir_item)
    for sub_dir_item in sub_dir_list:
        full_sub_dir_item = os.path.join('%s/%s' % (full_dir_item, sub_dir_item))
        full_sub_new_dir_item = os.path.join('%s/%s' % (full_new_dir_path, sub_dir_item))
        if not os.path.exists(full_sub_new_dir_item):
            os.mkdir(full_sub_new_dir_item)
        # print(full_sub_dir_item)
        # print(full_sub_new_dir_item)
        for _, _, files in os.walk(full_sub_dir_item):
            for file in files:
                full_file_path = os.path.join('%s/%s' % (full_sub_dir_item, file))
                full_new_file_path = os.path.join('%s/%s' % (full_sub_new_dir_item, 'n' + file))
                # print(full_new_file_path)
                if os.path.splitext(file)[1] == '.csv':
                    data_row = pd.read_csv(full_file_path, encoding='gb2312', header=None)
                    minmaxscaler(data_row, full_new_file_path)

