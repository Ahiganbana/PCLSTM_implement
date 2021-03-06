import os
import argparse
import datetime
from numpy.random.mtrand import shuffle

from torch.random import seed
import config.config as configurable
import torch
from model.model import LSTMModule, PCLSTMModule
import numpy as np
import random
import multiprocessing as mu
import train_model
import shutil
from dataLoader import mydataset
from torch.utils.data import DataLoader

seed_num = 223
torch.manual_seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)
torch.cuda.manual_seed(seed_num)

class PCModelArgs:
    def __init__(self, kernel_num, padding_size, input_dim):
        self.input_dim = input_dim
        self.kernel_num = kernel_num
        self.padding_size = padding_size
        self.dropout = 0.75
        self.cuda = True

class LSTMModelArgs:
    def __init__(self, lstm_hidden_dim, lstm_num_layers):
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.input_dim = 129
        self.init_weight = True
        self.init_weight_value = 2.0
        self.dropout = 0.75


def load_model():
    model = None
    if config.snapshot is None:
        if config.PCLSTM:
            print('loading PCLSTM model...')
            pcargs = []
            lstmargs = []
            args1 = PCModelArgs(36, 1, 14)
            args2 = PCModelArgs(36, 1, 36)
            args3 = PCModelArgs(144, 1, 36)
            args4 = PCModelArgs(144, 7, 144)
            args5 = LSTMModelArgs(129, 5)
            args6 = LSTMModelArgs(129, 5)
            pcargs.append(args1)
            pcargs.append(args2)
            pcargs.append(args3)
            pcargs.append(args4)
            lstmargs.append(args5)
            lstmargs.append(args6)
            # args7 = LSTMModelArgs(2000, 5)
            model = PCLSTMModule(pcargs, lstmargs)
            # model = LSTMModule(args7)
    else:
        print('\nLoading model from [%s]...' % config.snapshot)
        try:
            model = torch.load(config.snapshot)
        except:
            print('Sorry, This snapshot does not exist.')
            exit()
    if config.cuda is True:
        model = model.cuda()
    return model

def start_train(model, train_iter, dev_iter, test_iter):
    if config.predict is not None:
        pass
    elif config.test:
        pass
    else:
        print('\ncpu_count \n', mu.cpu_count())
        torch.set_num_threads(config.num_threads)
        if os.path.exists("./Test_Result.txt"):
            os.remove("./Test_Result.txt")
        if config.PCLSTM:
            print('PCLSTM training start.......')
            model_count = train_model.train(train_iter, dev_iter, test_iter, model, config)
        print('Model_count', model_count)


def mrs_five_mui(train_name, dev_name, test_name):
    """
    :function: load five-classification data
    :param path:
    :param train_name: train path
    :param dev_name: dev path
    :param test_name: test path
    :param char_data: char data
    :param text_field: text dict for finetune
    :param label_field: label dict for finetune
    :param static_text_field: text dict for static(no finetune)
    :param static_label_field: label dict for static(no finetune)
    :param kargs: others arguments
    :return: batch train, batch dev, batch test
    """
    train_data = mydataset.Mydataset(train_name)
    valid_data = mydataset.Mydataset(dev_name)
    test_data = mydataset.Mydataset(test_name)
    print("len(train_data) {} ".format(len(train_data)))
    print("len(valid_data) {} ".format(len(valid_data)))
    print("len(test_data) {} ".format(len(test_data)))
    train_dataloader = DataLoader(dataset=train_data, batch_size = 16, num_workers = 4, shuffle = False)
    valid_dataloader = DataLoader(dataset=valid_data, batch_size = 16, num_workers = 4, shuffle = False)
    test_dataloader = DataLoader(dataset=test_data, batch_size = 16, num_workers = 4, shuffle = False)
    return train_dataloader, valid_dataloader, test_dataloader

def Load_Data():
    train_iter, dev_iter, test_iter = None, None, None
    if config.MUTI_CLASS_TASK:
        print("Executing Muti Classification Task......")
        train_iter, dev_iter, test_iter = mrs_five_mui(config.name_trainfile, config.name_devfile, config.name_testfile)
    return train_iter, dev_iter, test_iter


def update_arguments():
    config.lr = config.learning_rate
    config.init_weight_decay = config.weight_decay
    config.init_clip_max_norm = config.clip_max_norm
    config.class_num = 6
    print(config.kernel_sizes)
    mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.mulu = mulu
    config.save_dir = os.path.join(""+config.save_dir, config.mulu)
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)



def main():
    update_arguments()
    model = load_model()
    train_iter, dev_iter, test_iter = Load_Data()
    start_train(model, train_iter, dev_iter, test_iter)


if __name__ == '__main__':
    print('Process ID {}, Process Parent ID {}'.format(os.getpid(), os.getppid()))
    parser = argparse.ArgumentParser(description='Neural Networks')
    parser.add_argument('--config_file', default='./config/config.cfg')
    config = parser.parse_args()

    config = configurable.Configurable(config_file=config.config_file)
    if config.cuda is True:
        print('Using GPU To Train...')
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        print('torch.cuda.initial_seed', torch.cuda.initial_seed())
    main()