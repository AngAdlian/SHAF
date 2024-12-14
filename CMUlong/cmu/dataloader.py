import time

from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from cmu import data_utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from torch.utils.data import Dataset
import pickle as pkl
import numpy as np
from os import walk
from h5py import File
import scipy.io as sio
from matplotlib import pyplot as plt
import torch

class H36motion3D(Dataset):
    def __init__(self, actions='all', input_n=10, output_n=10, split=0, scale=100, sample_rate=2):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        path_to_data = 'h36m/dataset'
        
        self.path_to_data = path_to_data 
        self.split = split  
        self.input_n = input_n 
        self.output_n = output_n 

        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        

        acts = data_utils.define_actions(actions)

        self.path_to_data = path_to_data

        subjs = subs[split] 
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate, input_n + output_n)
        
        
        
        
        all_seqs = all_seqs / scale 
        self.all_seqs_ori = all_seqs.copy()
        self.dim_used = dim_used 
        all_seqs = all_seqs[:, :, dim_used] 
        all_seqs = all_seqs.reshape(all_seqs.shape[0],all_seqs.shape[1],-1,3) 
        all_seqs = all_seqs.transpose(0,2,1,3) 
        all_seqs_vel = np.zeros_like(all_seqs)
        all_seqs_vel[:,:,1:] = all_seqs[:,:,1:] - all_seqs[:,:,:-1] 
        all_seqs_vel[:,:,0] = all_seqs_vel[:,:,1]

        self.all_seqs = all_seqs
        self.all_seqs_vel = all_seqs_vel

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        loc_data = self.all_seqs[item]
        vel_data = self.all_seqs_vel[item]
        loc_data_ori = self.all_seqs_ori[item] 
        return loc_data[:,:self.input_n], vel_data[:,:self.input_n], loc_data[:,self.input_n:self.input_n+self.output_n],loc_data_ori[self.input_n:self.input_n+self.output_n],item 



class DPW_Datasets(Dataset):
    def __init__(self, actions='all', input_n=10, output_n=10, split=0, scale=100, sample_rate=2):
        path_to_data = './3DPW/sequenceFiles'
        self.input_n =  input_n
        self.output_n = output_n

        
        
        
        their_input_n = input_n

        seq_len = their_input_n + output_n

        if split == 0:
            self.data_path = path_to_data + '/train/'
        elif split == 1:
            self.data_path = path_to_data + '/validation/'
        elif split == 2:
            self.data_path = path_to_data + '/test/'
        all_seqs = []
        files = []

        
        for (dirpath, dirnames, filenames) in walk(self.data_path):
            files.extend(filenames)
        for f in files:
            with open(self.data_path + f, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                joint_pos = data['jointPositions']
                for i in range(len(joint_pos)):
                    seqs = joint_pos[i]
                    seqs = seqs - seqs[:, 0:3].repeat(24, axis=0).reshape(-1, 72)

                    n_frames = seqs.shape[0]
                    fs = np.arange(0, n_frames - seq_len + 1)
                    fs_sel = fs
                    for j in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + j + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = seqs[fs_sel, :]

                    if len(all_seqs) == 0:
                        all_seqs = seq_sel
                    else:
                        all_seqs = np.concatenate((all_seqs, seq_sel), axis=0)

        

        self.dim_used = np.array(range(3, all_seqs.shape[2]))  
        


        

        all_seqs = all_seqs * 10

        self.all_seqs_ori = all_seqs.copy()
        dim_used = self.dim_used  
        all_seqs = all_seqs[:, :, dim_used]  
        all_seqs = all_seqs.reshape(all_seqs.shape[0], all_seqs.shape[1], -1, 3)  
        all_seqs = all_seqs.transpose(0, 2, 1, 3)  
        all_seqs_vel = np.zeros_like(all_seqs)  
        all_seqs_vel[:, :, 1:] = all_seqs[:, :, 1:] - all_seqs[:, :, :-1]  
        all_seqs_vel[:, :, 0] = all_seqs_vel[:, :, 1]  

        self.all_seqs = all_seqs  
        self.all_seqs_vel = all_seqs_vel  

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):

        loc_data = self.all_seqs[item]  
        vel_data = self.all_seqs_vel[item]
        loc_data_ori = self.all_seqs_ori[item]  
        return loc_data[:, :self.input_n], vel_data[:, :self.input_n], loc_data[:,self.input_n:self.input_n + self.output_n], loc_data_ori[self.input_n:self.input_n + self.output_n], item  



class CMU_Motion3D(Dataset):

    def __init__(self, actions='all', input_n=10, output_n=10, split=0, scale=100, sample_rate=2):

        self.path_to_data = './cmu_mocap'
        self.input_n =  input_n
        self.output_n = output_n

        self.split = split
        is_all = actions
        actions = data_utils.define_actions_cmu(actions)
        
        if split == 0:
            path_to_data = self.path_to_data + '/train/'
            is_test = False
        else:
            path_to_data = self.path_to_data + '/test/'
            is_test = True


        if not is_test:
            all_seqs, dim_ignore, dim_used = data_utils.load_data_cmu_3d_all(path_to_data, actions,
                                                                                             input_n, output_n,
                                                                                             is_test=is_test)
        else:
            
            
            

            all_seqs, dim_ignore, dim_used = data_utils.load_data_cmu_3d_n(path_to_data, actions,
                                                                                             input_n, output_n,
                                                                                             is_test=is_test)

        
        


        all_seqs = all_seqs / scale 
        self.all_seqs_ori = all_seqs.copy()
        self.dim_used = dim_used 
        all_seqs = all_seqs[:, :, dim_used] 
        all_seqs = all_seqs.reshape(all_seqs.shape[0],all_seqs.shape[1],-1,3) 
        all_seqs = all_seqs.transpose(0,2,1,3) 
        all_seqs_vel = np.zeros_like(all_seqs)
        all_seqs_vel[:,:,1:] = all_seqs[:,:,1:] - all_seqs[:,:,:-1] 
        all_seqs_vel[:,:,0] = all_seqs_vel[:,:,1]

        self.all_seqs = all_seqs
        self.all_seqs_vel = all_seqs_vel



    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):

        loc_data = self.all_seqs[item]  
        vel_data = self.all_seqs_vel[item]
        loc_data_ori = self.all_seqs_ori[item]  
        return loc_data[:, :self.input_n], vel_data[:, :self.input_n], loc_data[:,self.input_n:self.input_n + self.output_n], loc_data_ori[self.input_n:self.input_n + self.output_n], item  

import numpy as np
import pickle
class NTU_Motion3D(Dataset):

    def __init__(self, actions='all', input_n=10, output_n=10, split=0, scale=100, sample_rate=2):

        self.input_n =  input_n
        self.output_n = output_n






        self.split = split
        if split == 0:
            path_to_data = "D:/python_relevent/EqMotion-main/ntu_process/train_data.npy"
            path_to_pkl="D:/python_relevent/EqMotion-main/ntu_process/train_label.pkl"
            is_test = False
        else:
            path_to_data = "D:/python_relevent/EqMotion-main/ntu_process/eval_data.npy"
            path_to_pkl="D:/python_relevent/EqMotion-main/ntu_process/eval_label.pkl"
            is_test = True


        if not is_test:
            all_seqs, dim_ignore, dim_used = data_utils.load_data_ntu(path_to_data,path_to_pkl,input_n, output_n)
        else:

            all_seqs, dim_ignore, dim_used = data_utils.load_data_ntu(path_to_data,path_to_pkl, input_n, output_n)

        
        


        all_seqs = all_seqs*10  
        

        self.dim_used = dim_used 
        self.all_seqs_ori = all_seqs[:, :, dim_used]
        all_seqs = all_seqs[:, :, dim_used] 

        all_seqs = all_seqs.reshape(all_seqs.shape[0],all_seqs.shape[1],-1,3) 
        all_seqs = all_seqs.transpose(0,2,1,3) 
        all_seqs_vel = np.zeros_like(all_seqs)
        all_seqs_vel[:,:,1:] = all_seqs[:,:,1:] - all_seqs[:,:,:-1] 
        all_seqs_vel[:,:,0] = all_seqs_vel[:,:,1]

        self.all_seqs = all_seqs
        self.all_seqs_vel = all_seqs_vel



    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):

        loc_data = self.all_seqs[item]  
        vel_data = self.all_seqs_vel[item]
        loc_data_ori = self.all_seqs_ori[item]  
        return loc_data[:, :self.input_n], vel_data[:, :self.input_n], loc_data[:,self.input_n:self.input_n + self.output_n], loc_data_ori[self.input_n:self.input_n + self.output_n], item  



