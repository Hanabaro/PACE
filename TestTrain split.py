import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import shutil
import pandas as pd
from pandas import HDFStore

# path to dataset saved as hdf5 file
dataset_path = 'C:/Users/yousf/Downloads/'
dataset_filename = 'dataset.hdf5'

# specify folder name in which the dataset will be saved in the form of multiple npy-files  
foldername='extracted_numpy'  
save_path=dataset_path+foldername+"/"
# open h5py file
with h5py.File(dataset_path+ dataset_filename, 'r') as f:
    
    # iterate through the subsets "test" and "train"
    for subset in list(f.keys()):
        print(subset)
       
        # iterate through all groups in current subset
        for substrate in tqdm(list(f[subset].keys())):
            
            # if current group is contains substrate data, i.e. starts with "A", interate through all datasets (which contain data of the patches) 
            if substrate[0]=="A":
                
                for patch in list(f[subset+"/"+substrate].keys()):
                    path=subset+"/"+substrate+"/"+patch
                    
                    dir_to_make=save_path+subset+"/"+substrate+"/"
                    if not os.path.exists(dir_to_make):
                        os.makedirs(dir_to_make)
                        
                    # save content of current dataset as npy-file contianing data of current patch
                    np.save(save_path+path+".npy", np.array(f[path]).astype('uint16'))
               
                    
# use HDFStore function to extract panda dataframes from hdf5 file        
hdf =HDFStore(dataset_path+ 'dataset.hdf5', mode='r')

labels_test=hdf.get('test/labels')
labels_train=hdf.get('train/labels')

fold0_train=hdf.get('train/cv_splits_5fold/fold0/train')
fold0_val=hdf.get('train/cv_splits_5fold/fold0/val')
fold1_train=hdf.get('train/cv_splits_5fold/fold1/train')
fold1_val=hdf.get('train/cv_splits_5fold/fold1/val')
fold2_train=hdf.get('train/cv_splits_5fold/fold2/train')
fold2_val=hdf.get('train/cv_splits_5fold/fold2/val')
fold3_train=hdf.get('train/cv_splits_5fold/fold3/train')
fold3_val=hdf.get('train/cv_splits_5fold/fold3/val')
fold4_train=hdf.get('train/cv_splits_5fold/fold4/train')
fold4_val=hdf.get('train/cv_splits_5fold/fold4/val')

hdf.close()

# generate direcotries to save information regarding cross-validation splits
for ix in range(5):
    dir_to_make=save_path+"train/cv_splits_5fold/fold"+str(ix)
    if not os.path.exists(dir_to_make):
        os.makedirs(dir_to_make)
        
# save pandas dataframe to csv files        
labels_test.to_csv(os.path.join(save_path+'test/', 'labels.csv'), index=False)
labels_train.to_csv(os.path.join(save_path+'train/', 'labels.csv'), index=False)

fold0_train.to_csv(os.path.join(save_path+'train/cv_splits_5fold/fold0/', 'train.csv'), index=False)
fold0_val.to_csv(os.path.join(save_path+'train/cv_splits_5fold/fold0/', 'val.csv'), index=False)
fold1_train.to_csv(os.path.join(save_path+'train/cv_splits_5fold/fold1/', 'train.csv'), index=False)
fold1_val.to_csv(os.path.join(save_path+'train/cv_splits_5fold/fold1/', 'val.csv'), index=False)
fold2_train.to_csv(os.path.join(save_path+'train/cv_splits_5fold/fold2/', 'train.csv'), index=False)
fold2_val.to_csv(os.path.join(save_path+'train/cv_splits_5fold/fold2/', 'val.csv'), index=False)
fold3_train.to_csv(os.path.join(save_path+'train/cv_splits_5fold/fold3/', 'train.csv'), index=False)
fold3_val.to_csv(os.path.join(save_path+'train/cv_splits_5fold/fold3/', 'val.csv'), index=False)
fold4_train.to_csv(os.path.join(save_path+'train/cv_splits_5fold/fold4/', 'train.csv'), index=False)
fold4_val.to_csv(os.path.join(save_path+'train/cv_splits_5fold/fold4/', 'val.csv'), index=False)  

print("done")
