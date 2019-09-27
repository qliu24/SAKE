import os,cv2
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import warp, AffineTransform
from nltk.corpus import wordnet as wn
import pickle

def random_transform(img):
    if np.random.random() < 0.5:
        img = img[:,::-1,:]

    if np.random.random() < 0.5:
        sx = np.random.uniform(0.7, 1.3)
        sy = np.random.uniform(0.7, 1.3)
    else:
        sx = 1.0
        sy = 1.0

    if np.random.random() < 0.5:
        rx = np.random.uniform(-30.0*2.0*np.pi/360.0,+30.0*2.0*np.pi/360.0)
    else:
        rx = 0.0

    if np.random.random() < 0.5:
        tx = np.random.uniform(-10,10)
        ty = np.random.uniform(-10,10)
    else:
        tx = 0.0
        ty = 0.0

    aftrans = AffineTransform(scale=(sx, sy), rotation=rx, translation=(tx,ty))
    img_aug = warp(img,aftrans.inverse,preserve_range=True).astype('uint8')

    return img_aug


class TUBerlinDataset(Dataset):
    def __init__(self, split='train',
                 root_dir='../dataset/TUBerlin/',
                 version='png_ready', zero_version='zeroshot', \
                 cid_mask = False, transform=None, aug=False, shuffle=False, first_n_debug=9999999):
        
        self.root_dir = root_dir
        self.version = version
        self.split = split
        
        self.img_dir = self.root_dir
        
        if self.split == 'train':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_train.txt')
        elif self.split == 'val':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_test.txt')
        elif self.split == 'zero':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version+'_filelist_zero.txt')
        else:
            print('unknown split for dataset initialization: ' + self.split)
            return
        
        with open(file_ls_file, 'r') as fh:
            file_content = fh.readlines()
            
        self.file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        self.labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
        if shuffle:
            self.shuffle()
            
        self.file_ls = self.file_ls[:first_n_debug]
        self.labels = self.labels[:first_n_debug]
            
        self.transform = transform
        self.aug = aug
        
        self.cid_mask = cid_mask
        if cid_mask:
            cid_mask_file = os.path.join(self.root_dir, zero_version, 'cid_mask.pickle')
            with open(cid_mask_file, 'rb') as fh:
                self.cid_matrix = pickle.load(fh)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.file_ls[idx]))[:,:,::-1]
        if self.aug and np.random.random()<0.7:
            img = random_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]
        
        if self.cid_mask:
            mask = self.cid_matrix[label]
            return img, label, mask

        return img, label
    
    
    def shuffle(self):
        s_idx = np.random.shuffle(np.arange(len(self.labels)))
        self.file_ls = self.file_ls[s_idx]
        self.labels = self.labels[s_idx]
        
            
        
        
    
def wnid_to_synset(wnid):
    return wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))
        
