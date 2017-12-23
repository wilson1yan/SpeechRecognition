import os
import glob

from scipy.io import wavfile
import torch
from torch.utils.data import Dataset

class GoogleVoiceDataset(Dataset):
    def __init__(self, root, preprocessor, mode='train', window_size=0.02):
        super(GoogleVoiceDataset, self).__init__()
        self.root = root
        self.window_size = window_size
        
        with open(os.path.join(root, 'validation_list.txt'), 'r') as f:
            self.valset = [os.path.join(root, file) for file in f.read().split('\n') if file]
        with open(os.path.join(root, 'testing_list.txt'), 'r') as f:
            self.testset = [os.path.join(root, file) for file in f.read().split('\n') if file]
        self.trainset = set(glob.glob(os.path.join(root, '*', '**.wav'), recursive=True))
        self.trainset = self.trainset - set(self.valset) - set(self.testset)
        self.trainset = list(self.trainset)
        self.set_mode(mode)
        
        self.preprocessor = preprocessor
        self.label_id_map = {label: i for i, label in enumerate(self.get_labels())}
        self.id_label_map = {v: k for k, v in self.label_id_map.items()}
        self.n_classes = len(self.label_id_map)
    
    def set_mode(self, mode):
        if mode == 'train':
            self.currentset = self.trainset
        elif mode == 'val':
            self.currentset = self.valset
        elif mode == 'test':
            self.currentset = self.testset
        else:
            raise ValueError('mode must be train, val, or test')
        self.mode = mode
        
    def get_labels(self):
        files = os.listdir(self.root)
        return [file for file in files if os.path.isdir(os.path.join(self.root, file))]
    
    def get_parent_dir(self, filepath):
        return filepath.split('/')[-2]
    
    def __len__(self):
        return len(self.currentset)
    
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('index: %s out of bounds for size %s' % (index, len(self)))
        fname = self.currentset[index]
        label = self.get_parent_dir(fname)
        rate, data = wavfile.read(fname)
        
        x = torch.FloatTensor(self.preprocessor(data, rate, self.window_size).astype(float))
        y = self.label_id_map[label]
        return x, y
