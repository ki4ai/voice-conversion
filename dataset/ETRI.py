import torch
import torch.utils.data as data
from glob import glob
from os.path import join, basename, exists
import numpy as np
import pickle as pkl
from random import random
import os
HOME = os.environ['HOME']

class ETRI(data.Dataset):
    def __init__(self, which_set='train', datapath=HOME+'/data/etri'):
        # Load vocabulary
        vocab_path = datapath + '/bin/vocab_dict.pkl'
        self.vocab_dict = pkl.load(open(vocab_path, 'rb'))
        self.vocab_size = len(self.vocab_dict)

        # Filelist 
        self.txtlist = np.sort(glob(datapath+'/bin/*.txt'))
        self.mellist = np.sort(glob(datapath+'/bin/*.mel'))
        
        self.dbname = 'ETRI'
        self.gen_lu = {'female': 0, 'male': 1}
        self.age_lu = {'age20': 0, 'age30': 1, 'age40': 2}
        self.emo_lu = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'sur': 4, 'fea': 5, 'dis': 6}
        self.spkr_dict = ['F', 'M', 'TCCF', 'TCCM']
        self.num_spkr = len(self.spkr_dict)
        self.spkr_lu = {'_'.join((self.dbname, self.spkr_dict[ii])): xx for ii, xx in enumerate(range(self.num_spkr))}

        assert len(self.txtlist)==len(self.mellist), \
                'mellist({}) and txtlist({}) has different length'.format(len(self.mellist), len(self.txtlist))

        self.char2onehot = lambda x : self.vocab_dict[x] if x in self.vocab_dict.keys() else None 

    def __len__(self):
        return len(self.txtlist)

    def __getitem__(self, idx):
        # Text read
        with open(self.txtlist[idx], 'r') as f:
            txt = f.readline()
        txt_feat = list(filter(None, [self.char2onehot(xx) for xx in txt]))

        # Mel/Lin read
        mellin = pkl.load(open(self.mellist[idx], 'rb'))
        mel = mellin['mel']
        lin = mellin['lin']
        style = self.getstyle(self.txtlist[idx])

        return {'txt': np.asarray(txt_feat), 
                'style': style, 
                'target_lin': np.asarray(lin), 
                'target_mel': np.asarray(mel),
                'filename': {'target':self.mellist[idx]}
                }

    def getstyle(self, filename):
        age = self.age_lu['age30']
        emotion = 0 # neutral
        fname = basename(filename)
        if fname.startswith('F'):
            spkr = 'F'
            gender = self.gen_lu['female']
        elif fname.startswith('M'):
            spkr = 'M'
            gender = self.gen_lu['male']
        elif fname.startswith('TCCF'):
            spkr = 'TCCF'
            gender = self.gen_lu['female']
        elif fname.startswith('TCCM'):
            spkr = 'TCCM'
            gender = self.gen_lu['male']
        else:
            raise ValueError

        spkr = self.spkr_lu['_'.join((self.dbname, spkr))]
        return {'age': age, 'gender': gender,'emotion': emotion, 'spkr': spkr}

    def get_vocab_size(self):
        return self.vocab_size

    def set_vocab_dict(self, vocab_dict):
        self.vocab_dict = vocab_dict
        self.vocab_size = len(vocab_dict)
        self.char2onehot = lambda x : self.vocab_dict[x] if x in self.vocab_dict.keys() else None

    def set_spkr_lu(self, spkr_lu):
        self.spkr_lu = spkr_lu


if __name__=='__main__':
    aa = ETRI()
    print(aa[0])

