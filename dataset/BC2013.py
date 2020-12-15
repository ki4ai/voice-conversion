import torch
import torch.utils.data as data
from glob import glob
from os.path import join, basename, exists
import numpy as np
import pickle as pkl
from random import random

class BC2013(data.Dataset):
    def __init__(self, which_set='train', datapath='/hdd1/home/thkim/data/tts/blizzard2013/lessac/train/segmented'):
        # Load vocabulary
        vocab_path = datapath + '/vocab_dict.pkl'
        self.vocab_dict = pkl.load(open(vocab_path, 'rb'))
        self.vocab_size = len(self.vocab_dict)

        # Filelist 
        self.txtlist = np.sort(glob(datapath+'/bin/*.txt'))
        self.mellist = np.sort(glob(datapath+'/bin/*.mel'))
        
        self.gen_lu = {'female': 0, 'male': 1}
        self.age_lu = {'age20': 0, 'age30': 1, 'age40': 2}
        self.emo_lu = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'sur': 4, 'fea': 5, 'dis': 6}
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
                'style_mel': np.asarray(mel),
                'contents_mel': np.asarray(mel),
                'filename': {'target':self.mellist[idx]}
                }

    def getstyle(self, filename):
        filename = basename(filename)
        gender = self.gen_lu['male']
        age = self.age_lu['age30']
        emotion = self.emo_lu['neu']
        return {'age': age, 'gender': gender,'emotion': emotion}

    def get_vocab_size(self):
        return self.vocab_size

if __name__=='__main__':
    #stat('/hdd1/home/thkim/data/temp/bin')
    tt = TTSDataset()
    data = [tt[ii] for ii in range(10)]
    from collate_fn import collate_fn
    txt, mel, lin, contents_mel, txt_len, mel_len, gender, age, emotion, emb, filename = collate_fn(data)
   # txt, mel, lin ,gender, age, emotion= collate_fn(data)
    print(emb)
    import ipdb
    ipdb.set_trace()

   # print(len(tt), tt[0])
    #print(len(tt), tt[0])
