import torch
import torch.utils.data as data
from glob import glob
from os.path import join, basename, exists
import numpy as np
import pickle as pkl
from random import random

class KNTTS_two(data.Dataset):
    def __init__(self, which_set='train', datapath='/home/thkim/data/KNTTS/bin', spkr_dict=['20m', '20f']):

        self.__dict__.update(locals())
        # Load vocabulary
        vocab_path = datapath + '/vocab_dict.pkl'
        self.vocab_dict = pkl.load(open(vocab_path, 'rb'))
        self.vocab_size = len(self.vocab_dict)

        # Filelist 
        self.txtlist = np.sort(glob(datapath+'/*.txt'))
        self.mellist = np.sort(glob(datapath+'/*.mel'))

        get_sent_no = lambda x: int(x.split('_')[-1][:-4])
        self.txtlist = [xx for xx in self.txtlist if get_sent_no(xx)> 2700]
        self.mellist = [xx for xx in self.mellist if get_sent_no(xx)> 2700]
        
        self.dbname = 'KNTTS'
        self.gen_lu = {'f': 0, 'm': 1}
        self.age_lu = {'age20': 0, 'age30': 1, 'age40': 2}
        self.emo_lu = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'sur': 4, 'fea': 5, 'dis': 6}
        self.spkr_dict = spkr_dict
        self.txtlist = [xx for xx in self.txtlist if basename(xx)[:3] in self.spkr_dict]
        self.mellist = [xx for xx in self.mellist if basename(xx)[:3] in self.spkr_dict]

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

        target_mel_name = basename(self.mellist[idx])
        spk = target_mel_name.split('_')[-3]
        sent_no = int(target_mel_name.split('_')[-1][:-4])

        while True:
            new_sent = np.random.randint(300) + 2700
            style_mel_name = f'{spk}_trim_{new_sent:05}.mel'
            style_mel_path = join(self.datapath, style_mel_name)
            if exists(style_mel_path):
                break

        while True:
            #new_spk = np.random.choice(list(self.spkr_dict))
            if spk == '20f':
                new_spk = '20m'
            else:
                new_spk = '20f'
            contents_mel_name = f'{new_spk}_trim_{sent_no:05}.mel'
            contents_mel_path = join(self.datapath, contents_mel_name)
            if exists(contents_mel_path):
                break

        contents_mel = pkl.load(open(contents_mel_path, 'rb'))['mel']
        style_mel = pkl.load(open(style_mel_path, 'rb'))['mel']
        style = self.getstyle(self.txtlist[idx])

        return {'txt': np.asarray(txt_feat), 
                'style': style, 
                'target_lin': np.asarray(lin), 
                'target_mel': np.asarray(mel),
                'style_mel': np.asarray(style_mel),
                'contents_mel': np.asarray(contents_mel),
                'filename': {'target':self.mellist[idx], 'style':style_mel_path, 'contents':contents_mel_path}
                }

    def getstyle(self, filename):
        filename = basename(filename)

        spkr = filename[:3]
        gender = self.gen_lu[spkr[2]]
        age = self.age_lu[f'age{spkr[:2]}']
        emotion = self.emo_lu['neu']
        spkr = self.spkr_lu['_'.join((self.dbname, spkr))]
        return {'age': age, 'gender': gender,'emotion': emotion, 'dbname': self.dbname, 'spkr': spkr}

    def get_vocab_size(self):
        return self.vocab_size

    def set_vocab_dict(self, vocab_dict):
        self.vocab_dict = vocab_dict
        self.vocab_size = len(vocab_dict)
        self.char2onehot = lambda x : self.vocab_dict[x] if x in self.vocab_dict.keys() else None 

    def set_spkr_lu(self, spkr_lu):
        self.spkr_lu = spkr_lu


if __name__=='__main__':
    aa = KNTTS()
    aa[0]
    import ipdb
    ipdb.set_trace()
