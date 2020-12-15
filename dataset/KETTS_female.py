import torch
import torch.utils.data as data
from glob import glob
from os.path import join, basename, exists
import numpy as np
import pickle as pkl
from random import random

class KETTS_30f(data.Dataset):
    def __init__(self, which_set='train', datapath='/home/thkim/data/KETTS/30f_bin', mismatch_style=True, mismatch_contents=True):
        '''
        mismatch_style means sampling style_wav of which contents is different from that of target_wav
        mismatch_contents means sampling style_wav of which style is different from that of target_wav
        '''
        self.__dict__.update(locals())
        # Load vocabulary
        vocab_path = datapath + '/vocab_dict.pkl'
        self.vocab_dict = pkl.load(open(vocab_path, 'rb'))
        self.vocab_size = len(self.vocab_dict)
        self.num_spkr = 1

        # Filelist 
        self.txtlist = np.sort(glob(datapath+'/*.txt'))
        self.mellist = np.sort(glob(datapath+'/*.mel'))
        
        self.dbname = 'KETTS_30f'
        self.gen_lu = {'female': 0, 'male': 1}
        self.age_lu = {'age20': 0, 'age30': 1, 'age40': 2}
        self.emo_lu = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'sur': 4, 'fea': 5, 'dis': 6}
        self.spkr_lu = {'_'.join((self.dbname, str(ii))): xx for ii, xx in enumerate(range(self.num_spkr))}
        assert len(self.txtlist)==len(self.mellist), \
                'mellist({}) and txtlist({}) has different length'.format(len(self.mellist), len(self.txtlist))

        self.char2onehot = lambda x : self.vocab_dict[x] if x in self.vocab_dict.keys() else None 
        self.get_emo = lambda x: x.split('_')[1]

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
        emo = target_mel_name.split('_')[-2]
        sent_no = int(target_mel_name.split('_')[-1][:-4])

        if self.mismatch_contents:
            while True:
                new_sent = np.random.randint(3000)
                style_mel_name = f'acriil_{emo}_{new_sent:08}.mel'
                style_mel_path = join(self.datapath, style_mel_name)
                if exists(style_mel_path):
                    break
                
        if self.mismatch_style:
            while True:
                new_emo = np.random.choice(list(self.emo_lu.keys()))
                contents_mel_name = f'acriil_{new_emo}_{sent_no:08}.mel'
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
        gender = self.gen_lu['female']
        age = self.age_lu['age30']
        emotion = self.emo_lu[filename.split('_')[1]]
        spkr = self.spkr_lu['_'.join((self.dbname, str(0)))]
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
