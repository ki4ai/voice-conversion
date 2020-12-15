import torch
import torch.utils.data as data
from glob import glob
from os.path import join, basename, exists
import numpy as np
import pickle as pkl
from random import random

class KETTS_30m(data.Dataset):
    def __init__(self, which_set='train', datapath='/home/thkim/data/KETTS/30m_bin'):
        # Load vocabulary
        vocab_path = datapath + '/vocab_dict.pkl'
        self.vocab_dict = pkl.load(open(vocab_path, 'rb'))
        self.vocab_size = len(self.vocab_dict)

        # Filelist 
        self.txtlist = np.sort(glob(datapath+'/*.txt'))
        self.mellist = np.sort(glob(datapath+'/*.mel'))
        self.txtlist = [xx for xx in self.txtlist if not '00432' in xx and not '00480' in xx]
        self.mellist = [xx for xx in self.mellist if not '00432' in xx and not '00480' in xx]

        
        self.dbname = 'KETTS_30m'
        self.num_spkr = 1
        self.gen_lu = {'female': 0, 'male': 1}
        self.age_lu = {'age20': 0, 'age30': 1, 'age40': 2}
        self.emo_lu = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3, 'sur': 4, 'fea': 5, 'dis': 6}
        self.spkr_lu = {'_'.join((self.dbname, str(ii))): xx for ii, xx in enumerate(range(self.num_spkr))}

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

        #contents_mel_path = self.mellist[idx]
        mel_emo = basename(self.mellist[idx])[:3]
        emo_set = set(self.emo_lu.keys())
        emo_set.remove(mel_emo)
        while emo_set:
            input_emo = emo_set.pop()
            contents_mel_path = self.mellist[idx].replace(mel_emo, input_emo)
            if exists(contents_mel_path):
                break

        #ref_mel_path = self.mellist[idx]
        while True:
            sent_no = '{:05d}'.format(np.random.randint(3000))
            ref_mel_path = self.mellist[idx]
            ref_mel_path = ref_mel_path.replace(ref_mel_path[-9:-4], sent_no)
            if exists(ref_mel_path):
                break

        contents_mel = pkl.load(open(contents_mel_path, 'rb'))['mel']
        style_mel = pkl.load(open(ref_mel_path, 'rb'))['mel']
        style = self.getstyle(self.txtlist[idx])

        return {'txt': np.asarray(txt_feat), 
                'style': style, 
                'target_lin': np.asarray(lin), 
                'target_mel': np.asarray(mel),
                'style_mel': np.asarray(style_mel),
                'contents_mel': np.asarray(contents_mel),
                'filename': {'target':self.mellist[idx], 'ref':ref_mel_path, 'input':contents_mel_path}
                }

    def getstyle(self, filename):
        filename = basename(filename)
        gender = self.gen_lu['male']
        age = self.age_lu['age30']
        emotion = self.emo_lu[filename[:3]]
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
    tt = KETTS_30m()
    data = [tt[ii] for ii in range(10)]
    from collate_fn import collate_fn
    txt, mel, lin, contents_mel, txt_len, mel_len, gender, age, emotion, emb, filename = collate_fn(data)
   # txt, mel, lin ,gender, age, emotion= collate_fn(data)
    print(emb)
    import ipdb
    ipdb.set_trace()

   # print(len(tt), tt[0])
    #print(len(tt), tt[0])
