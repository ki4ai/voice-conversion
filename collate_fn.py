import torch
import torch.utils.data as data
import random
import numpy as np

class collate_class:
    def __init__(self, use_txt):
        self.use_txt = use_txt

    def fn(self, data):
        if random.random() < self.use_txt:
            contents_domain = 'txt'
        else:
            contents_domain = 'contents_mel'

        for x in data:
            if 'contents_mel' not in x.keys():
                x['contents_mel'] = x['target_mel']
            if 'style_mel' not in x.keys():
                x['style_mel'] = x['target_mel']

        n_batch = len(data)
        data.sort(key=lambda x: len(x[contents_domain]), reverse=True)
    
        contents_mel_len = torch.tensor([len(x['contents_mel']) for x in data])
        max_contents_mel_len = max(contents_mel_len)
        style_mel_len = torch.tensor([len(x['style_mel']) for x in data])
        max_style_mel_len = max(style_mel_len)
        target_mel_len = torch.tensor([len(x['target_mel']) for x in data])
        max_target_mel_len = max(target_mel_len)
        txt_len = torch.tensor([len(x['txt']) for x in data])
        max_txt_len = max(txt_len)
        max_lin_len = max([len(x['target_lin']) for x in data])

        txt = torch.zeros(n_batch, max_txt_len).long()
        target_mel = torch.zeros(n_batch, max_target_mel_len, data[0]['target_mel'].shape[-1])
        style_mel = torch.zeros(n_batch, max_style_mel_len, data[0]['style_mel'].shape[-1])
        contents_mel = torch.zeros(n_batch, max_contents_mel_len, data[0]['contents_mel'].shape[-1])
        target_lin = torch.zeros(n_batch, max_lin_len, data[0]['target_lin'].shape[-1])

        gender = torch.zeros(n_batch).long()
        age = torch.zeros(n_batch).long()
        emotion = torch.zeros(n_batch).long()
        spkemb = torch.zeros((n_batch, 256))
        filename = []

        for ii, item in enumerate(data):
            style_mel[ii, :len(item['style_mel'])] = torch.tensor(item['style_mel'])
            target_mel[ii, :len(item['target_mel'])] = torch.tensor(item['target_mel'])
            contents_mel[ii, :len(item['contents_mel'])] = torch.tensor(item['contents_mel'])
            target_lin[ii, :len(item['target_lin'])] = torch.tensor(item['target_lin'])
            txt[ii, :len(item['txt'])] = torch.tensor(item['txt']).long()

            gender[ii]  = item['style']['gender']
            age[ii]     = item['style']['age']
            emotion[ii] = item['style']['emotion']
            if 'speaker' in item['style'].keys():
                spkemb[ii] = torch.tensor(item['style']['speaker'])
            filename.append(item['filename'])

        out_list = ['target_mel', 'target_lin', 'txt', 'contents_mel', 'style_mel',
                    'target_mel_len', 'txt_len', 'contents_mel_len', 'style_mel_len',
                    'gender', 'age', 'emotion', 'spkemb', 'filename',
                    'contents_domain']

        if not contents_domain == 'txt':
            out_list = [xx for xx in out_list if xx not in ['txt', 'txt_len']]
        elif not contents_domain == 'contents_mel':
            out_list = [xx for xx in out_list if xx not in ['contents_mel', 'contents_mel_len']]

        return_dict = {k:v for k, v in locals().items() if k in out_list}

        assert len(out_list) == len(return_dict)

        return return_dict 
