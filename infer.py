import argparse, multiprocessing, os, time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
from os.path import join, basename, splitext, exists

from model import Tacotron as Tacotron
from dataset.get_dataset import get_dataset 
from collate_fn import collate_class 
import numpy as np
from util import *
import wandb
import random
from generate import spectrogram2wav_gpu, saveAttention
from stft import STFT
import librosa
import shutil

def main():
    parser = argparse.ArgumentParser(description='training script')
    # Mendatory arguments
    parser.add_argument('--data', type=str, nargs='+', default=['KNTTS_two'], help='dataset type')
    parser.add_argument('--dbroot', type=str, default='/home/thkim/data/')

    # data load
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    # model
    parser.add_argument('--charvec_dim', type=int, default=256, help='')
    parser.add_argument('--hidden_size', type=int, default=128, help='')
    parser.add_argument('--dec_out_size', type=int, default=80, help='decoder output size')
    parser.add_argument('--post_out_size', type=int, default=1025, help='should be n_fft / 2 + 1(check n_fft from "input_specL" ')
    parser.add_argument('--style_embed_size', type=int, default=32, help='should be n_fft / 2 + 1(check n_fft from "input_specL" ')
    parser.add_argument('--num_filters', type=int, default=16, help='number of filters in filter bank of CBHG')
    parser.add_argument('--r_factor', type=int, default=5, help='reduction factor(# of multiple output)')
    parser.add_argument('--use_txt', type=float, default=0.5, help='0~1, higher value means y_t batch is more sampled')
    parser.add_argument('--freeze_pretrained', action='store_true', default=False, help='whether to freeze pretrained model')
    parser.add_argument('--except_for', type=str, nargs='+', help='Weights contain this key would be excluded from frozen')

    # optimization
    parser.add_argument('--max_epochs', type=int, default=100000, help='maximum epoch to train')
    parser.add_argument('--grad_clip', type=float, default=1., help='gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='2e-3 from Ito, I used to use 5e-4')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1, help='value between 0~1, use this for scheduled sampling')
    # loading
    parser.add_argument('--init_from', type=str, default='', help='load parameters from...')
    parser.add_argument('--resume', type=int, default=0, help='1 for resume from saved epoch')
    # misc
    parser.add_argument('--print_every', type=int, default=10, help='')
    parser.add_argument('--save_every', type=int, default=10, help='')
    parser.add_argument('--save_dir', type=str, default='result', help='')
    parser.add_argument('-g', '--gpu', type=int, nargs='+', help='index of gpu machines to run')
    args = parser.parse_args()

    args.out_dir = './generated'
    args.frame_len_inMS = 50
    args.frame_shift_inMS = 12.5
    args.sample_rate = 16000
    args.n_fft = 2048
    args.num_recon_iters = 50
    model_name = args.init_from.split('/')[-1][:-3]
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(0)

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if args.gpu is None:
        args.use_gpu = False
        args.gpu = []
    else:
        args.use_gpu = True
        torch.cuda.manual_seed(0)
        torch.cuda.set_device(args.gpu[0])


    print('[*] Dataset: {}'.format(args.data))
    dataset = torch.utils.data.ConcatDataset([get_dataset(xx, args.dbroot) for xx in args.data])
    
    
    collate = collate_class(use_txt=0.)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, \
            shuffle=False, collate_fn=collate.fn, drop_last=True, **kwargs)

    # set misc options
    args.vocab_dict = {}
    args.spkr_lu = {}
    args.vocab_size = 0
    args.gender_num = 2
    args.age_num = 4
    args.emotion_num = 7

    for dataset_name in dataset.datasets:
        keys = dataset_name.vocab_dict.keys()
        for k in keys:
            if k not in args.vocab_dict.keys():
                args.vocab_dict[k] = len(args.vocab_dict)

        spkrs = dataset_name.spkr_lu.keys()
        for spkr in spkrs:
            if spkr not in args.spkr_lu:
                args.spkr_lu[spkr] = len(args.spkr_lu)

    args.vocab_size = len(args.vocab_dict)

    for dataset_name in dataset.datasets:
        dataset_name.set_vocab_dict(args.vocab_dict)
        dataset_name.set_spkr_lu(args.spkr_lu)

    # model define
    model = Tacotron(args)
    model_optim = optim.Adam(model.parameters(), args.learning_rate)
    scheduler = lr_scheduler.StepLR(model_optim, step_size=10)


    start = time.time()
    iter_per_epoch = len(dataset)//args.batch_size
    losses = []
    loss_total = 0
    start_epoch = 0
    it = 1

    if args.init_from:
        model, model_optim, start_epoch, losses  = load_model(
                finetune_model = model,
                pretrained_path = args.init_from,
                model_optim = model_optim,
                resume = args.resume,
                freeze_pretrained=args.freeze_pretrained,
                except_for = args.except_for
                )
        print('loaded checkpoint %s (epoch %d)' % (args.init_from, start_epoch))

    epoch = start_epoch
    model = model.eval()
    stft = STFT(filter_length=args.n_fft)
    if args.use_gpu:
        model = model.cuda()
        stft = stft.cuda()

    print('Start training... {} iter per epoch'.format(iter_per_epoch))
    for epoch in range(args.max_epochs):
        for it, this_batch in enumerate(loader):
            start_it = time.time()

            if args.use_gpu:
                for k, v in this_batch.items():
                    try:
                        this_batch[k] = Variable(v.cuda(), requires_grad=False)
                    except AttributeError:
                        pass

            model.reset_decoder_states()
            model.mask_decoder_states()
            model_optim.zero_grad()
            
            pred_mel, pred_lin, att = model(**this_batch)

            style_vec = model.style_vec.detach().cpu().numpy()
            context_vec = model.context_vec


            window_len = int(np.ceil(args.frame_len_inMS * args.sample_rate / 1000))
            hop_length = int(np.ceil(args.frame_shift_inMS * args.sample_rate / 1000))

            # write to file
            wave = spectrogram2wav_gpu(
                pred_lin.data,
                n_fft=args.n_fft,
                win_length=window_len,
                hop_length=hop_length,
                num_iters=args.num_recon_iters,
                stft=stft
            )
            wave = wave.cpu().numpy()


            for jj in range(args.batch_size):
                attentions = torch.cat(model.attn_weights, dim=-1)[jj]
                contents_filename = os.path.basename(this_batch['filename'][jj]['contents'])[:-4]
                style_filename = os.path.basename(this_batch['filename'][jj]['style'])[:-4]
                target_filename = os.path.basename(this_batch['filename'][jj]['target'])[:-4]
                contents_domain = this_batch['contents_domain']

                
                outpath1 = '%s/%s_%s_%s_%s_%s.wav' % (args.out_dir, model_name, contents_filename, style_filename, target_filename, contents_domain)
                
                shutil.copy2(this_batch['filename'][jj]['style'].replace('bin', 'wav').replace('.mel', '.wav'), 
                        outpath1[:-4] + '_style.wav')
                shutil.copy2(this_batch['filename'][jj]['contents'].replace('bin', 'wav').replace('.mel', '.wav'),
                        outpath1[:-4] + '_conents.wav')
                shutil.copy2(this_batch['filename'][jj]['target'].replace('bin', 'wav').replace('.mel', '.wav'),
                        outpath1[:-4] + '_target.wav')
                librosa.output.write_wav(outpath1, wave[jj], 16000)
                outpath2 = '%s/%s_%s_%s_%s_%s.png' % (args.out_dir, model_name, 
                        contents_filename, style_filename, target_filename, contents_domain)
                saveAttention(None, attentions, outpath2)
                outpath3 = '%s/%s_%s_%s_%s_%s.pt' % (args.out_dir, model_name, 
                        contents_filename, style_filename, target_filename, contents_domain)
                #torch.save({'style_vec': style_vec, 'mel': pred_mel.detach().cpu().numpy(), 
                #        'context_vec': context_vec, 'att': attentions}, outpath3)
                print(outpath2)




if __name__ == '__main__':
    main()

