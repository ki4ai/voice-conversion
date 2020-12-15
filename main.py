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

def main():
    parser = argparse.ArgumentParser(description='training script')
    # Mendatory arguments
    parser.add_argument('--data', type=str, nargs='+', default=['KETTS_male'], help='dataset type')
    parser.add_argument('--dbroot', type=str, default='/home/thkim/data/')
    parser.add_argument('-m', '--message', type=str, help='')

    # data load
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
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

    torch.manual_seed(0)

    kwargs = {'num_workers': 4, 'pin_memory': True}

    if args.gpu is None:
        args.use_gpu = False
        args.gpu = []
    else:
        args.use_gpu = True
        torch.cuda.manual_seed(0)
        torch.cuda.set_device(args.gpu[0])


    print('[*] Dataset: {}'.format(args.data))
    dataset = torch.utils.data.ConcatDataset([get_dataset(xx, args.dbroot) for xx in args.data])
    
    assert args.message is not None, "You have to set message"

    today = time.strftime('%y%m%d')
    savepath = join('result', '{}_{}'.format(today, args.message))
    if not exists(savepath):
        os.makedirs(savepath)
    elif args.message=='test':
        os.system("rm -rf {}/*".format(savepath))
    else:
        input("Path already exists, wish to continue?")
        os.system("rm -rf {}/*".format(savepath))
        os.system("rm -rf wandb/*{}*{}*".format(today, args.message))
    
    collate = collate_class(use_txt=0.5)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, \
            shuffle=True, collate_fn=collate.fn, drop_last=True, **kwargs)

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
    criterion_mel = nn.L1Loss()
    criterion_lin = nn.L1Loss()

    # wandb
    wandb.init(project='vctts', name=args.message)
    wandb.config['hostname'] = os.uname()[1]
    wandb.config.update(args)
    wandb.watch(model)
    with open(join(savepath, 'model.txt'), 'w') as f:
        f.write(str(model))
    torch.save(args, join(savepath, 'arg.pt'))
    os.system('cp *.py {}'.format(savepath))

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
    model = model.train()
    if args.use_gpu:
        model = model.cuda()
        criterion_mel = criterion_mel.cuda()
        criterion_lin = criterion_lin.cuda()


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

            for param_group in model_optim.param_groups:
                param_group['lr'] = decay_learning_rate(args.learning_rate, it, iter_per_epoch, start_epoch)

            model.reset_decoder_states()
            model.mask_decoder_states()
            model_optim.zero_grad()
            
            pred_mel, pred_lin, att = model(**this_batch)

            loss_mel = criterion_mel(pred_mel, this_batch['target_mel'])
            loss_linear = criterion_lin(pred_lin, this_batch['target_lin'])

            loss = loss_mel + loss_linear
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            model_optim.step()
            #scheduler.step()

            losses.append(loss.data.item())
            loss_total += loss.data.item()

            if it % args.print_every == 0:
                seen_it = iter_per_epoch * epoch  + it
                seen_samples = epoch * len(loader.dataset) + it  * args.batch_size
                seen_epochs = seen_samples / float(len(loader.dataset))
                print('epoch: {:2d} iter: {:3d} loss: {:5.3f} elapsed: {}  periter: {:4.2f}s'.format(
                    epoch, it, np.mean(losses[-args.print_every:]), asHMS(time.time()-start), time.time()-start_it))

                log_dict = {
                        'epoch/train': seen_epochs,
                        'mel_loss/train': loss_mel,
                        'lin_loss/train': loss_linear,
                        'total_loss/train': loss,
                        'att': wandb.Image(torch.cat(att, dim=-1)[0].detach().cpu().numpy().T, caption='Attention graph'),
                        }
                wandb.log(log_dict, step=seen_it)


        if epoch % args.save_every == 0:
            save_name = '{}/model_{}th.pt'.format(savepath, epoch)
            state = {
                'epoch': epoch,
                'args': args,
                'state_dict': model.state_dict(),
                'optimizer': model_optim.state_dict(),
                'plot_losses': losses
            }
            torch.save(state, save_name)
            print('model saved to', save_name)

if __name__ == '__main__':
    main()

