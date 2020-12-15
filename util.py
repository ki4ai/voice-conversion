import time, math, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

def savePlot(points, outpath):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(outpath)
    plt.close('all')

import time, math
def asHMS(s):
    d = int(s / 60 / 60 / 24)
    h = int(s / 60 / 60)
    m = int(s / 60)
    s = int(s % 60)
    return '{:02d}d{:02d}h{:02d}m{:02d}s'.format(d, h, m, s)

def create_mask(lengths):
    N = lengths.shape[0]
    L = lengths.max()
    mask = torch.zeros(N,L)
    for i, length in enumerate(lengths):
        mask[i,:length] += 1
    return mask

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def decompose_hangul(text):
    Start_Code, ChoSung, JungSung = 44032, 588, 28
    ChoSung_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JungSung_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                     'ㅣ']
    JongSung_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                     'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    line_dec = ""
    line = list(text.strip())

    for keyword in line:
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - Start_Code
            char1 = int(char_code / ChoSung)
            line_dec += ChoSung_LIST[char1]
            char2 = int((char_code - (ChoSung * char1)) / JungSung)
            line_dec += JungSung_LIST[char2]
            char3 = int((char_code - (ChoSung * char1) - (JungSung * char2)))
            line_dec += JongSung_LIST[char3]
        else:
            line_dec += keyword
    return line_dec

def stat(datapath):
    flist = glob.glob(datapath+'/*.mel')
    wavlen = []
    for fname in flist:
        mellin = pickle.load(open(fname, 'rb'))
        mel = mellin['mel']
        lin = mellin['lin']
        wavlen.append(len(mel))
        if len(mel) > 1000:
            print(fname)
    plt.hist(wavlen)
    plt.title('mean: {:.3f}, stdev: {:.3f}'.format(np.mean(wavlen), np.std(wavlen)))
    plt.savefig('stat_hist.png')

def decay_learning_rate(init_lr, it, iter_per_epoch, start_epoch=0):
    warmup_threshold = 4000
    step = start_epoch * iter_per_epoch + it + 1
    decayed_lr = init_lr * warmup_threshold ** 0.5 * min(step * warmup_threshold**-1.5, step**-0.5)
    return decayed_lr

def saveAttention(input_sentence, attentions, outpath):
    # Set up figure with colorbar
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots()
    cax = ax.matshow(attentions.cpu().numpy(), aspect='auto', origin='upper',cmap='gray')
    # fig.colorbar(cax)
    plt.ylabel('Encoder timestep', fontsize=18)
    plt.xlabel('Decoder timestep', fontsize=18)

    if input_sentence:
        plt.ylabel('Encoder timestep', fontsize=18)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close('all')


def spectrogram2wav(spectrogram, n_fft, win_length, hop_length, num_iters):
    '''
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    min_level_db = -100
    ref_level_db = 20

    spec = spectrogram.T
    # denormalize
    spec = (np.clip(spec, 0, 1) * - min_level_db) + min_level_db
    spec = spec + ref_level_db

    # Convert back to linear
    spec = np.power(10.0, spec * 0.05)

    return _griffin_lim(spec ** 1.5, n_fft, win_length, hop_length, num_iters)  # Reconstruct phase


def _griffin_lim(S, n_fft, win_length, hop_length, num_iters):
    # angles = np.exp(2j * np.pi * np.ones(S.shape))
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    for i in range(num_iters):
        if i > 0:
            angles = np.exp(1j * np.angle(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)))
        y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
    return y


def preprocess_text(dataset, text):
    if dataset == 'emotion':
        return decompose_hangul(text)
    elif dataset == 'librispeech':
        return text.lower()
    else:
        return text

def load_model(finetune_model, pretrained_path, model_optim, resume, freeze_pretrained=True, except_for=None):
    '''
        load pretrained model to finetun_model.
        finetune_model (nn.Module): model will be fine tuned.
        pretrained_path (str): path to pretrained model. state_dict should be indexed.
        freeze_pretrained (bool or list): freeze all pretrained weight or given list.
    '''
    checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    feed_weight = checkpoint['state_dict'].copy()

    if type(freeze_pretrained) == list:
        frozen_weights = freeze_pretrained
    elif freeze_pretrained:
        frozen_weights = feed_weight.keys()
        if except_for is not None:
            for except_key in except_for:
                frozen_weights = [xx for xx in frozen_weights if except_key not in xx]
    else:
        frozen_weights = []

    finetune_state_dict = finetune_model.state_dict()

    # If pretrained weights have different shape or non-exist, then compensate it.
    for k, v in checkpoint['state_dict'].items():
        if k in finetune_state_dict.keys():
            if checkpoint['state_dict'][k].shape != finetune_state_dict[k].shape:
                feed_weight[k] = finetune_state_dict[k]
                if k in frozen_weights:
                    frozen_weights.remove(k)
                resume = False
                print('[*] Weights in model-will-be-finetuned is not in pretrained model. Resume is not available')
            else:
                # k is in finetune network and shape is same.
                pass
        else:
            del feed_weight[k]
            if k in frozen_weights:
                frozen_weights.remove(k)

    # If new weights in finetune network is not in pretrained weight, 
    for k, v in finetune_state_dict.items():
        if k not in feed_weight.keys():
            feed_weight[k] = v
    finetune_model.load_state_dict(feed_weight)

    # freeze params
    for name, param in finetune_model.named_parameters():
        if name in frozen_weights:
            param.requires_grad = False
        print('{}\t{}\t{}'.format(name, param.shape, param.requires_grad))

    print('loaded checkpoint %s' % (pretrained_path)) 
    
    if resume:
        start_epoch = checkpoint['epoch']
        model_optim.load_state_dict(checkpoint['optimizer'])
        for state in model_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        plot_losses = checkpoint['plot_losses']
    else:
        start_epoch = 0
        plot_losses = []

    return finetune_model, model_optim, start_epoch, plot_losses
