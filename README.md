This is the code for voice conversion



## Prerequisite

Install required packages 

```shell
pip3 install -r requirements.txt
```



## Inference

Few samples and pretraiend model for VC are provided, so you can try with below command.

Samples contain 20 types of sentences and 7 emotions, 140 utterances in total.

```shell
python3 generate.py --init_from pretrained_model.pt --gpu <gpu_id> --out_dir <out_dir>
```

Below is an example of generated wav.

It means the model takes contents of `(fear, 20th contents)` and style of `(anger, 2nd contents)` to make `(anger, 20th contents)`.

```shell
pretrained_model_fea_00020_ang_00002_ang_00020_input_mel.wav
```


## Training

```shell
# remove silence within wav files
python3 trimmer.py --in_dir <in_dir> --out_dir <out_dir>

# Extract mel/lin spectrogram and dictionary of characters/phonemes
python3 preprocess.py --txt_dir <txt_dir> --wav_dir <wav_dir> --bin_dir <bin_dir>

# train the model, --use_txt will control vc path or tts path
python3 main.py -m <message> -g <gpu_id> --use_txt <0~1, higher value means y_t batch is more sampled>
```

