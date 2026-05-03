import torch
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import torch.nn.functional as F

import json
import os
import subprocess
import transformers
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
import re
from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import CharacterTokenizer
import argparse

# helper 2
def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict."""

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
    #if 'backbone' in key:
        # the state dicts differ by one prefix, '.model', so we add that  
        # key_loaded = 'model.' + key
        # breakpoint()
        # need to add an extra ".layer" in key
        if key.startswith('backbone.'):
            key_loaded = 'model.' + key
        elif key == 'head.output_transform.weight':
            #CS key_loaded = 'model.lm_head.weight'  #model.lm_head.weight is for generation of 16 kinds of tokens.
            key_loaded = 'decoder.0.output_transform.weight'
        elif key == 'head.output_transform.bias':
            key_loaded = 'decoder.0.output_transform.bias'
        try:
            scratch_dict[key] = pretrained_dict[key_loaded]
        except:
            raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated


    return scratch_dict


class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name,
                        download=False,
                        config=None,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                      ):
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and download == False:
            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        else:
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'

            subprocess.run(f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(**config, use_head=use_head, n_classes=n_classes)  # the new model format
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            map_location=torch.device(device)
        )

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model

# you only need to select which model to use here, we'll do the rest!
pretrained_model_name = 'hyenadna-medium-160k-seqlen-FTed-val_DMD45'

max_length = 500 # CS this is to be consisitent with what was sed when training.
print("max_length")
print(max_length)

# data settings:
use_padding = True
rc_aug = False  # reverse complement augmentation
add_eos = False  # add end of sentence token

# we need these for the decoder head, if using
#use_head = False
use_head = True # CS changed here.
n_classes = 2  # not used for embeddings only

# you can override with your own backbone config here if you want,
# otherwise we'll load the HF one in None
backbone_cfg = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# instantiate the model (pretrained here)
if pretrained_model_name in ['hyenadna-tiny-1k-seqlen',
                             'hyenadna-small-32k-seqlen',
                             'hyenadna-medium-160k-seqlen',
                             'hyenadna-medium-160k-seqlen-FTed',
                             'hyenadna-medium-160k-seqlen-FTed-val_DMD45',
                             'hyenadna-medium-450k-seqlen',
                             'hyenadna-large-1m-seqlen']:
    # use the pretrained Huggingface wrapper instead
    model = HyenaDNAPreTrainedModel.from_pretrained(
        './checkpoints',
        pretrained_model_name,
        download=False,
        config=backbone_cfg,
        device=device,
        use_head=use_head,
        n_classes=n_classes,
    )
# from scratch
elif pretrained_model_name is None:
    model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)

# create tokenizer
tokenizer = CharacterTokenizer(
    characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
    model_max_length=max_length + 2,  # to account for special tokens, like EOS
    add_special_tokens=False,  # we handle special tokens elsewhere
    padding_side='left', # since HyenaDNA is causal, we pad on the left
)



# 1. CSV 読み込み
# CSVは "sequence" と "label" の2列を想定
#df = pd.read_csv('COL7A1_73.csv')
df = pd.read_csv('COL7A1_80.csv')
#df = pd.read_csv('INPUTCSV')

# モデル・トークナイザを準備済みとして
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

all_probs = []
all_preds = []
all_labels = []

for idx, row in df.iterrows():
    sequence = row['sequence']
    label = row['label']
    
    # 文字列→tokenize
    tok_seq = tokenizer(sequence)
    tok_seq = tok_seq["input_ids"]

    if len(tok_seq) > 0 and tok_seq[0] == 0:
        tok_seq = tok_seq[1:]
    # ③ 末尾の“1”を削る
    if len(tok_seq) > 0 and tok_seq[-1] == 1:
        tok_seq = tok_seq[:-1]
    # ④ 足りない分を左側に4でパディング
    pad_len = max_length - len(tok_seq)
    if pad_len > 0:
        tok_seq = [4] * pad_len + tok_seq
    else:
        # 万一長すぎる場合は右を切る
        tok_seq = tok_seq[-self.cfg.max_length :]



    
    # Tensorに変換
    tok_seq = torch.LongTensor(tok_seq).unsqueeze(0).to(device)  # batch dim

    with torch.inference_mode():
        logits = model(tok_seq)  # logitsを直接出力する場合
    #CS
    print(logits)
    
    # logits → probability
    probs = F.softmax(logits, dim=-1)
    prob_positive = probs[0,1].item()  # クラス1の確率
    
    pred = int(torch.argmax(probs, dim=-1))
    
    all_probs.append(prob_positive)
    all_preds.append(pred)
    all_labels.append(label)

# 結果をDataFrameにまとめる
df['probability'] = all_probs
df['prediction'] = all_preds

# 2. 結果をCSVに保存
#df.to_csv('predict_COL7A1_73.csv', index=False)
df.to_csv('predict_COL7A1_80.csv', index=False)

# 3. 評価指標計算
acc = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
mcc = matthews_corrcoef(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
# TN is C00 , FP is C01
# FN is C10 , TP is C11

# 4. 標準出力
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"MCC: {mcc:.4f}")
print("Confusion Matrix:")
print(cm)

