"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import ipdb

C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load('3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
num_uttrs = 10
len_crop = 128

# Directory containing mel-spectrograms
rootDir = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    # fileList = list((Path(dirName)/speaker).glob("**/*.npy"))

    fileList = []
    for f in (Path(dirName)/speaker).glob("**/*.npy"):
        try:
            np.load(f)
        except:
            continue
        fileList.append(f)
    # Path(dirName)
    # make speaker embedding
    if len(fileList) < num_uttrs:
        continue
    
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    # import ipdb; ipdb.set_trace()
    for i in range(num_uttrs):
        #tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        tmp = np.load(fileList[idx_uttrs[i]])
        candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        # choose another utterance if the current one is too short
        ok = False
        while tmp.shape[0] <= len_crop:
            if len(candidates) == 0:
                break
            idx_alt = np.random.choice(candidates)
            #tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            tmp = np.load(fileList[idx_alt])
            candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
        else:
            ok = True

        if not ok:
            continue
        
        try:
            left = np.random.randint(0, tmp.shape[0]-len_crop)
        except:
            continue
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())     
    utterances.append(np.mean(embs, axis=0))
    
    # create file list
    for fileName in sorted(fileList):
        #utterances.append(os.path.join(speaker,fileName))
        utterances.append(fileName)
    speakers.append(utterances)
    
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)

