import ipdb
from tqdm import tqdm
import shutil
from pathlib import Path

kids_path = Path('../data/kids_speech')

if __name__ == "__main__":
    wavDir = kids_path / 'wav'
    targetDir = kids_path / 'wavs'

    wavs = [f.stem[:4] for f in wavDir.glob('*.wav')]
    speakers = list(set(wavs))
    for speaker in tqdm(speakers):
        targetSubDir = targetDir / speaker
        targetSubDir.mkdir(parents=True, exist_ok=True)
        for src_file in wavDir.glob('{}*.wav'.format(speaker)):
            shutil.copy(src_file, targetSubDir / src_file.name)
