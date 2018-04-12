### Google Audio Set classification with Keras and pytorch

## Description
This codebase is an implementation of [2, 3] with both Keras and pytorch. In [2, 3], attention neural networks are proposed and achieves a mean average precision (mAP) of 0.360. 
Audio Set is a large scale weakly labelled dataset containing over 2 million 10-second audio clips with 527 classes published by Google in 2017. 

## Download dataset
We convert the tensorflow type data to numpy data and stored in hdf5 file. The hdf5 data can be downloaded here https://drive.google.com/open?id=0B49XSFgf-0yVQk01eG92RHg4WTA

# Run with pytorch backend
./pytorch/keras/runme.sh

# Or, run with Keras backend
./pytorch/pytorch/runme.sh

## Results

## References
[1] Gemmeke, Jort F., et al. "Audio set: An ontology and human-labeled dataset for audio events." Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on. IEEE, 2017.
[2] Kong, Qiuqiang, et al. "Audio Set classification with attention model: A probabilistic perspective." arXiv preprint arXiv:1711.00927 (2017).
[3] Yu, Changsong, et al. "Multi-level Attention Model for Weakly Supervised Audio Classification." arXiv preprint arXiv:1803.02353 (2018).

## External links
The original implmentation of [3] is created by Changsong Yu https://github.com/ChangsongYu/Eusipco2018_Google_AudioSet
