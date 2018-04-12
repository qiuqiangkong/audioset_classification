# Google Audio Set classification with Keras and pytorch
Audio Set is a large scale weakly labelled dataset containing over 2 million 10-second audio clips with 527 classes published by Google in 2017. 

This codebase is an implementation of [2, 3], where attention neural networks are proposed for Audio Set classification and achieves a mean average precision (mAP) of 0.360. 

## Download dataset
We convert the tensorflow type data to numpy data and stored in hdf5 file. The size of the dataset is 2.3 G. The hdf5 data can be downloaded here https://drive.google.com/open?id=0B49XSFgf-0yVQk01eG92RHg4WTA

## Run
Users may optionaly choose Keras or pytorch as backend in runme.sh to run the code. 

**./runme.sh**

## Results
Mean average precision (mAP) of different models. 
<pre>
----------------------------------------------
Models                mAP     AUC     d-prime
----------------------------------------------
Google's baseline [1] 0.314   0.959   2.452
average pooling       0.300   0.964   2.536
max pooling           0.292   0.960   2.471
single_attention [2]  0.337   0.968   2.612
multi_attention [3]   <b>0.357</b>   <b>0.968</b>   <b>2.621</b>
----------------------------------------------
</pre>

Blue bars show the number of audio clips of classes. Red stems show the mAP of classes. 

![alt text](https://github.com/qiuqiangkong/audioset_classification/blob/master/appendixes/data_distribution.png)

## References
[1] Gemmeke, Jort F., et al. "Audio set: An ontology and human-labeled dataset for audio events." Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on. IEEE, 2017.

[2] Kong, Qiuqiang, et al. "Audio Set classification with attention model: A probabilistic perspective." arXiv preprint arXiv:1711.00927 (2017).

[3] Yu, Changsong, et al. "Multi-level Attention Model for Weakly Supervised Audio Classification." arXiv preprint arXiv:1803.02353 (2018).

## External links
The original implmentation of [3] is created by Changsong Yu https://github.com/ChangsongYu/Eusipco2018_Google_AudioSet
