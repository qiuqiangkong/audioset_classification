import os
import soundfile
import librosa

import tensorflow as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
slim = tf.contrib.slim

    
def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs
    

### Feature extraction. 
def extract_audioset_embedding():
    """Extract log mel spectrogram features. 
    """
    
    # Arguments & parameters
    mel_bins = vggish_params.NUM_BANDS
    sample_rate = vggish_params.SAMPLE_RATE
    input_len = vggish_params.NUM_FRAMES
    embedding_size = vggish_params.EMBEDDING_SIZE
    
    '''You may modify the EXAMPLE_HOP_SECONDS in vggish_params.py to change the 
    hop size. '''

    # Paths
    audio_path = 'appendixes/01.wav'
    checkpoint_path = os.path.join('vggish_model.ckpt')
    pcm_params_path = os.path.join('vggish_pca_params.npz')
    
    if not os.path.isfile(checkpoint_path):
        raise Exception('Please download vggish_model.ckpt from '
            'https://storage.googleapis.com/audioset/vggish_model.ckpt '
            'and put it in the root of this codebase. ')
        
    if not os.path.isfile(pcm_params_path):
        raise Exception('Please download pcm_params_path from '
        'https://storage.googleapis.com/audioset/vggish_pca_params.npz '
        'and put it in the root of this codebase. ')
    
    # Load model
    sess = tf.Session()
    
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
    
    pproc = vggish_postprocess.Postprocessor(pcm_params_path)

    # Read audio
    (audio, _) = read_audio(audio_path, target_fs=sample_rate)
    
    # Extract log mel feature
    logmel = vggish_input.waveform_to_examples(audio, sample_rate)

    # Extract embedding feature
    [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: logmel})
    
    # PCA
    postprocessed_batch = pproc.postprocess(embedding_batch)
    
    print('Audio length: {}'.format(len(audio)))
    print('Log mel shape: {}'.format(logmel.shape))
    print('Embedding feature shape: {}'.format(postprocessed_batch.shape))
        

if __name__ == '__main__':
    
    extract_audioset_embedding()