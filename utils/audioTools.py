import scipy.io.wavfile as wav
import soundfile


def load_wavfile(wavfile):
    """
    Read a wav file using scipy.io.wavfile
    """
    if wavfile.endswith('.wav'):
        rate, sig = wav.read(wavfile)
    elif wavfile.endswith('.flac'):
        sig, rate = soundfile.read(wavfile, dtype='int16')
    else:
        raise IOError('NOT support file type or not a filename: {}'.format(wavfile))
    # data_name = os.path.splitext(os.path.basename(wavfile))[0]
    return rate, sig
