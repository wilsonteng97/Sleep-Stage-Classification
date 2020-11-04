from scipy.signal import butter, periodogram, sosfilt

# Util
def varname(variable):
    names = []
    for name in list(globals().keys()):
        string = f'id({name})'
        if id(variable) == eval(string):
            names.append(name)
    return names[0]

# band pass butter worth filter
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def bandpass_filter(data, lowcut, highcut, fs=100, order=14):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = sosfilt(sos, data)
    return filtered

def lowpass_filter(data, highcut, fs=100, order=14):
    sos = butter(order, highcut, btype='lowpass', fs=fs, output='sos')
    filtered = sosfilt(sos, data)
    return filtered