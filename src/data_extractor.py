import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.special import digamma

from config import Config
from error import Error


class DataExtractor():
    """ 
    DataExtractor class contains functions that extract features from x and returns a DataFrame.
    """
    def __init__(self, x, y, config=Config()):
        assert x.shape[1] == 3000
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y


    """ TODO : Function to generate pandas DataFrame with all the features stipulated in config. """
    def generateDF(self):
        # TODO：return dataframe
        print("TODO")


    def calculateMean(self):
        return self.x.mean(axis=1)


    def calculateStandardDeviation(self):
        return self.x.std(axis=1)
    

    def calculateHjorth(self):
        first_deriv = np.diff(self.x, n=1, axis=1)
        second_deriv = np.diff(self.x, n=2, axis=1)
        var_zero = np.mean(self.x ** 2, axis=1)
        var_d1 = np.mean(first_deriv ** 2)
        var_d2 = np.mean(second_deriv ** 2)
        activity = var_zero
        morbidity = np.sqrt(var_d1 / var_zero)
        complexity = np.sqrt(var_d2 / var_d1) / morbidity
        return activity, morbidity, complexity


    def calculateMMD(self):
        return np.max(self.x, axis=1) - np.min(self.x, axis=1)


    def calculatePFD(self):
        N = np.full((self.x.shape[0]), self.x.shape[1])
        diff = np.diff(self.x, n=1, axis=1)
        sign_change = (diff[:,1:-1] * diff[:,0:-2] < 0)
        M = sign_change.sum(axis=1)
        return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * M)))


    # TODO: WIP
    def calculateKFD(self, a):
        distance = np.abs(np.ediff1d(a))
        LL = distance.sum() #'line length'
        LL_normalized = np.log10(np.divide(LL, distance.mean())) #original paper uses constant hyperparameter M=1
        aux_d = a - a[0]
        d = np.max(np.abs(aux_d[1:]))
        return np.divide(LL_normalized, np.add(LL_normalized, np.log10(np.divide(d, LL))))


    # TODO: WIP
    def calculateHurstExponent(self, a):
        alen = len(a)
        if alen < 100:
            raise Error(f"Not enough datapoints to calculate Hurst Exp | Only {alen} of the required 100.")
        lags = getLags(alen)
        
        ln_lag_ls = []
        ln_Rs_ls = []
        for i, lag in enumerate(lags):
            chunk = a.reshape(-1, lag)
            
            chunk_mean = np.mean(chunk, axis=1, dtype=np.float32, keepdims=True)
            chunk_std = np.std(chunk, axis=1, dtype=np.float32, ddof=1) # Sample Standard deviation
            chunk_mean_centered = np.subtract(chunk, chunk_mean)
            cum_sum = np.cumsum(chunk_mean_centered, axis=1, dtype=np.float32)
            
            R = np.max(cum_sum, axis=1) - np.min(cum_sum, axis=1)
            Rs = np.divide(R, chunk_std)
            avg_Rs = np.mean(Rs)
            
            ln_lag_ls.append(np.log(lag))
            ln_Rs_ls.append(np.log(avg_Rs))
    #     print("x\n", ln_lag_ls)
    #     print("y\n", ln_Rs_ls)
        _ = plt.plot(ln_lag_ls, ln_Rs_ls, '.')
        plt.show()
        m, c = np.polyfit(ln_lag_ls,ln_Rs_ls,1)
        return m


    #Log Root Sum of Sequential Variations
    #measure the sequential variations
    def calculateLRSSV(self):
        diff = np.diff(self.x, n=1, axis=1)
        return np.log10(np.sqrt(np.sum(diff**2, axis=1)))


    #Normalized Spectral Entropy, https://raphaelvallat.com/entropy/build/html/_modules/entropy/entropy.html
    # TODO: WIP
    def calculateSE(self, a):
        _, psd = periodogram(a, 100)  #fft transform
        psd_norm = np.divide(psd, psd.sum()) #power spectral density, measure of signal's power content versus frequency
        se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
        se /= np.log2(psd_norm.size)
        return se


    #Rényi entropy (RE) a measure of the entropy of the distribution P = (p1, p2,..., pn)
    def calculateRE(self, m=2):
        return 1/(1-m) * np.log2(np.sum(self.x**m, axis=1))

 
    #Kraskov entropy (KE) an estimate for Shannon entropy using N samples of an m-dimensional random vector x
    #very confused???
    # TODO: WIP
    def calculateKE(self, a):
        N = len(a)
        k = int(np.sqrt(N)) # sqrt of epoch len (3000)
        sum_ = 0
        for i in range(len(a)):
            kth_nearest = sorted(sorted(a, key = lambda n: abs(a[i]-n))[:k])[-1]
            ri = np.abs(kth_nearest - a[i])
            if ri!=0:
                sum_ += np.log(2*ri)
        return digamma(N) - digamma(k) + 1/N * sum_
