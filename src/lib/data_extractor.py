import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.special import digamma
from scipy.stats import kurtosis, skew

import lib.processing as p
from lib.config import Config
from lib.error import Error


class DataExtractor():
    """ 
    DataExtractor class contains functions that extract features from x and returns a DataFrame.
    """
    def __init__(self, x, y, config=Config(), collated_freq_bands_path=None):
        assert x.shape[1] == 3000
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

        if os.path.exists(collated_freq_bands_path):
            self.freq_band_dict = np.load(collated_freq_bands_path, allow_pickle=True)
            print("self.freq_band_dict loaded.")
        else:
            self.delta = p.lowpass_filter(x, 4, config.sampling_rate, 14)
            self.theta = p.bandpass_filter(x, 4, 8, config.sampling_rate, 14)
            self.alpha = p.bandpass_filter(x, 8, 12, config.sampling_rate, 14)
            self.sigma = p.bandpass_filter(x, 12, 15, config.sampling_rate, 14)
            self.beta1 = p.bandpass_filter(x, 15, 22, config.sampling_rate, 14)
            self.beta2 = p.bandpass_filter(x, 22, 30, config.sampling_rate, 14)
            self.gamma1 = p.bandpass_filter(x, 30, 40, config.sampling_rate, 14)
            self.gamma2 = p.bandpass_filter(x, 40, 49.5, config.sampling_rate, 14)

            self.freq_band_dict = {
                "delta": self.delta,
                "theta": self.theta, 
                "alpha": self.alpha, 
                "sigma": self.sigma, 
                "beta1": self.beta1, 
                "beta2": self.beta2, 
                "gamma1": self.gamma1, 
                "gamma2": self.gamma2
            }
            print("self.freq_band_dict created.")

            np.savez(collated_freq_bands_path, **self.freq_band_dict)

            del self.delta, self.theta, self.alpha, self.sigma, self.beta1, self.beta2, self.gamma1, self.gamma2
        print("freq bands processing done.")


    """ TODO : Function to generate pandas DataFrame with all the features stipulated in config. """
    def generateDF(self):
        # TODO：return dataframe
        df = pd.DataFrame()
        df['y'] = self.y
        df['avg'] = self.calculateMean(self.x)
        df['std'] = self.calculateStandardDeviation(self.x)
        df['skew'] = self.calculateSkew(self.x)
        df['kurtosis'] = self.calculateKurtosis(self.x)
        df['hjorth'] = self.calculateHjorth(self.x)
        df['mmd'] = self.calculateMMD(self.x)
        df['pfd'] = self.calculatePFD(self.x)
        df['kfd'] = self.calculateKFD(self.x)
        df['lrssv'] = self.calculateLRSSV(self.x)
        df['se'] = self.calculateSE(self.x)
        df['re'] = self.calculateRE(self.x)

        for i, band_name in enumerate(self.freq_band_dict.keys()):
            data = self.freq_band_dict[band_name]
            df[band_name + 'Avg'] = self.calculateMean(data)
            df[band_name + 'STD'] = self.calculateStandardDeviation(data)
            df[band_name + 'Skew'] = self.calculateSkew(data)
            df[band_name + 'Kurtosis'] = self.calculateKurtosis(data)
            df[band_name + 'Hjorth'] = self.calculateHjorth(data)
            df[band_name + 'MMD'] = self.calculateMMD(data)
            df[band_name + 'PFD'] = self.calculatePFD(data)
            df[band_name + 'KFD'] = self.calculateKFD(data)
            df[band_name + 'LRSSV'] = self.calculateLRSSV(data)
            df[band_name + 'SE'] = self.calculateSE(data)
            df[band_name + 'RE'] = self.calculateRE(data)
        return df
    
    def calculateMean(self, arr):
        return arr.mean(axis=1)

    def calculateStandardDeviation(self, arr):
        return arr.std(axis=1)

    def calculateSkew(self, arr):
        return skew(arr, axis=1)

    def calculateKurtosis(self, arr):
        return kurtosis(arr, axis=1)
    
    def calculateHjorth(self, arr):
        first_deriv = np.diff(arr, n=1, axis=1)
        second_deriv = np.diff(arr, n=2, axis=1)
        var_zero = np.mean(arr ** 2, axis=1)
        var_d1 = np.mean(first_deriv ** 2)
        var_d2 = np.mean(second_deriv ** 2)
        activity = var_zero
        morbidity = np.sqrt(var_d1 / var_zero)
        complexity = np.sqrt(var_d2 / var_d1) / morbidity
        return activity, morbidity, complexity

    def calculateMMD(self, arr):
        return np.max(arr, axis=1) - np.min(arr, axis=1)

    def calculatePFD(self, arr):
        N = np.full((arr.shape[0]), arr.shape[1])
        diff = np.diff(arr, n=1, axis=1)
        sign_change = (diff[:,1:-1] * diff[:,0:-2] < 0)
        M = sign_change.sum(axis=1)
        return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * M)))

    def calculateKFD(self, arr):
        distance = np.abs(np.diff(arr, n=1, axis=1))
        LL = distance.sum(axis=1)
        LL_normalized = np.log10(np.divide(LL, distance.mean(axis=1)))
        aux_d = arr- arr[:,0][:, None]
        d = np.max(np.abs(aux_d[:,1:]), axis=1)
        return np.divide(LL_normalized, np.add(LL_normalized, np.log10(np.divide(d, LL))))

    # TODO: convert to 2D
    def calculateHurstExponent(self, a):
        #Hurst Exponent
        def getLags(arr_len):
            if arr_len == 3000:
                return [3000, 1500, 750, 250, 125, 60, 30, 15, 8, 4, 2]
            else:
                return [2**x for x in range(int(np.log2(arr_len)+1))][::-1][:-1]

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

    # Log Root Sum of Sequential Variations
    # Measure the sequential variations
    def calculateLRSSV(self, arr):
        diff = np.diff(arr, n=1, axis=1)
        return np.log10(np.sqrt(np.sum(diff**2, axis=1)))

    # Normalized Spectral Entropy, https://raphaelvallat.com/entropy/build/html/_modules/entropy/entropy.html
    def calculateSE(self, arr):
        def calculateSE1D(a):
            _, psd = periodogram(a, 100)  #fft transform
            psd_norm = np.divide(psd, psd.sum()) #power spectral density, measure of signal's power content versus frequency
            se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
            se /= np.log2(psd_norm.size)
            return se
        return np.apply_along_axis(calculateSE1D, 1, arr)

    # Rényi entropy (RE) a measure of the entropy of the distribution P = (p1, p2,..., pn)
    def calculateRE(self, arr, m=2):
        return 1/(1-m) * np.log2(np.sum(arr**m, axis=1))

    # Kraskov entropy (KE) an estimate for Shannon entropy using N samples of an m-dimensional random vector x
    # very confused???
    # TODO: Conver to 2D, using lambda takes a long time
    def calculateKE1D(self, a):
        N = len(a)
        k = int(np.sqrt(N)) # sqrt of epoch len (3000)
        sum_ = 0
        for i in range(len(a)):
            kth_nearest = sorted(sorted(a, key = lambda n: abs(a[i]-n))[:k])[-1]
            ri = np.abs(kth_nearest - a[i])
            if ri!=0:
                sum_ += np.log(2*ri)
        return digamma(N) - digamma(k) + 1/N * sum_
