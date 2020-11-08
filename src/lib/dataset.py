import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.pyplot import figure

from lib.config import Config
from lib.data_extractor import DataExtractor


class Dataset():
    """ 
    DataSet class to interface with collated normalized EEG data with npz format.
    
    Terms used :
    Block/Chunk -> Refers to the contiguous set of time-series x values (3000 in this situation) that has a corresponding class.
    """
    
    def __init__(self, collated_data_path, config=Config()):
        self.data_path = collated_data_path
        self.data_dir = os.path.dirname(self.data_path)
        self.config = config
        
        data = np.load(self.data_path, allow_pickle=True)
        self.file_headers = data.files
        self.x = data['x']
        self.y = data['y']
        
        self.xlen = len(self.x)
        self.ylen = len(self.y)
        self.num_blocks = self.xlen
    
        self.sampling_rate = config.sampling_rate
        self.x_interval_seconds = config.x_interval_seconds
        self.num_data_points = self.xlen * self.x_interval_seconds
        
        self.compressed_y_str = None
        self.compressed_y_ls = None
        self.awakeBlks = None
        self.classCount = None
        self.awakeBlksCount = None
        self.unawakeBlksCount = None
        del data
    
    """ TODO : Get DataFrame of extracted features. """
    def getDF(self, save_pth):
        collated_freq_band_path = os.path.join(self.data_dir, "collated_freq_bands.npz")
        data_extractor = DataExtractor(self.x, self.y, self.config, collated_freq_bands_path=collated_freq_band_path)
        df = data_extractor.generateDF()
        df.to_csv(save_pth, index=False, header=True)
        df = pd.read_csv(save_pth)
        return df

    """ Get X value of data. Shape will be (N, 3000). """
    def getX(self):
        return self.x
    
    """ Get y value of data. Shape will be (N). """
    def getY(self):
        return self.y
    
    """ Get a compressed string representation of contiguous blocks with the same class. """
    def getCompressedYString(self):
        if self.compressed_y_str is not None: return self.compressed_y_str
        y = self.y
        prev = y[0]
        count = 1
        s = ""
        for i, curr in enumerate(y[1:]):
            if curr == prev:
                count += 1
                continue
            else:
                s += self.config.label_dict[prev] + str(count) + "|"
                count = 1
            prev = curr
        s += self.config.label_dict[prev] + str(count) + "|"
        self.compressed_y_str = s
        return s
    
    """ Get an array of tuples with format (class, length, start_idx, end_idx) of the contiguous blocks with the same class """
    def getCompressedY(self):
        if self.compressed_y_ls is not None: return self.compressed_y_ls
        trimmed_y = self.y
        prev = trimmed_y[0]
        count = 1
        start_idx = 0
        compressed_y_ls = []
        for idx, curr in enumerate(trimmed_y[1:]):
            idx_w_offset  = idx
            if curr == prev:
                count += 1
                continue
            else:
                compressed_y_ls.append((self.config.label_dict[prev], count, start_idx, idx_w_offset))
                count = 1
                start_idx = idx_w_offset + 1
            prev = curr
        compressed_y_ls.append((self.config.label_dict[prev], count, start_idx, idx_w_offset + 1))
        self.compressed_y_ls = compressed_y_ls
        return compressed_y_ls
    
    """ Get the contiguous blocks where the class is awake in tuple with format (class, length, start_idx, end_idx) """
    def getAwakeBlocksCount(self):
        if self.awakeBlks is not None: return self.awakeBlks
        awakeBlks = []
        for idx, curr in enumerate(self.getCompressedY()):
            if curr[0] == self.config.label_dict[0]:
                awakeBlks.append(curr)
        self.awakeBlks = awakeBlks
        self.awakeBlksCount = len(awakeBlks)
        self.unawakeBlksCount = self.num_blocks - self.awakeBlksCount
        return awakeBlks
    
    """ Get the contiguous blocks of specified 'class_' argument in tuple with format (class, length, start_idx, end_idx) """
    def getBlocksCount(self, class_):
        if self.awakeBlks is not None: return self.awakeBlks
        blks = []
        for idx, curr in enumerate(self.getCompressedY()):
            if curr[0] == self.config.label_dict[class_]:
                blks.append(curr)
        return blks

    """ Get class distribution count and percentages """
    def getClassCount(self):
        if self.classCount is not None: return self.classCount
        unique, counts = np.unique(self.y, return_counts=True)
        total = sum(counts)
        count_tuple = tuple(zip(counts, counts / total * 100))
        ret = dict(zip(unique, count_tuple))
        ret["total"] = total
        self.classCount = ret
        return self.classCount
    
    """ Plot class distribution as a Histogram """
    def getClassHist(self):
        h = {key:self.getClassCount()[key] for key in list(range(0,5))}
        counts, percentages = list(zip(*h.values()))
        plt.figure(figsize=self.config.default_fig_size)
        plt.bar([*h.keys()], counts)
        plt.title('Sleep Stage Count Histogram')
        plt.xlabel('Sleep Stage')
        plt.ylabel('Count')
        plt.show()
    
    """ TODO: Add function to plot sleep stages similar to generateFullSleepStagePlot() but with specified indexes. """
    def getSleepStagePlot(self, start_idx, end_idx):
        print("To-Do")
    
    """ Plot the classes and x values in the time-domain for the entire dataset. """
    def generateFullSleepStagePlot(self):
        print("Plot generation will take some time...")
        print("Please use getSleepStagePlot() to generate desired indexes.")
        fig, (ax1, ax2) = plt.subplots(2, figsize=(18,10))
        fig.suptitle("Sleep Stage Plots")

        plt.sca(ax1)
        plt.xticks(np.arange(0, len(self.y)+1, 20))
        plt.yticks(np.arange(0, 5))
        ax1.plot(range(len(self.y)), self.y)

        ax2.plot(range(self.xlen * self.x_interval_seconds), self.x.reshape(-1))
        plt.show()
    
    """ Plot a specified block """
    def getBlockPlot(self, blk_no):
        assert blk_no >= 0, "negative blk_no {blk_no} is invalid"
        assert blk_no < self.xlen, f"Invalid blk_no, max blk_no is {self.num_blocks-1}"
        fig = plt.figure(figsize=(18,5))
        plt.title(f"Block {blk_no} | Sleep Stage : {self.config.stage_name[self.y[blk_no]]}")
        plt.plot(range(self.x_interval_seconds), self.x[blk_no])


if __name__ == "__main__":
    data_path = r"C:\Users\wilso\Desktop\Sleep-Stage-Classification\src\data_norm\collated.npz"
    dataset = Dataset(data_path)
    pprint(dataset.getClassCount())
