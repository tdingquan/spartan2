
from torch.utils.data import DataLoader, Dataset
import torch
from gensim import similarities
import os
import logging
import numpy as np
import pandas as pd


class NeighborSampler():
    def __init__(self, chip_name, stage=0, x_data=None):
        self.index = None
        self.chip_name = chip_name
        self.stage = stage
        self.build(x_data)

    def build(self, x_data=np.array([])):
        corpus = []
        for i in range(x_data.shape[0]):
            gensim_format_vec = []
            for j in range(x_data.shape[1]):
                gensim_format_vec.append((j, x_data[i][j]))
            corpus.append(gensim_format_vec)
        logging.info("#sample to build index: %s" % x_data.shape[0])
        self.get_index(corpus, x_data.shape[1])

    def get_index(self, corpus=None, n_feature=None):
        self.index = similarities.Similarity("%s_%s_neighbor.index.tmp" % (self.chip_name, self.stage), corpus, num_features=n_feature)
        return self.index

    def get_topk(self, vec, k):
        gensim_format_vec = []
        for i in range(len(vec)):
            gensim_format_vec.append((i, vec[i]))
        sims = self.index[gensim_format_vec]

        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1])
        top_k = sim_sort[0:k]
        return top_k


def build_sampled_coexpression_matrix(x_data, filename, k=1000):
    logging.info("Co-expression matrix building process starts.")
    logging.info("data shape: %s %s" % (x_data.shape[0], x_data.shape[1]))
    n_gene = x_data.shape[1]

    neighbor_sampler = NeighborSampler(filename, x_data=x_data)
    with open("matrix.txt", "w") as f:
        for i in range(n_gene):
            list_neighbor = neighbor_sampler.get_topk(x_data[i], k)
            list_neighbor.extend(neighbor_sampler.get_topk(x_data[i], k))
            list_neighbor = list(set(list_neighbor))
            logging.info("sample %s's topk neighbor: %s" % (i, list_neighbor))
            for j, value in list_neighbor:
                pearson_value = pearson(x_data[i], x_data[j])
                f.write("%s\t%s\t%s\n" % (i, j, pearson_value))
            f.flush()
            os.fsync(f.fileno())


def pearson(X, Y):
    return np.corrcoef(X, Y)[0][1]


if __name__ == "__main__":
    x_data = pd.read_csv("F:\gene2\spartan2\live-tutorials\inputData\example.csv", sep=",", header=None)
    build_sampled_coexpression_matrix(x_data, "out")


class ToyDataSet(Dataset):
    def __init__(self, data, y):
        self.data = data
        self.y = y

    def __getitem__(self, index):
        item = self.data[index]
        sample_X = torch.Tensor(item)
        sample_y = torch.tensor(self.y[index])
        return (sample_X, sample_y)

    def __len__(self):
        return len(self.data)


def sine_wave(seq_length=64, num_samples=5000, num_signals=1,
              freq_low=1, freq_high=5, amplitude_low=0.1, amplitude_high=0.9, **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            f = np.random.uniform(low=freq_high, high=freq_low)     # frequency
            A = np.random.uniform(low=amplitude_high, high=amplitude_low)        # amplitude
            # offset
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            signals.append(A*np.sin(2*np.pi*f*ix/float(seq_length) + offset))
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples)
    return samples


def load_sine_data(config=None):
    if config is None:
        raise Exception("Config is None")
    train_low_A_low_F = sine_wave(64, 50000, 1, 1, 5, 0.1, 0.5)
    train_low_A_high_F = sine_wave(64, 50000, 1, 5, 10, 0.1, 0.5)
    train_high_A_low_F = sine_wave(64, 50000, 1, 1, 5, 0.5, 0.9)
    train_high_A_high_F = sine_wave(64, 50000, 1, 5, 10, 0.5, 0.9)
    X_train_list = [train_low_A_low_F, train_low_A_high_F, train_high_A_low_F, train_high_A_high_F]

    X_train = X_train_list[config.train_index]
    y_train = np.zeros(X_train.shape[0])

    # test_low_A_low_F = sine_wave(64, 5000, 1, 1, 5, 0.1, 0.5)
    # test_low_A_high_F = sine_wave(64, 5000, 1, 5, 10, 0.1, 0.5)
    # test_high_A_low_F = sine_wave(64, 5000, 1, 1, 5, 0.5, 0.9)
    # test_high_A_high_F = sine_wave(64, 5000, 1, 5, 10, 0.5, 0.9)

    # X_test=np.concatenate([test_low_A_low_F,test_high_A_high_F,test_low_A_high_F,test_high_A_low_F])
    # y_test=np.concatenate([np.ones(test_low_A_low_F.shape[0]),
    #                       np.ones(test_high_A_high_F.shape[0]),
    #                       np.ones(test_low_A_high_F.shape[0]),
    #                       np.zeros(test_high_A_low_F.shape[0])])

    # X_test = np.concatenate([
    #                          # test_low_A_low_F
    #                          # test_high_A_high_F,
    #                          # test_low_A_high_F,
    #                          test_high_A_low_F
    #                          ])
    # y_test = np.concatenate([
    #                          # np.ones(test_low_A_low_F.shape[0])
    #                          # np.ones(test_high_A_high_F.shape[0]),
    #                          # np.ones(test_low_A_high_F.shape[0]),
    #                          np.ones(test_high_A_low_F.shape[0])
    #                          ])

    # for draw graph

    # for test A
    X_test_1 = sine_wave(64, 100, 1, 1, 5, 0.1, 0.25)
    X_test_2 = sine_wave(64, 100, 1, 1, 5, 0.25, 0.4)
    X_test_3 = sine_wave(64, 100, 1, 1, 5, 0.4, 0.55)
    X_test_4 = sine_wave(64, 100, 1, 1, 5, 0.55, 0.7)
    X_test_5 = sine_wave(64, 100, 1, 1, 5, 0.7, 0.85)
    X_test_6 = sine_wave(64, 100, 1, 1, 5, 0.85, 1.0)

    # for test F
    X_test_7 = sine_wave(64, 100, 1, 1, 2.5, 0.5, 0.9)
    X_test_8 = sine_wave(64, 100, 1, 2.5, 4, 0.5, 0.9)
    X_test_9 = sine_wave(64, 100, 1, 4, 5.5, 0.5, 0.9)
    X_test_10 = sine_wave(64, 100, 1, 5.5, 7, 0.5, 0.9)
    X_test_11 = sine_wave(64, 100, 1, 7, 8.5, 0.5, 0.9)
    X_test_12 = sine_wave(64, 100, 1, 8.5, 10, 0.5, 0.9)

    X_test_list = [
        X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6,
        X_test_7, X_test_8, X_test_9, X_test_10, X_test_11, X_test_12
    ]

    X_test = X_test_list[config.test_index]

    y_test = np.ones(X_test.shape[0])

    train_dataset = ToyDataSet(X_train, y_train)
    val_dataset = ToyDataSet(X_test, y_test)
    test_dataset = ToyDataSet(X_test, y_test)

    print("train size:{}".format(len(train_dataset)))
    print("val size:{}".format(len(val_dataset)))
    print("test size:{}".format(len(test_dataset)))

    data_loader = {
        "train": DataLoader(
            dataset=train_dataset,  # torch TensorDataset format
            batch_size=config.batch_size,  # mini batch size
            shuffle=True,
            num_workers=8,
            drop_last=True),
        "val": DataLoader(
            dataset=val_dataset,  # torch TensorDataset format
            batch_size=config.batch_size,  # mini batch size
            shuffle=False,
            num_workers=8,
            drop_last=False),
        "test": DataLoader(
            dataset=test_dataset,  # torch TensorDataset format
            batch_size=config.batch_size,  # mini batch size
            shuffle=False,
            num_workers=8,
            drop_last=False),

    }

    return data_loader
