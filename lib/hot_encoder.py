from sklearn import preprocessing
import numpy as np
from sets import Set

class HotOneEncoder(object):
    def __init__(self, words):
        word_set = Set(words)
        self.words_to_ints = {}
        self.length = len(word_set)
        count = 0
        for word in word_set:
            self.words_to_ints[word] = count
            count += 1

    def encoding_for_word(self, word):
        arr = np.zeros(self.length)
        if not word in self.words_to_ints:
            return arr
        num = self.words_to_ints[word]
        arr[num] = 1
        return arr

    def encode(self, words):
        return np.array(map(lambda w: self.encoding_for_word(w), words))

class Encoder(object):
    def __init__(self):
        self.hot_one_encoders = {}

    def encode_words(self, idx, strings):
        return self.hot_one_encoders[idx].encode(strings)

    def train_encode_strings(self, idx, strings):
        self.hot_one_encoders[idx] = HotOneEncoder(strings)

    def encode_for_indices(self, indices, mat, train=False):
        transformed = {}
        for i in indices:
            if train:
                self.train_encode_strings(i, mat[i].tolist()[0])
            transformed[i] = self.encode_words(i, mat[i].tolist()[0])
        return transformed

    def encode_matrix(self, indices, matrix, train=False):
        transpose = matrix.T
        encoded = self.encode_for_indices(indices, transpose, train)
        to_concat = []
        for i in range(matrix.shape[1]):
            if i in indices:
                to_concat.append(encoded[i])
            else:
                to_concat.append(transpose[i].T)
        return np.concatenate(to_concat, axis=1)
