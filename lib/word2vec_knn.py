from lib.entropy.parser import Parser
from sklearn.neighbors import KNeighborsClassifier
import gensim
import numpy as np

class Word2VecKNN(object):
    types_to_ints = { 'LOC': 0, 'GPE': 1, 'PER': 2, 'ORG': 3 }

    def __init__(self, num_neighbors=5, hidden_size=100, power=2.0, resample=False):
        self.power = power
        self.num_neighbors = num_neighbors
        self.hidden_size = hidden_size

    def produce_raw_x_and_y(self, file_path):
        parsed = Parser('../' + file_path)
        sentences = filter(lambda x: x.contains_entity(), parsed.sentences)
        X = []
        Y = []
        entity_to_tag = {}

        for sentence in sentences:
            X.append(sentence.full_sentence())
            Y.append(sentence.entity)
            entity_to_tag[sentence.entity] = sentence.tag
            
        return X, Y, entity_to_tag

    def produce_word2vec_representation(self, X):
        return map(lambda x: self.word2vec[x], X)

    def weight_function(self, distances):
        return map(lambda x: 1.0/(pow(x, self.power)), distances)

    def train(self, file_path):
        X, Y, entity_to_tag = self.produce_raw_x_and_y(file_path)
        self.word2vec = gensim.models.Word2Vec(X, size=self.hidden_size, min_count=1, workers=4)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=self.num_neighbors, weights=self.weight_function)
        X = self.produce_word2vec_representation(X)
        Y = map(lambda y: Word2VecKNN.types_to_ints[entity_to_tag[y]], Y)
        print X
        print Y
        self.knn_classifier.fit(X, Y)
        return self

    def predict(self, file_path):
        if not self.knn_classifier:
            raise 'Model must be trained before prediction'
        else:
            X, Y, entity_to_tag = self.produce_raw_x_and_y(file_path)
            predictions = map(lambda y: self.word2vec[y], Y)
            actual = map(lambda y: Word2VecKNN.types_to_ints[entity_to_tag[y]], Y)
            return self.knn_classifier.predict(predictions), actual
