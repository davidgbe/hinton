from lib.entropy.parser import Parser
from sklearn.neighbors import KNeighborsClassifier
import gensim
import numpy as np

class Word2VecKNN(object):
    tags_to_ints = { 'LOC': 0, 'GPE': 1, 'PER': 2, 'ORG': 3 }

    def __init__(self, num_neighbors=5, hidden_size=100, power=2.0, resample=False):
        self.power = power
        self.num_neighbors = num_neighbors
        self.hidden_size = hidden_size

    def produce_sentences(self, file_path):
        parsed = Parser('../' + file_path)
        return filter(lambda s: s.contains_entity(), parsed.sentences)

    def weight_function(self, distances):
        return map(lambda x: 1.0/(pow(x, self.power)), distances)

    def train(self, file_path):
        sentences = self.produce_sentences(file_path)
        self.training_sentences = []
        self.training_entities = []
        self.training_tags = []
        for sen in sentences:
            self.training_sentences.append(sen.full_sentence())
            self.training_entities.append(sen.entity)
            self.training_tags.append(Word2VecKNN.tags_to_ints[sen.tag])
        return self

    def predict(self, file_path):
        test_sentences = self.produce_sentences(file_path)
        X = self.training_sentences + map(lambda s: s.full_sentence(), test_sentences)
        word2vec = gensim.models.Word2Vec(X, size=self.hidden_size, min_count=1, workers=4)

        knn_classifier = KNeighborsClassifier(n_neighbors=self.num_neighbors, weights=self.weight_function)
        knn_classifier.fit(map(lambda s: word2vec[s], self.training_entities), self.training_tags)

        test_entities = map(lambda s: s.entity, test_sentences)

        vectorized_test_entities = map(lambda s: word2vec[s.entity], test_sentences)
        actual_tags = map(lambda s: Word2VecKNN.tags_to_ints[s.tag], test_sentences)
        return knn_classifier.predict(vectorized_test_entities), actual_tags
