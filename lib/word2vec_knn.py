from parser import Parser
from sklearn.neighbors import KNeighborsClassifier
import gensim
import numpy as np

class Word2VecKNN(object):
  types_to_ints = { 'LOC': 0, 'GPE': 1, 'PER': 2, 'ORG': 3 }

  def __init__(self, file_path, num_neighbors=5, hidden_size=100, power=2.0):
    sentences = Parser(file_path)
    self.word2vec = gensim.models.Word2Vec(sentences, size=hidden_size, min_count=1, workers=4)
    self.entities_to_types = sentences.retrieve_entities()
    self.power = power
    self.num_neighbors = num_neighbors

  def produce_word2vec_representation(self):
    X = []
    Y = []
    for key in self.entities_to_types:
      X.append(self.word2vec[key])
      Y.append(Word2VecKNN.types_to_ints[self.entities_to_types[key]])
    return X, Y

  def weight_function(self, distances):
    return map(lambda x: 1.0/(pow(x, self.power)), distances)

  def train(self):
    X, Y = self.produce_word2vec_representation()
    self.knn_classifier = KNeighborsClassifier(n_neighbors=self.num_neighbors, weights=self.weight_function)
    self.knn_classifier.fit(X, Y)
    return self

  def predict(self, X):
    if not self.knn_classifier:
      raise 'Model must be trained before prediction'
    else:
      return self.knn_classifier.predict(X)
