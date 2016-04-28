from lib.entropy.parser import Parser
from sklearn.linear_model import LogisticRegression
from lib.hot_encoder import Encoder
from sklearn.metrics import classification_report
import numpy as np

class EntropyClassifier(object):
    to_hot_encode = [5, 10, 11, 12, 13, 15, 16, 17, 18]
    tag_to_num = {
      'LOC': 0,
      'GPE': 1,
      'PER': 2,
      'ORG': 3
    }

    def __init__(self, training_file_path):
        self.classifier = LogisticRegression()
        self.encoder = Encoder()
        self.train(training_file_path)

    def produce_x_and_y(self, path, train=False):
        parsed = Parser(path)
        sentences = filter(lambda x: x.contains_entity(), parsed.sentences)
        X = np.matrix(map(lambda x: x.get_features(), sentences))
        X = self.encoder.encode_matrix(EntropyClassifier.to_hot_encode, X, train=train).astype('float64')
        Y = map(lambda x: EntropyClassifier.tag_to_num[x.tag], sentences)
        return (X, Y)

    def train(self, train_data_path):
        (X, Y) = self.produce_x_and_y(train_data_path, True)
        self.classifier.fit(X, Y)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_and_report_from_file(self, test_file_path):
        (X, Y) = self.produce_x_and_y(test_data_path)
        Y_pred = self.predict(X)
        print classification_report(Y, Y_pred, target_names=[key for key in EntropyClassifier.tag_to_num])

