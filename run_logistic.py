from lib.entropy.parser import Parser
from sklearn.linear_model import LogisticRegression
from lib.hot_encoder import Encoder
from sklearn.metrics import classification_report
import numpy as np

tag_to_num = {
  'LOC': 0,
  'GPE': 1,
  'PER': 2,
  'ORG': 3
}

training_file_path = '../../I-CAB_All/NER-09/I-CAB-evalita09-NER-training.iob2'
test_file_path = '../../I-CAB_All/NER-09/I-CAB-evalita09-NER-test.iob2'

train_parsed = Parser(training_file_path)
train_sentences = filter(lambda x: x.contains_entity(), train_parsed.sentences)

test_parsed = Parser(test_file_path)
test_sentences = filter(lambda x: x.contains_entity(), test_parsed.sentences)

train_X = np.matrix(map(lambda x: x.get_features(), train_sentences))
train_Y = map(lambda x: tag_to_num[x.tag], train_sentences)

test_X = np.matrix(map(lambda x: x.get_features(), test_sentences))
test_Y = map(lambda x: tag_to_num[x.tag], test_sentences)

e = Encoder()
train_X = e.encode_matrix([4, 5, 6, 7, 9], train_X, train=True).astype('float64')
test_X = e.encode_matrix([4, 5, 6, 7, 9], test_X).astype('float64')

print test_X.dtype

lg = LogisticRegression()

lg.fit(train_X, train_Y)

predicted_Y = lg.predict(test_X)

print classification_report(test_Y, predicted_Y, target_names=[key for key in tag_to_num])
