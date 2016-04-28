from lib.entropy.parser import Parser
from sklearn.linear_model import LogisticRegression
from lib.hot_encoder import Encoder
import numpy as np

tag_to_num = {
  'PER': 0,
  'ORG': 1,
  'LOC': 2,
  'GPE': 3
}

training_file_path = '../../I-CAB_All/NER-09/I-CAB-evalita09-NER-training.iob2'
test = '../../I-CAB_All/test.txt'
parsed = Parser(training_file_path)
sentences = filter(lambda x: x.contains_entity(), parsed.sentences)

X = np.matrix(map(lambda x: x.get_features(), sentences))
Y = map(lambda x: tag_to_num[x.tag], sentences)

print X
print Y

e = Encoder()
X = e.encode_matrix([4, 5, 6, 7, 9], X, train=True)

print X.shape

lg = LogisticRegression()

lg.fit(X, Y)


