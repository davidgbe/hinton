from lib.entropy.parser import Parser
from sklearn.linear_model import LogisticRegression

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

X = map(lambda x: x.get_features(), sentences)
Y = map(lambda x: tag_to_num[x.tag], sentences)

print X
print Y

lg = LogisticRegression()

lg.fit(X, Y)
