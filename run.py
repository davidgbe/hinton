import sys

from lib.crf import CRF
from lib.performance_reporter import PerformanceReporter
from lib.word2vec_knn import Word2VecKNN

training_file_path = '../I-CAB_All/NER-09/I-CAB-evalita09-NER-training.iob2'
testing_file_path = '../I-CAB_All/NER-09/I-CAB-evalita09-NER-test.iob2'
if len(sys.argv) == 1:
    tagger = 'word2vec'
else:
    tagger = sys.argv[1]

if tagger == 'word2vec':
    training_model = Word2VecKNN(training_file_path).train()
    X, Y = Word2VecKNN(testing_file_path).produce_word2vec_representation()
    predictions = training_model.predict(X)
    reporter = PerformanceReporter(predictions, Y)
    reporter.give_report()
elif tagger == 'crf':
    crf = CRF(training_file_path, testing_file_path)
    crf.train()
    crf.predict()
    print crf.performance_report()
