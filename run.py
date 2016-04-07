from lib.word2vec_knn import Word2VecKNN
from lib.performance_reporter import PerformanceReporter

training_file_path = '../I-CAB_All/NER-09/I-CAB-evalita09-NER-training.iob2'
testing_file_path = '../I-CAB_All/NER-09/I-CAB-evalita09-NER-test.iob2'

training_model = Word2VecKNN(training_file_path).train()

X, Y = Word2VecKNN(training_file_path).produce_word2vec_representation()

predictions = training_model.predict(X)

reporter = PerformanceReporter(predictions, Y)
reporter.give_report()



