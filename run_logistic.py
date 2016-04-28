from lib.entropy.entropy_classifier import EntropyClassifier

training_file_path = '../../I-CAB_All/NER-09/I-CAB-evalita09-NER-training.iob2'
test_file_path = '../../I-CAB_All/NER-09/I-CAB-evalita09-NER-test.iob2'

ec = EntropyClassifier(training_file_path)
ec.predict_and_report_from_file(test_file_path)
