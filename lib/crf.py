from lib.simple_parser import SimpleParser

train_sets = SimpleParser('../I-CAB_All/NER-09/I-CAB-evalita09-NER-training.iob2').parse()
test_sets = SimpleParser('../I-CAB_All/NER-09/I-CAB-evalita09-NER-test.iob2').parse()
print train_sets[1]
