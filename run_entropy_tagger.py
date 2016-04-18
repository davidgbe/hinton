from lib.parser_with_pos import ParserWithPOS

training_file_path = '../I-CAB_All/NER-09/I-CAB-evalita09-NER-training.iob2'
parsed = ParserWithPOS(training_file_path)
for p in parsed:
  print p
