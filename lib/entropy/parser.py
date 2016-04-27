import os
from sentence import Sentence

class Parser(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.sentences = []
        self.parse()

    def parse(self):
        this_file = os.path.dirname(__file__)
        current_sentence = []
        for line in open(os.path.join(this_file, self.data_path)):
            split_line = line.split()
            if not split_line:
                self.sentences += Sentence.process_sentence_data(current_sentence)
                current_sentence = []
            else:
                (word, pos_tag, source, tag) = split_line
                tag = tag if tag == 'O' else tag.split('-')[1]
                current_sentence.append((word, pos_tag, tag))

