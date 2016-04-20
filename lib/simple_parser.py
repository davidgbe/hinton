import os
from lib.sentence import Sentence


class SimpleParser:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []

    def parse(self):
        if self.data:
            return self.data
        this_file = os.path.dirname(__file__)
        with open(os.path.join(this_file, self.file_name)) as f:
            lines = f.readlines()
        sentences = []
        current_sentence = Sentence()
        for line in lines:
            line = line.strip()
            if line == "":
                sentences.append(current_sentence)
                current_sentence = Sentence()
            else:
                (word, pos_tag, source, tag) = line.split()
                current_sentence.add(word, pos_tag, tag)
        self.data = sentences
        return self.data
