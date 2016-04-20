class SimpleParser:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []

    def parse(self):
        if self.data:
            return self.data
        with open(self.file_name) as f:
            lines = f.readlines()
        sentences = []
        current_sentence = []
        for line in lines:
            line = line.strip()
            if line == "":
                sentences.append(current_sentence)
                current_sentence = []
            else:
                (word, pos_tag, source, tag) = line.split()
                current_sentence.append((word, pos_tag, tag))
        self.data = sentences
        return self.data
