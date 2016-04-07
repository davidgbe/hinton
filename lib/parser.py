import os

class Parser(object):
  def __init__(self, data_path):
    self.data_path = data_path
    self.entities = {}

  def __iter__(self):
    this_file = os.path.dirname(__file__)
    sentence = []
    need_clear = False
    curr_entity = ''
    curr_entity_type = ''
    for line in open(os.path.join(this_file, self.data_path)):
      parsed_line = line.split()
      if not parsed_line:
        need_clear = True
        yield sentence
      else:
        if need_clear:
          del sentence[:]
          need_clear = False
        word = parsed_line[0]
        classification = parsed_line[3]
        if classification != 'O':
          if curr_entity == '':
            curr_entity = word
            curr_entity_type = classification.split('-')[1]
          else:
            curr_entity += (' ' + word)
        else:
          if curr_entity:
            sentence.append(curr_entity)
            self.entities[curr_entity] = curr_entity_type
            curr_entity = ''
          sentence.append(word)

  def retrieve_entities(self):
    return self.entities


