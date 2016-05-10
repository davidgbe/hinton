import os

class ParserWithPOS(object):
  def __init__(self, data_path):
    self.data_path = data_path
    self.entities = {}

  def fresh_container(self):
    return {
      'sentence': [],
      'case': [],
      'pos': [],
      'ner_type': []
    }

  def __iter__(self):
    this_file = os.path.dirname(__file__)
    container = self.fresh_container()
    need_clear = False
    curr_entity = ''
    curr_entity_type = ''
    for line in open(os.path.join(this_file, self.data_path)):
      parsed_line = line.split()
      if not parsed_line:
        need_clear = True
        yield container
      else:
        if need_clear:
          container = self.fresh_container()
          need_clear = False
        word, pos, doc_ID, classification = tuple(parsed_line)
        if classification != 'O':
          if curr_entity == '':
            curr_entity = word
            curr_entity_type = classification.split('-')[1]
          else:
            curr_entity += (' ' + word)
          ner_type = curr_entity_type
        else:
          if curr_entity:
            curr_entity = ''
          ner_type = 'O'
        container['sentence'].append(word.lower())
        container['case'].append(word[0].isupper())
        container['pos'].append(pos)
        container['ner_type'].append(ner_type)

  def retrieve_entities(self):
    return self.entities

