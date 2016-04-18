import os

class ParserWithPOS(object):
  def __init__(self, data_path):
    self.data_path = data_path
    self.entities = {}

  def fresh_container(self):
    return {
      'sentence': [],
      'context_POSes': [],
      'entity': [],
      'entity_POSes': [],
      'tag': None,
      'position': None
    }

  def __iter__(self):
    this_file = os.path.dirname(__file__)
    container = self.fresh_container()
    need_clear = False
    curr_entity = ''
    curr_entity_type = ''
    position = 0
    for line in open(os.path.join(this_file, self.data_path)):
      parsed_line = line.split()
      if not parsed_line:
        need_clear = True
        yield container
      else:
        if need_clear:
          container = self.fresh_container()
          position = 0
          need_clear = False
        word = parsed_line[0]
        pos = parsed_line[1]
        classification = parsed_line[3]
        if classification != 'O':
          if curr_entity == '':
            curr_entity = word
            curr_entity_type = classification.split('-')[1]
            container['tag'] = curr_entity_type[:]
            container['position'] = position
          else:
            curr_entity += (' ' + word)
          container['entity'].append(word)
          container['entity_POSes'].append(pos)
        else:
          if curr_entity:
            self.entities[curr_entity] = curr_entity_type
            curr_entity = ''
          container['sentence'].append(word)
          container['context_POSes'].append(pos)
        position += 1

  def retrieve_entities(self):
    return self.entities
