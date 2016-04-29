class Sentence(object):
    to_print = ['entity', 'entity_pos', 'context', 'context_poses', 'entity_position', 'other_entities', 'tag']
    features = {
        'num_capitalized_words_in_entity': 0,    
        'entity_is_title': 0,
        'entity_has_first_capitalized': 0,
        'len_entity': 0,
        'entity_contains_number': 0,
        'entity_pos_tag': 1,
        'entity_is_upper': 0,
        'occurrences_of_per': 0,
        'occurrences_of_gpe': 0,
        'occurrences_of_org': 0,
        'occurrences_of_loc': 0,
        'prev_pos': 1,
        'prev_word': 1,
        'prev_word_suff_2': 1,
        'prev_word_suff_3': 1,
        'prev_pos_is_prep': 0,
        'next_pos': 1,
        'next_word': 1,
        'next_word_suff_2': 1,
        'next_word_suff_3': 1,
    }

    def __init__(self, entity, entity_pos, context, context_poses, other_entities, position=-1, tag=None):
        self.entity = entity
        self.entity_pos = entity_pos
        self.context = context
        self.context_poses = context_poses
        self.other_entities = other_entities
        self.entity_position = position
        self.tag = tag

    def __str__(self):
        return '\n'.join(map(lambda x: x.upper() + ': ' + str(self.__dict__[x]), Sentence.to_print))

    def contains_entity(self):
        return True if len(self.entity) else False

    def get_features(self):
        return map(lambda f: getattr(self,f)(), Sentence.features)

    @staticmethod
    def process_sentence_data(sentence_data):
        last_tag = ''
        unique_entities = []
        in_entity = False
        entity_start = 0
        sentences = []
        for i in range(len(sentence_data)):
            (word, pos_tag, tag) = sentence_data[i]
            if tag != 'O':
                if tag != last_tag:
                    in_entity = True
                    entity_start = i
            elif in_entity:
                in_entity = False
                unique_entities.append((entity_start, i))
            last_tag = tag
        for entity in unique_entities:
            (start, end) = entity
            entity = ' '.join(map(lambda x: x[0], sentence_data[start:end]))
            entity_pos = sentence_data[start][1]
            rest = sentence_data[:start] + sentence_data[end:]
            context = map(lambda x: x[0], rest)
            context_poses = map(lambda x: x[1], rest)
            tag = sentence_data[start][2]
            other_entities = []
            for ent in unique_entities:
                if ent[0] != start:
                    other_entities.append(sentence_data[ent[0]][2])
            sentences.append(Sentence(entity, entity_pos, context, context_poses, other_entities, start, tag))
        return sentences

    #features
    def num_capitalized_words_in_entity(self):
        return reduce(lambda x, y: (x + 1) if y.istitle() else x, self.entity.split(' '), 0)

    def entity_is_title(self):
        return 1 if self.entity.istitle() else 0

    def entity_has_first_capitalized(self):
        return 1 if self.entity.split(' ')[0].istitle() else 0

    def len_entity(self):
        return len(self.entity.split(' '))

    def entity_contains_number(self):
        return 1 if any(char.isdigit() for char in self.entity) else 0

    def entity_pos_tag(self):
        return self.entity_pos

    def entity_is_upper(self):
        return 1 if self.entity.isupper() else 0

    def occurrences_of_per(self):
        return self.other_entity_of_type('PER')

    def occurrences_of_gpe(self):
        return self.other_entity_of_type('GPE')

    def occurrences_of_org(self):
        return self.other_entity_of_type('ORG')

    def occurrences_of_loc(self):
        return self.other_entity_of_type('LOC')

    #prev word features
    def prev_pos(self):
        if self.entity_position == 0:
            return 'BOS'
        else:
            return self.context_poses[self.entity_position - 1]

    def prev_word(self):
        if self.entity_position == 0:
            return 'BOS'
        else:
            return self.context[self.entity_position - 1]

    def prev_word_suff_2(self):
        if self.entity_position == 0:
            return 'BOS'
        else:
            return self.context[self.entity_position - 1][-2:]

    def prev_word_suff_3(self):
        if self.entity_position == 0:
            return 'BOS'
        else:
            return self.context[self.entity_position - 1][-3:]

    def prev_pos_is_prep(self):
        if self.entity_position == 0:
            return 0
        elif self.context_poses[self.entity_position - 1] == 'E':
            return 1
        else:
            return 0

    #next word features
    def next_pos(self):
        if self.entity_position == len(self.context_poses):
            return 'EOS'
        else:
            return self.context_poses[self.entity_position]

    def next_word(self):
        if self.entity_position == len(self.context):
            return 'EOS'
        else:
            return self.context[self.entity_position]

    def next_word_suff_2(self):
        if self.entity_position == 0:
            return 'BOS'
        else:
            return self.context[self.entity_position - 1][-2:]

    def next_word_suff_3(self):
        if self.entity_position == 0:
            return 'BOS'
        else:
            return self.context[self.entity_position - 1][-3:]

    #helpers
    def other_entity_of_type(self, ent_type):
        return reduce(lambda x, y: (x + 1) if ent_type == y else x, self.other_entities, 0)
