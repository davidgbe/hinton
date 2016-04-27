class Sentence(object):
    to_print = ['entity', 'entity_pos', 'context', 'context_poses', 'entity_position', 'other_entities', 'tag']
    features = [
        'num_capitalized_words_in_entity',    
        'entity_is_title',
        'entity_has_first_capitalized',
        'len_entity',
        'prev_pos',
        'next_pos',
        'prev_word',
        'next_word',
        'entity_contains_number',
        'entity_pos_tag',
        'prev_pos_is_prep',
        'occurrences_of_per',
        'occurrences_of_gpe',
        'occurrences_of_org',
        'occurrences_of_loc'
    ]

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
        return self.entity.istitle()

    def entity_has_first_capitalized(self):
        return self.entity.split(' ')[0].istitle()

    def len_entity(self):
        return len(self.entity.split(' '))

    def prev_pos(self):
        if self.entity_position == 0:
            return 'BOS'
        else:
            return self.context_poses[self.entity_position - 1]

    def next_pos(self):
        if self.entity_position == len(self.context_poses):
            return 'EOS'
        else:
            return self.context_poses[self.entity_position]

    def prev_word(self):
        if self.entity_position == 0:
            return 'BOS'
        else:
            return self.context[self.entity_position - 1]

    def next_word(self):
        if self.entity_position == len(self.context):
            return 'EOS'
        else:
            return self.context[self.entity_position]

    def entity_contains_number(self):
        return any(char.isdigit() for char in self.entity)

    def entity_pos_tag(self):
        return self.entity_pos

    def prev_pos_is_prep(self):
        if self.entity_position == 0:
            return False
        elif self.context_poses[self.entity_position - 1] == 'E':
            return True
        else:
            return False

    def occurrences_of_per(self):
        return self.other_entity_of_type('PER')

    def occurrences_of_gpe(self):
        return self.other_entity_of_type('GPE')

    def occurrences_of_org(self):
        return self.other_entity_of_type('ORG')

    def occurrences_of_loc(self):
        return self.other_entity_of_type('LOC')

    #helpers
    def other_entity_of_type(self, ent_type):
        return reduce(lambda x, y: (x + 1) if ent_type == y else x, self.other_entities, 0)
