class Sentence:
    def __init__(self):
        self.data = []

    def add(self, word, pos_tag, tag):
        self.data.append((word, pos_tag, tag))

    def words(self):
        return map(lambda x: x[0], self.data)

    def entity_tags(self):
        return map(lambda x: x[2], self.data)

    def __str__(self):
        return ''.join(self.words())

    def features(self):
        all_features = []
        for i, word in enumerate(self.data):
            all_features.append(self.feature(i))
        return all_features

    def feature(self, i):
        word = self.data[i][0]
        pos_tag = self.data[i][1]
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'postag=' + pos_tag,
            'postag[:2]=' + pos_tag[:2],
        ]
        self.extend_previous_word_features(features, i)
        self.extend_next_word_features(features, i)
        return features

    def extend_next_word_features(self, features, i):
        if i < len(self.data) - 1:
            word = self.data[i + 1][0]
            pos_tag = self.data[i + 1][1]
            features.extend([
                '+1:word.lower=' + word.lower(),
                '+1:word.istitle=%s' % word.istitle(),
                '+1:word.isupper=%s' % word.isupper(),
                '+1:postag=' + pos_tag,
                '+1:postag[:2]=' + pos_tag[:2],
            ])
        else:
            features.append('EOS')

    def extend_previous_word_features(self, features, i):
        if i > 0:
            word = self.data[i - 1][0]
            pos_tag = self.data[i - 1][1]
            features.extend([
                '-1:word.lower=' + word.lower(),
                '-1:word.istitle=%s' % word.istitle(),
                '-1:word.isupper=%s' % word.isupper(),
                '-1:postag=' + pos_tag,
                '-1:postag[:2]=' + pos_tag[:2],
            ])
        else:
            features.append('BOS')
