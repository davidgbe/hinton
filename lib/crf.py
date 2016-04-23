from itertools import chain

import pycrfsuite
from lib.simple_parser import SimpleParser
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


class CRF:
    def __init__(self, train, test):
        train_sets = SimpleParser(train).parse()
        test_sets = SimpleParser(test).parse()
        self.X_train, self.y_train = [s.features() for s in train_sets], [s.entity_tags() for s in train_sets]
        self.X_test, self.y_test = [s.features() for s in test_sets], [s.entity_tags() for s in test_sets]

    def train(self, params):
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(self.X_train, self.y_train):
            trainer.append(xseq, yseq)

        trainer.set_params(params)

        trainer.train('result.out')

    def predict(self):
        tagger = pycrfsuite.Tagger()
        tagger.open('result.out')
        self.info = tagger.info
        self.y_pred = [tagger.tag(xseq) for xseq in self.X_test]
        return self.y_pred

    def performance_report(self):
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(self.y_test)))
        y_pred_combined = lb.transform(list(chain.from_iterable(self.y_pred)))

        tagset = sorted(set(lb.classes_) - {'O'})
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
                y_true_combined,
                y_pred_combined,
                labels=[class_indices[cls] for cls in tagset],
                target_names=tagset,
        )
