from parser_with_pos import ParserWithPOS
from hmm import HMM

from sklearn.metrics import classification_report

def parse_test(testing_file):
	sentences = []
	pos = []
	ner_types = []
	new_sentence = True
	with open(testing_file, 'r') as tf:
		for line in tf:
			tokens = line.split()
			if len(tokens) == 0:
				new_sentence = True
				continue
			if new_sentence:
				sentences.append([])
				pos.append([])
				ner_types.append([])
			sentences[-1].append(tokens[0])
			pos[-1].append(tokens[1])
			ner_types[-1].append(tokens[3].split('-')[-1])
			new_sentence = False
	return sentences, pos, ner_types




if __name__ == "__main__":

	prefix = 'I-CAB_All/NER-09/'
	training_file = prefix + 'I-CAB-evalita09-NER-training.iob2'
	testing_file = prefix + 'I-CAB-evalita09-NER-test.iob2'

	clf = HMM(training_file)
	clf.train()

	sentences, pos, ner_types = parse_test(testing_file)

	ner_pred = []
	ner_true = []
	for sentence, ner_true_i in zip(sentences, ner_types):
		prediction = clf.predict(' '.join(sentence))
		ner_pred += [tag[1] for tag in prediction]
		ner_true += ner_true_i

	print classification_report(ner_true, ner_pred)

