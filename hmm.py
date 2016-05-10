from parser_with_pos import ParserWithPOS
from collections import Counter, OrderedDict
import math
"""
TODO: Conditional probability for capitalization
"""

class HMM(object):

	case_IDs = { False: 0, True: 1 }
	ner_types = [ 'LOC', 'GPE', 'PER', 'ORG', 'O' ]

	def __init__(self, file_path):

		self.containers = [ container for container in ParserWithPOS(file_path) ]

		# num_tags x num_words
		self.emission_probabilities = []

		# num_tags x num_tags
		self.transition_probabilities = []

	def train(self):

		"""
		Trains an HMM model
		"""

		# extract Part Of Speech (POS) vocabulary

		pos_vocabulary = set()

		for container in self.containers:
			for pos in container['pos']:
				pos_vocabulary.add(pos)

		# construct tag vocabulary; a tag is a (POS, NER type) pair.

		tag_vocabulary = []

		for pos in pos_vocabulary:
			for ner_type in self.ner_types:
				tag_vocabulary.append((pos, ner_type))

		# construct tag -> int mapping

		self.tag_IDs = OrderedDict()

		for tag_ID, tag in enumerate(tag_vocabulary):
			self.tag_IDs[tag] = tag_ID

		# extract word vocabulary

		word_vocabulary = set()

		for container in self.containers:
			for word in container['sentence']:
				word_vocabulary.add(word.lower())

		# construct word -> int mapping

		self.word_IDs = OrderedDict()

		for word_ID, word in enumerate(word_vocabulary):
			self.word_IDs[word] = word_ID

		# initialize word emission counts

		word_emission_counts = [ None ] * len(self.tag_IDs)
		for tag_ID in range(len(word_emission_counts)):
			word_emission_counts[tag_ID] = Counter()

		# calculate word emission counts

		for container in self.containers:
			for word, pos, ner_type in zip(container['sentence'], container['pos'], container['ner_type']):
				word_ID = self.word_IDs[word]
				tag_ID = self.tag_IDs[(pos, ner_type)]
				word_emission_counts[tag_ID][word_ID] += 1

		# initialize case emission counts

		case_emission_counts = [ None ] * len(self.tag_IDs)
		for tag_ID in range(len(case_emission_counts)):
			case_emission_counts[tag_ID] = Counter()

		# calculate capitalization emission counts

		for container in self.containers:
			for case, pos, ner_type in zip(container['case'], container['pos'], container['ner_type']):
				case_ID = self.case_IDs[case]
				tag_ID = self.tag_IDs[(pos, ner_type)]
				case_emission_counts[tag_ID][case_ID] += 1

		# initialize transition counts

		transition_counts = [ None ] * len(self.tag_IDs)
		for tag1_ID in range(len(self.tag_IDs)):
			transition_counts[tag1_ID] = Counter()

		# calculate transition counts

		for container in self.containers:
			for i in range(len(container['pos'])-1):
				tag_start = (container['pos'][i],   container['ner_type'][i]  )
				tag_end   = (container['pos'][i+1], container['ner_type'][i+1])
				tag_start_ID = self.tag_IDs[tag_start]
				tag_end_ID   = self.tag_IDs[tag_end]
				transition_counts[tag_start_ID][tag_end_ID] += 1

		# normalize word emission counts to get word emission probabilities

		self.word_emission_probabilities = [ None ] * len(self.tag_IDs)

		for tag_ID in range(len(self.word_emission_probabilities)):
			self.word_emission_probabilities[tag_ID] = {}
			denom = sum(word_emission_counts[tag_ID].values())
			for word_ID, count in word_emission_counts[tag_ID].items():
				self.word_emission_probabilities[tag_ID][word_ID] = float(count) / denom

		# normalize case emission counts to get case emission probabilities

		self.case_emission_probabilities = [ None ] * len(self.tag_IDs)

		for tag_ID in range(len(self.case_emission_probabilities)):
			self.case_emission_probabilities[tag_ID] = {}
			denom = sum(case_emission_counts[tag_ID].values())
			for case_ID, count in case_emission_counts[tag_ID].items():
				self.case_emission_probabilities[tag_ID][case_ID] = float(count) / denom

		# normalize transition counts to get transition probabilities

		self.transition_probabilities = [ None ] * len(self.tag_IDs)

		for tag1_ID in range(len(self.transition_probabilities)):
			self.transition_probabilities[tag1_ID] = [ 0.0 ] * len(self.tag_IDs)
			transition_count_denom = sum(transition_counts[tag1_ID].values())
			for tag2_ID, count in transition_counts[tag1_ID].items():
				self.transition_probabilities[tag1_ID][tag2_ID] = float(count) / transition_count_denom

		# calculate and normalize tag counts to get tag probabilities

		tag_counts = [ sum(case_emission_count.values()) for case_emission_count in case_emission_counts ]
		tag_counts_denom = sum(tag_counts)
		self.tag_priors = [ float(tag_count) / tag_counts_denom for tag_count in tag_counts ]

	def predict(self, sentence):
		"""
		Predicts tags for a sentence using the Viterbi Algorithm
		"""

		words = [ word.lower() for word in sentence.split() ]
		cases = [ word[0].isupper() for word in sentence.split() ]
		cases[0] = False

		# special case for length-zero sentences

		if len(sentence) == 0:
			return []

		# convert list of words to word_IDs; a word_ID of -1 represents a never-before seen word.

		for i in range(len(words)):
			if words[i] in self.word_IDs:
				words[i] = self.word_IDs[words[i]]
			else:
				words[i] = -1

		# convert list of cases to case_IDs

		for i in range(len(cases)):
			cases[i] = self.case_IDs[cases[i]]

		# calculate initial distribution

		emission_initial = []
		for tag_ID in range(len(self.tag_IDs)):
			emission_probability = 1.0
			case_emission_row = self.case_emission_probabilities[tag_ID]
			word_emission_row = self.word_emission_probabilities[tag_ID]
			if words[0] in word_emission_row:
				emission_probability *= word_emission_row[words[0]]
			if cases[0] in case_emission_row:
				emission_probability *= case_emission_row[cases[0]]
			emission_initial.append(emission_probability)

		p_initial = [ self.tag_priors[tag_ID] * emission_initial[tag_ID] for tag_ID in range(len(self.tag_IDs)) ]

		# enter loop

		prediction = self.viterbi_helper(p_initial, words[1:], cases[1:])

		# convert list of tag_IDs to tags

		prediction = [ self.tag_IDs.keys()[ID] for ID in prediction ]

		# return

		return prediction

	def viterbi_helper(self, p_prev, words, cases):
		"""
		Recursive helper function for the Viterbi Algorithm
		"""

		# exit condition
		if len(words) == 0:
			return [ p_prev.index(max(p_prev)) ]

		backpointers = [ 0 ] * len(self.tag_IDs)
		p_next = [ 0.0 ] * len(self.tag_IDs) 

		word_cur = words[0]
		case_cur = cases[0]

		# calculate p_next
		for tag_ID_next in range(len(self.word_emission_probabilities)):

			emission_probability = 1.0
			case_emission_row = self.case_emission_probabilities[tag_ID_next]
			word_emission_row = self.word_emission_probabilities[tag_ID_next]
			if word_cur != -1:
				if word_cur in word_emission_row:
					emission_probability *= word_emission_row[word_cur]
				else:
					continue
			if case_cur in case_emission_row:
				emission_probability *= case_emission_row[case_cur]

			"""
			# calculate emission probability
			if word_cur == -1:
				# handle never-before seen words; should be tag prior, weighted by entropy, so that tags for which there are many one-word appearances are weighted more heavily
				emission_probability = 1.0
			else:
				emission_row = self.word_emission_probabilities[tag_ID_next]
				if word_cur in emission_row:
					emission_probability = emission_row[word_cur]
				else:
					# prune on zero emission probability
					continue
			"""

			# calculate max transition probability
			for tag_ID_prev in range(len(self.tag_IDs)):
				transition_probability = p_prev[tag_ID_prev] * self.transition_probabilities[tag_ID_prev][tag_ID_next]
				if transition_probability > p_next[tag_ID_next]:
					p_next[tag_ID_next] = transition_probability
					backpointers[tag_ID_next] = tag_ID_prev

			p_next[tag_ID_next] *= emission_probability

		prediction = self.viterbi_helper(p_next, words[1:], cases[1:])
		
		path_head = prediction[0]
		return [ backpointers[path_head] ] + prediction


