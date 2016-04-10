class PerformanceReporter(object):
  ints_to_types = { 0: 'LOC', 1: 'GPE', 2: 'PER', 3: 'ORG' }

  def __init__(self, predictions, actual_values):
    self.report = {}
    for i in range(4):
      self.report[i] = {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0
      }
    self.predictions = predictions
    self.actual_values = actual_values

  def analyze(self):
    if len(self.predictions) != len(self.actual_values):
      raise 'Must provide same number of predictions and results'
    for idx in range(len(self.predictions)):
      pred = self.predictions[idx]
      actual = self.actual_values[idx]
      if pred == actual:
        self.report[actual]['true_positives'] += 1
      else:
        self.report[actual]['false_negatives'] += 1
        self.report[pred]['false_positives'] += 1
    for idx in self.report:
      tps = float(self.report[idx]['true_positives'])
      self.report[idx]['precision'] = tps / (self.report[idx]['false_positives'] + tps)
      self.report[idx]['recall'] = tps / (self.report[idx]['false_negatives'] + tps)
      p = self.report[idx]['precision']
      r = self.report[idx]['recall']
      self.report[idx]['f1'] = 2 * (p * r) / (p + r)
    return self.report

  def give_report(self):
    self.analyze()
    props = ['true_positives', 'false_positives', 'false_negatives', 'precision', 'recall', 'f1']
    for idx in PerformanceReporter.ints_to_types:
      print PerformanceReporter.ints_to_types[idx] + ':'
      for key in props:
        print key + ' ' + str(self.report[idx][key])

  def abbreviated_report(self):
    self.analyze()
    for idx in PerformanceReporter.ints_to_types:
      print PerformanceReporter.ints_to_types[idx] + ':'
      print self.report[idx]['f1']

