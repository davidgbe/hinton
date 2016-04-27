NER-07.zip  contains the data  used for  the Named  Entity Recognition
Task at EVALITA 2007 (I-CAB Version 4.1).

Data  consist of  525 news  documents taken  from the  local newspaper
L'Adige.   The selected  news stories  belong to  four  different days
(September, 7th  and 8th 2004 and  October, 7th and 8th  2004) and are
grouped into five categories: Attualita', Cultura, Economia, Sport and
Trento.  They are  divided into a development part  (335 news stories,
for  a total  of  around 113,000  words)  and a  test  part (190  news
stories, for a total of around 69,000 words).

Development  data and  test data  for Named  Entity  Recognition (NER)
consist of  two separate text  files, with one  token per line  and an
empty line after each sentence.

Named Entities are annotated in the IOB2 format.

The Named Entity tag consists of two parts:
  1. the  IOB2 tag: 'B'  (for 'begin')  denotes the  first token  of a
  Named Entity,  I (for 'inside')  is used for  all other tokens  in a
  Named Entity, and 'O' (for 'outside') is used for all other words;
  2. the Entity  type tag: PER  (for Person), ORG  (for Organization),
  GPE (for Geo- Political Entity), or LOC (for Location).

Both development and  test data have been further  annotated with Part
of Speech information1 using  the Elsnet tagset for Italian (available
at:  http://www.evalita.it/sites/evalita.fbk.eu/files/doc2007/elsnet-tagset-IT.pdf).  
Please notice that  the corpus  has  been PoS-tagged  automatically  
with no  manual correction.

Each file  consists of four  columns separated by a  blank, containing
respectively the  token, the Elsnet  PoS-tag, the Adige news  story to
which the token belongs, and the Named Entity tag.

Example:
il RS adige20041008_id414157 O
capitano SS adige20041008_id414157 O
della ES adige20041008_id414157 O
Gerolsteiner SPN adige20041008_id414157 B-ORG
Davide SPN adige20041008_id414157 B-PER
Rebellin SPN adige20041008_id414157 I-PER
