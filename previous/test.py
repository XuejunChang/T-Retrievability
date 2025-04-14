import pyterrier as pt
import pyterrier_dr

model = pyterrier_dr.TctColBert()
print(model([
  {'qid': '0', 'query': 'Hello Terrier', 'docno': '0', 'text': 'The Five Find-Outers and Dog, also known as The Five Find-Outers, is a series of children\'s mystery books written by Enid Blyton.'},
  {'qid': '0', 'query': 'Hello Terrier', 'docno': '1', 'text': 'City is a 1952 science fiction fix-up novel by American writer Clifford D. Simak.'},
]))