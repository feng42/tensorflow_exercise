import codecs
import collections
from operator import itemgetter
RAW_DATA = "/home/feng/code_exercise/dataset/simple-examples/data/ptb.train.txt"
VOCAB_OUTPUT = "/home/feng/code_exercise/dataset/simple-examples/gen_data/ptb.vocab"

counter = collections.Counter()
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

sorted_words_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_words_to_cnt]

sorted_words = ["<eos>"] + sorted_words

with codecs.open(VOCAB_OUTPUT, 'w','utf-8') as fo:
    for word in sorted_words:
        fo.write(word+'\n')