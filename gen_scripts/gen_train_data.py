import codecs
import sys

#RAW_DATA = "/home/feng/code_exercise/dataset/simple-examples/data/ptb.train.txt"
RAW_DATA = "/home/feng/code_exercise/dataset/simple-examples/data/ptb.valid.txt"
VOCAB = "/home/feng/code_exercise/dataset/simple-examples/gen_data/ptb.vocab"
OUTPUT_DATA = "/home/feng/code_exercise/dataset/simple-examples/gen_data/ptb.valid"

with codecs.open(VOCAB, "r", "utf-8") as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}


def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]


fin = codecs.open(RAW_DATA, 'r', 'utf-8')
fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')
for line in fin:
    words = line.strip().split() + ["<eos>"]
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)
fin.close()
fout.close()
