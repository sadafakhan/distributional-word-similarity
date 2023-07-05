"""Sadaf Khan, LING571, HW7, 11/23/2021. Program that evaluates a predictive CBOW distributional model of word
similarity using word2vec."""
import os
import sys
import nltk
import re
from gensim.models import Word2Vec
from scipy.stats.stats import spearmanr

window_size = int(sys.argv[1])
judgement_filename = sys.argv[2]
output_filename = sys.argv[3]

# Read in a corpus that will form the basis of the predictive CBOW distributional model and perform basic preprocessing.
brown_sents = nltk.corpus.brown.sents()
processed_sents = []

# All words should be lowercase
# Punctuation should removed, both from individual words and from the corpus
for sent in brown_sents:
    new_sent = []
    for i in range(len(sent)):
        if sent[i].isalnum():
            word_edited = re.sub(r'\W+', '', sent[i].lower())
            new_sent.append(word_edited)
    processed_sents.append(new_sent)

# Build a continuous bag of words model using a standard implementation package, such as gensim’s word2vec
cbow = Word2Vec(sentences=processed_sents, window=window_size, min_count=1)

# Read in a file of human judgments of similarity between pairs of words.
judgements = open(os.path.join(os.path.dirname(__file__), judgement_filename), 'r').read().split("\n")[:-1]

with open(output_filename, 'w', encoding='utf8') as g:
    ratings = []
    sims = []

    # For each word pair in the file:
    for entry in judgements:
        entry_formatted = entry.split(",")
        rating = float(entry_formatted[2])
        word1 = entry_formatted[0]
        word2 = entry_formatted[1]
        ratings.append(rating)

        # Compute the similarity between the two words, using the word2vec model
        similarity = cbow.wv.similarity(word1, word2)
        sims.append(similarity)

        # Print out the similarity as: wd1,wd2:similarity
        g.write(word1 + "," + word2 + ":" + str(similarity) + "\n")

    # Compute & print the Spearman correlation between the computed similarity scores vs. the human-generated scores
    correlation = spearmanr(sims, ratings)[0]
    g.write("Correlation:" + str(correlation))
