"""Sadaf Khan, LING571, HW7, 11/21/2021. Program that creates and evaluates a distributional model of word similarity 
based on local context term cooccurrence."""

import os
import sys
import nltk
import re
import copy
import math
import scipy
from collections import defaultdict
from scipy.stats.stats import spearmanr

nltk.download('brown')

window_size = int(sys.argv[1])
weighting = sys.argv[2]
judgment_filename = sys.argv[3]
output_filename = sys.argv[4]

# Read in a corpus that will form the basis of the distributional model and perform basic preprocessing.
brown_words = nltk.corpus.brown.words()

tokens = []
# All words should be lowercase
# Punctuation should removed, both from individual words and from the corpus
for word in brown_words:
    word = re.sub(r'\W+', '', word.lower())
    if word == '':
        pass
    else:
        tokens.append(word)

# stores the feature vectors for each word as a dictionary
vectors = {}

# For each word in the corpus:
for i in range(len(tokens)):
    center = tokens[i]
    if i < window_size:
        window_neg = tokens[0:i]
    else:
        window_neg = tokens[i - window_size:i]
    window_pos = tokens[i + 1:i + window_size + 1]
    window = window_neg + window_pos

    # Create a vector representation based on word co-occurrence in a specified window around the word.
    if center not in vectors:
        vectors[center] = defaultdict(int)
    for feature in window:
        vectors[center][feature] += 1

# stores the total of all the counts per word/feature. row and column totals for a given word are the same
prob = defaultdict(float)

# count total of every row
for vector in vectors:
    prob[vector] = sum(vectors[vector].values())

# Each element in the vector should receive weight according to a specified weighting
if weighting == 'PMI' or weighting == 'PPMI':
    # hold onto the original values if needed for later
    counts_frozen = copy.deepcopy(vectors)
    table_total = sum(prob.values())

    for word in vectors:
        for feature in vectors[word]:
            p_wf = (vectors[word][feature]) / table_total
            p_w = prob[word] / table_total
            p_f = prob[feature] / table_total
            ppmi = max(math.log(p_wf, 2) - (math.log(p_w, 2) + math.log(p_f, 2)), 0)
            vectors[word][feature] = ppmi

# Read in a file of human judgments of similarity between pairs of words.
judgements = open(os.path.join(os.path.dirname(__file__), judgment_filename), 'r').read().split("\n")[:-1]

with open(output_filename, 'w', encoding='utf8') as g:
    ratings = []
    sims = []

    # For each word pair in the file:
    for entry in judgements:
        entry_formatted = entry.split(",")
        word_pair = entry_formatted[0:2]
        rating = float(entry_formatted[2])
        word1 = word_pair[0]
        word2 = word_pair[1]
        ratings.append(rating)

        # For each word in the word pair:
        for word in word_pair:
            # Print the word...
            g.write(word + ": ")

            if word in vectors:
                top_features = sorted(vectors[word], key=vectors[word].get, reverse=True)[:10]

                # and its ten (10) highest weighted features (words) and their weights, in the form feature:weight
                for feature in top_features:
                    g.write(feature + ":" + str(vectors[word][feature]) + " ")
                g.write("\n")
            else:
                g.write("\n")

        # Compute the similarity between the two words, based on cosine similarity, print in the form wd1,wd2:similarity
        if word1 in vectors and word2 in vectors:
            cosine_headers = list(set(list(vectors[word1].keys()) + list(vectors[word2].keys())))
            word1_st_vec = []
            word2_st_vec = []
            for header in cosine_headers:
                word1_st_vec.append(vectors[word1][header])
                word2_st_vec.append(vectors[word2][header])
            similarity = 1 - (scipy.spatial.distance.cosine(word1_st_vec, word2_st_vec))
        else:
            similarity = 0
        sims.append(similarity)
        g.write(word1 + "," + word2 + ":" + str(similarity) + "\n")

    # Compute and print the Spearman correlation between the computed similarity scores vs. the human-generated scores
    correlation = spearmanr(sims, ratings)[0]
    g.write("Correlation:" + str(correlation))
