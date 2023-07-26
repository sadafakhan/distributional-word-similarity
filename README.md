# distributional-word-similarity
```distributional-word-similarity```creates and evaluates a distributional model of word similarity based on local context term cooccurrence. 

```hw7_dist_similarity.sh``` implements and evaluates a distributional similarity model. 

Args: 
* ```window```: an integer specifying the size of the context window (on either side) for the model
* ```weighting```: a string specifying the weighting scheme. "FREQ" refers to "term frequency" (the number of times the word appeared in the context of the target), and "PMI" refers to "positive point-wise mutual information" (a variant of PMI where negative association scores are removed). 
* ```judgment_filename```: path to the human judgment file, which has pairs of words and their similarity to evaluate against (cf. mc_similarity.txt). Each line has the format: ```word1, word2, similarity_score```

Returns: 
* ```output_filename```: path to the output file which contains the results of comptuing similarities and correlations over the word pairs. The file name identifies the configuration under which is was run, e.g. ````hw7_sim_<window>_<weighting>_output.txt```

To run: 
```
src/hw7_dist_similarity.sh <window> <weighting> input/mc_similarity.txt output/<output_filename>
```

```hw7_cbow_similarity.sh``` implements and evaluates a continuous bag-of-words similarity model. 

Args: 
* ```window```: an integer specifying the size of the context window (on either side) for the model
** ```judgment_filename```: path to the human judgment file, which has pairs of words and their similarity to evaluate against (cf. mc_similarity.txt). Each line has the format: ```word1, word2, similarity_score```

Returns: 
* ```output_filename```: path to the output file which contains the results of comptuing similarities and correlations over the word pairs. The file name identifies the configuration under which is was run, e.g. ````hw7_sim_<window>_CBOW_output.txt```


To run: 
```
src/hw7_cbow_similarity.sh <window> input/mc_similarity.txt output/<output_filename>
```

HW7 OF LING571 (11/24/2021)