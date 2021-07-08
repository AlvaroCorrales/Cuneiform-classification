#%% CUNEIFORM LANGUAGE MODEL
### Alvaro Corrales Cano
### June 2021

# Imports 
import pandas as pd
import numpy as np

# Helper functions
## Preprocess cuneiform data
def preprocess(input):
    '''
    Preprocess lines in cuneiform script. Performs two steps: (1) Adds beggining- and end-of-sentence tokens 
    (B and E, respectively) (2) Adds a space between characters. Parameters:
    :input: Pandas Series, each line being a document in cuneiform alphabet with no separation between characters.
    '''
    # Add beggining- and end-of-sentence caracter to the end - just a B and an E
    input_mod = 'B' + input + 'E'

    # Split line by character
    for i in range(len(input)):
        input_mod[i] = " ".join(input_mod[i])

    return input_mod

## Count ngrams and put them in a dictionary
## From https://stackoverflow.com/questions/13423919/computing-n-grams-using-python
def ngrams(input, n):
    '''
    Create a dictionary with each ngram and the number of occurrences
    :input: string of characters separated by a white space (' ')
    :n: int. Length of ngram
    '''
   
    input = input.split(' ')
    output = {}
   
    for i in range(len(input)-n+1):
        g = ' '.join(input[i:i+n])
        output.setdefault(g, 0)
        output[g] += 1
       
    for key in list(output):
        if key[0] == 'E': # Drops 'E B' bigram
            del output[key]

    return output

## Get a list of ngrams - similar to ngrams function, but this one doesn't return a dictionary with frequencies
def ngram_list(input, n):
    '''
    Return a list of ngrams within sentence. Params:
    :input: String. Sequence of cuneiform characters
    :n: Int. length of ngram. 
    '''
    input = input.split(' ')
    _ngrams = []
    for i in range(len(input)-n+1):
        g = ' '.join(input[i:i+n])
        _ngrams.append(g)
    return _ngrams

## Calculate log probability of a sequence
def logprob(input, lang, probs):
    '''
    Calculate the probability of a sequence of ngrams. Params:
    :input: list of ngrams
    :lang: summerian language out of ['NEA', 'LTB', 'MPB', 'OLB', 'NEB', 'SUX', 'STB']
    '''
    _logprob = 0
    for _ngram in input:                                                                                            # Iterate over ngrams within list of ngrams for that sequence
        if _ngram in list(probs[lang]['Bigram']):                                                 
            _logprob = _logprob + np.log(probs[lang]['Probability'][probs[lang]['Bigram'] == _ngram].values)      # If there is a probability for that ngram, we add it to our conditional probability
        else:
            _logprob = _logprob + np.log(0.00001)                                                                   # Update by very low probability (almost zero)

    return float(_logprob)

# Define language model class
class CuneiPy:

    def __init__(self):
        self.languages = []
        self.train_data = pd.DataFrame()
        self._dfs = {}
        self._freqs = {}
        self._probs = {}
        self.n = None
        self.predictions = pd.DataFrame()

    def fit(self, df, input, target, n):
        '''
        Fits the model; i.e. calculates probability matrices with ngrams of length n for each language.
        Parameters:
        :_df: Pandas dataframe with raining data.
        :input: String. Name of column containing text in cuneiform alphabet. 
        :target: String. Name of column containing the target values of our model. Values should be strings
            such as "SUX" for Sumerian, "LTB" for Late Babylonian, etc.
        :n: Integer. Length of n-grams.
        '''
        self.languages = list(set(df[target]))
        self.train_data = df
        self.n = n

        # Preprocess training data
        self.train_data['cuneiform_mod'] = preprocess(df[input])

        # Split dataframes by language
        for lang in self.languages:
            self._dfs[lang] = self.train_data[self.train_data[target] == lang]

        # Put frequencies in dictionaries by language
        for lang in self.languages:
            self._freqs[lang] = ngrams(" ".join(self._dfs[lang]['cuneiform_mod']), n)

        # Put probabilities in a dictionary of dataframes - These are our probability matrices
        for lang in self.languages:
            self._probs[lang] = pd.DataFrame(list(self._freqs[lang].items()),columns = ['Bigram','Frequency']).sort_values(by = 'Frequency', ascending=False)
            self._probs[lang]['Probability'] = self._probs[lang]['Frequency'] / sum(self._probs[lang]['Frequency'])

    def predict(self, input):
        '''
        Predict language of a given sequence in cuneiform language. Parameters:
        :input: Pandas series containing text in cuneiform alphabet. Each observation will get assigned a 
            predicted language. 
        '''
        self.predictions['cuneiform'] = input

        # Preprocess text, just like in training
        self.predictions['cuneiform_mod'] = preprocess(self.predictions['cuneiform']) 

        # Get all n-grams from the text
        self.predictions['ngrams'] = [ngram_list(self.predictions['cuneiform_mod'][i], self.n) for i in range(len(self.predictions))]

        # Return a column with the probability of the sequence belonging to each language
        for lang in self.languages:
            self.predictions[lang] = self.predictions.apply(lambda x: logprob(x['ngrams'], lang, self._probs), axis = 1)

        # Predict label - Name of the column for which log probability is maximised
        self.predictions['lang_pred'] = self.predictions[self.languages].idxmax(axis = 1)

        return self.predictions['lang_pred']
