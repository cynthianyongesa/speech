#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import spacy
import textstat
import nltk
import math
from collections import Counter
from spacy.lang.en import English
from datetime import datetime
from spacytextblob.spacytextblob import SpacyTextBlob
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.corpus import words
from spacy.tokens import Doc
from textblob import TextBlob

data = pd.read_excel('OPEN_AI.xlsx')
original_data = data.copy()


# Register custom extensions for polarity and subjectivity
def polarity(doc):
    return TextBlob(doc.text).sentiment.polarity

def subjectivity(doc):
    return TextBlob(doc.text).sentiment.subjectivity

Doc.set_extension("polarity", getter=polarity)
Doc.set_extension("subjectivity", getter=subjectivity)

#A total of 42 variables have been examined, including some novel 
#factors and the linguistic/stylometric indices suggested in the 
#literature.

# Function to calculate lexical diversity
def calculate_lexical_diversity(doc):
    """
    Calculate the lexical diversity of a document.

    Parameters:
    doc (spacy.tokens.Doc): The processed document.

    Returns:
    float: The lexical diversity (unique lemmas / total tokens).
    """
    unique_words = {token.lemma_ for token in doc if token.is_alpha}
    return len(unique_words) / len(doc) if len(doc) > 0 else 0

# Function to tokenize and extract POS, sentiment, and lexical diversity
def tokenize_and_pos(text):
    """
    Tokenize text and extract POS tags, sentiment polarity, subjectivity, and lexical diversity.

    Parameters:
    text (str): The input text.

    Returns:
    tuple: POS-tagged data, polarity, subjectivity, lexical diversity.
    """
    if pd.isna(text):
        return [], None, None, None
    
    doc = nlp(text)
    
    polarity = doc._.polarity
    subjectivity = doc._.subjectivity
    lexical_diversity = calculate_lexical_diversity(doc)
    
    pos_tagged_data = [(token.text, token.pos_) for token in doc]

    return pos_tagged_data, polarity, subjectivity, lexical_diversity

# Function to count words in a text
def word_count(text):
    """
    Count the number of words in a text.

    Parameters:
    text (str): The input text.

    Returns:
    int: The total word count.
    """
    if isinstance(text, str):
        words = text.split()
        total_words = len(words)
        return total_words
    else:
        return 0

# Function to extract POS counts
def extract_counts(pos_tags):
    """
    Extract counts of various POS tags from a list of POS-tagged tokens.

    Parameters:
    pos_tags (list): A list of tuples containing token text and POS tag.

    Returns:
    dict: A dictionary with counts of different POS tags and calculated metrics.
    """
    pos_tags = [tag[1] for tag in pos_tags if len(tag) == 2]  # Ensure each item has two elements
    pos_counts = Counter(pos_tags)
    
    counts = {
        'verbs': pos_counts.get('VERB', 0),
        'nouns': pos_counts.get('NOUN', 0),
        'pronouns': pos_counts.get('PRON', 0),
        'adjectives': pos_counts.get('ADJ', 0),
        'adverbs': pos_counts.get('ADV', 0),
        'interjections': pos_counts.get('INTJ', 0),
        'determiners': pos_counts.get('DET', 0),
        'conjunctions': pos_counts.get('CCONJ', 0),
        'prepositions': pos_counts.get('ADP', 0),
        'auxiliary_verbs': pos_counts.get('AUX', 0),
        'particles': pos_counts.get('PART', 0),
        'numbers': pos_counts.get('NUM', 0)
    }

    total_counts = sum(counts.values())
    ocw = counts['verbs'] + counts['nouns'] + counts['adjectives'] + counts['adverbs']
    ccw = total_counts - ocw
    content_density = ocw / ccw if ccw != 0 else float('nan')
    
    counts['total_counts'] = total_counts
    counts['ocw'] = ocw
    counts['ccw'] = ccw
    counts['content_density'] = content_density

    return counts

# Function to count disfluencies
def count_disfluencies(text):
    """
    Count the number of disfluencies in a text.

    Parameters:
    text (str): The input text.

    Returns:
    int: The total disfluency count.
    """
    if pd.notna(text):
        disfluencies = ['uh', 'uhh', 'um', 'umm', 'oh', 'ohh', 'hm', 'hmm', 'er', 'erm', 'well', 'you know',
                        'like', 'so', 'actually', 'basically', 'i mean', 'alright', 'okay', 'right']
        text_lower = text.lower()
        return sum(text_lower.count(disfluency) for disfluency in disfluencies)
    else:
        return 0

# Function to calculate POS rates
def pos_rates(data, columns):
    """
    Calculate the rate of each POS category in the dataset.

    Parameters:
    data (pd.DataFrame): The input data.
    columns (list): The list of columns containing POS counts.

    Raises:
    ValueError: If the 'Total_Counts' column is not present in the columns list.
    """
    if 'Total_Counts' not in columns:
        raise ValueError("The 'Total_Counts' column is required in the 'columns' list.")
    
    total_counts_column = 'Total_Counts'
    
    for pos_category in columns:
        if pos_category != total_counts_column:
            data[pos_category + '_Rate'] = data[pos_category] / data[total_counts_column]

# Function to calculate readability metrics
def calculate_readability(text):
    """
    Calculate various readability metrics for a text.

    Parameters:
    text (str): The input text.

    Returns:
    tuple: Various readability metrics including Dale-Chall, Flesch Reading Ease, Coleman-Liau Index, Automated Readability Index, reading time, and syllable count.
    """
    if not isinstance(text, str): 
        text = str(text)
    
    dale_chall = textstat.dale_chall_readability_score(text)
    flesch = textstat.flesch_reading_ease(text)
    coleman_liau_index = textstat.coleman_liau_index(text)
    automated_readability_index = textstat.automated_readability_index(text)
    r_time = textstat.reading_time(text, ms_per_char=14.69)
    syllables = textstat.syllable_count(text)

    return dale_chall, flesch, coleman_liau_index, automated_readability_index, r_time, syllables

# Function to calculate deixis rates
def calculate_deixis_rates(text_column):
    """
    Calculate the rates of personal, spatial, and temporal deixis in the texts.

    Parameters:
    text_column (pd.Series): The column containing the input texts.

    Returns:
    tuple: Lists of rates for personal, spatial, and temporal deixis.
    """
    p_deixis_rates = []
    s_deixis_rates = []
    t_deixis_rates = []

    p_deixis_tags = ["PRP", "PRP$", "WP", "WP$"]
    s_deixis_tags = ["LOC", "GPE", "FAC", "ORG"]
    t_deixis_tags = ["DATE", "TIME"]

    for text in text_column:
        if isinstance(text, str):
            doc = nlp(text)
            p_deixis_count = sum(1 for token in doc if token.tag_ in p_deixis_tags)
            s_deixis_count = sum(1 for token in doc if token.ent_type_ in s_deixis_tags)
            t_deixis_count = sum(1 for token in doc if token.ent_type_ in t_deixis_tags)

            total_words = len(doc)
            p_deixis_rate = p_deixis_count / total_words if total_words > 0 else 0
            s_deixis_rate = s_deixis_count / total_words if total_words > 0 else 0
            t_deixis_rate = t_deixis_count / total_words if total_words > 0 else 0

            p_deixis_rates.append(p_deixis_rate)
            s_deixis_rates.append(s_deixis_rate)
            t_deixis_rates.append(t_deixis_rate)
        else:
            p_deixis_rates.append(0)
            s_deixis_rates.append(0)
            t_deixis_rates.append(0)

    return p_deixis_rates, s_deixis_rates, t_deixis_rates

# Function to calculate lexical richness metrics
def calculate_lexical_richness(text):
    """
    Calculate lexical richness metrics for a text.

    Parameters:
    text (str): The input text.

    Returns:
    tuple: Type-Token Ratio (TTR), Brunet's Index, and Honoré's Statistic.
    """
    if isinstance(text, float):
        text = str(text)

    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    ttr = len(set(filtered_tokens)) / len(filtered_tokens) if len(filtered_tokens) > 0 else 0
    v = len(set(filtered_tokens))
    n = len(filtered_tokens)
    brunets_index = v / (n ** 0.165) if n > 0 else 0

    fdist = FreqDist(filtered_tokens)
    r1 = len(fdist.hapaxes())
    r = len([word for word in fdist if fdist[word] == 1])
    honor_statistic = 100 * (n * (np.log(n) - np.log(r1))) if r1 > 0 else 0

    return ttr, brunets_index, honor_statistic

# Function to calculate POS rates
def calculate_pos_rates(data, columns):
    """
    Calculate the rate of each POS category in the dataset.

    Parameters:
    data (pd.DataFrame): The input data.
    columns (list): The list of columns containing POS counts.

    Raises:
    ValueError: If the 'total_counts' column is not present in the columns list.
    """
    if 'total_counts' not in columns:
        raise ValueError("The 'total_counts' column is required in the 'columns' list.")
    
    for pos_category in columns:
        if pos_category != 'total_counts':
            data[pos_category + '_Rate'] = data[pos_category] / data['total_counts']

# Function to calculate mean and standard deviation of sentence length
def sentence_lengths(text):
    if not isinstance(text, str):
        return 0, 0
    
    sentences = sent_tokenize(text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    mean_length = np.mean(sentence_lengths) if sentence_lengths else 0
    std_length = np.std(sentence_lengths) if sentence_lengths else 0
    
    return mean_length, std_length

# Function to calculate lexical density
def lexical_density(text):
    if isinstance(text, float):
        text = str(text)

    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    content_words = [word for word in tokens if word.isalpha() and word not in stop_words]

    return len(content_words) / len(tokens) if tokens else 0

# Function to count the number of sentences
def count_sentences(text):
    if not isinstance(text, str):
        return 0

    sentences = sent_tokenize(text)
    return len(sentences)

# Function to count function words
def count_function_words(text):
    if isinstance(text, float):
        text = str(text)

    tokens = word_tokenize(text.lower())
    function_words = [word for word in tokens if word in stopwords.words("english")]
    
    return len(function_words)
# Function to calculate mean and standard deviation of sentence length
def sentence_lengths(text):
    """
    Calculate the mean and standard deviation of sentence lengths in a text.

    Parameters:
    text (str): The input text.

    Returns:
    tuple: Mean sentence length and standard deviation of sentence length.
    """
    if not isinstance(text, str):
        return 0, 0
    
    sentences = sent_tokenize(text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    mean_length = np.mean(sentence_lengths) if sentence_lengths else 0
    std_length = np.std(sentence_lengths) if sentence_lengths else 0
    
    return mean_length, std_length

# Function to calculate lexical density
def lexical_density(text):
    """
    Calculate the lexical density of a text.

    Parameters:
    text (str): The input text.

    Returns:
    float: The lexical density (content words / total words).
    """
    if isinstance(text, float):
        text = str(text)

    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    content_words = [word for word in tokens if word.isalpha() and word not in stop_words]

    return len(content_words) / len(tokens) if tokens else 0

# Function to count the number of sentences
def count_sentences(text):
    """
    Count the number of sentences in a text.

    Parameters:
    text (str): The input text.

    Returns:
    int: The total number of sentences.
    """
    if not isinstance(text, str):
        return 0

    sentences = sent_tokenize(text)
    return len(sentences)

# Function to count function words
def count_function_words(text):
    """
    Count the number of function words in a text.

    Parameters:
    text (str): The input text.

    Returns:
    int: The total count of function words.
    """
    if isinstance(text, float):
        text = str(text)

    tokens = word_tokenize(text.lower())
    function_words = [word for word in tokens if word in stopwords.words("english")]
    
    return len(function_words)

# Tokenize and extract POS tags, sentiment, and lexical diversity
tokenized_sentiments = data['Participant_Text'].apply(tokenize_and_pos)
data[['POS_Tagged', 'Polarity', 'Subjectivity', 'Lexical_Diversity']] = pd.DataFrame(tokenized_sentiments.tolist())

# Extract POS counts
pos_counts = data['POS_Tagged'].apply(extract_counts)
pos_counts_df = pd.DataFrame(pos_counts.tolist())
data = pd.concat([data, pos_counts_df], axis=1)

# Calculate deixis rates
p_rates, s_rates, t_rates = calculate_deixis_rates(data['Participant_Text'])
data['Personal_Deixis_Rate'] = p_rates
data['Spatial_Deixis_Rate'] = s_rates
data['Temporal_Deixis_Rate'] = t_rates

# Calculate lexical richness
lexical_richness = data['Participant_Text'].apply(calculate_lexical_richness)
data[['TTR', 'Brunets_Index', 'Honors_Statistic']] = pd.DataFrame(lexical_richness.tolist())

# Calculate readability metrics
readability_metrics = data['Participant_Text'].apply(calculate_readability)
data[['Dale_Chall', 'Flesch', 'Coleman_Liau_Index', 'Automated_Readability_Index', 'Reading_Time', 'Syllables']] = pd.DataFrame(readability_metrics.tolist())

# Calculate additional metrics
data['Word_Count'] = data['Participant_Text'].apply(word_count)
data['Disfluencies'] = data['Participant_Text'].apply(count_disfluencies)
data['Syntactic_Complexity'] = (2 * data['conjunctions'] + 2 * data['pronouns'] + data['nouns'] + data['verbs']) / data['total_counts']
data['RefRReal'] = data['nouns'] / data['verbs']

# Calculate new variables
mean_sentence_lengths, std_sentence_lengths = zip(*data['Participant_Text'].apply(sentence_lengths))
data['Mean_Sentence_Length'] = mean_sentence_lengths
data['Std_Sentence_Length'] = std_sentence_lengths
data['Lexical_Density'] = data['Participant_Text'].apply(lexical_density)
data['Number_of_Sentences'] = data['Participant_Text'].apply(count_sentences)
data['Function_Word_Count'] = data['Participant_Text'].apply(count_function_words)

# Ensure the 'total_counts' column exists before calling calculate_pos_rates
if 'total_counts' in data.columns:
    calculate_pos_rates(data, ['verbs', 'nouns', 'pronouns', 'adjectives', 'adverbs', 'interjections', 'determiners', 'conjunctions', 'prepositions', 'auxiliary_verbs', 'particles', 'numbers', 'total_counts'])
else:
    print("The 'total_counts' column is missing from the data.")

# Display the first few rows of the processed dataset
print(data.head())

# Create a new DataFrame from the result
data.to_excel('NEW_DATA_NLP_May_19.xlsx', index = False)

