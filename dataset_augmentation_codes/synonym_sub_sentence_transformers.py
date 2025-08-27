from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet as wn
import nltk 
from nltk.corpus import wordnet 
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import requests
import pandas as pd
import json
import spacy
nlp = spacy.load('en_core_web_md')
import inflect
from nltk.corpus import wordnet

p = inflect.engine()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def find_subject_of_sentence(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "nsubj":
            return token.text
    return None

def is_just_plural_form(word, synonym):
    return (
        synonym.lower() == p.singular_noun(word.lower()) or
        p.plural(synonym.lower()) == word.lower()
    )


def find_synonyms_of_word(givenWord):
    synonyms = set()
    bestMatch = ""
    bestScore = 0

    givenDoc = nlp(givenWord.replace("_", " "))

    for syn in wordnet.synsets(givenWord, pos=wordnet.NOUN):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() == givenWord.lower():
                continue
            if is_just_plural_form(givenWord, name):
                continue
            if name == givenWord or name.istitle() or '_' in name:
                continue
            synonyms.add(name)
    return synonyms


def embedding_nearest_neighbors(sentence, word, top_k=10):
    # Get embedding for the query word
    query_emb = model.encode(word, convert_to_tensor=True)
    sent_emb = model.encode(sentence, convert_to_tensor=True)

    # Candidate vocab from WordNet nouns
    candidates = set()
    for syn in wn.synsets(word, pos=wn.NOUN):
        for lemma in syn.lemmas():
            candidates.add(lemma.name().replace("_", " "))

        # also explore hypernyms and hyponyms for more coverage
        for hyper in syn.hypernyms() + syn.hyponyms():
            for lemma in hyper.lemmas():
                candidates.add(lemma.name().replace("_", " "))

    candidates = list(candidates)
    results = []
    for candidate in candidates:
        if candidate.lower() == word.lower() or word in candidate:
            continue
        cand_sent = sentence.replace(word, candidate)
        cand_emb = model.encode(cand_sent, convert_to_tensor=True)
        # similarity = cosine_similarity(sent_emb, cand_emb)
        similarity = cosine_similarity(sent_emb.reshape(1, -1), cand_emb.reshape(1, -1))[0][0]
        # Store (candidate, similarity)
        results.append((cand_sent, similarity))

    # Sort by similarity descending
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    return results


def find_best_match_with_cosine(sentence):
    subject = find_subject_of_sentence(sentence)
    if not subject:
        print("No subject found")
        return None
    synonyms = find_synonyms_of_word(subject)
    embedding_original = model.encode(sentence)
    print("Original sentence:", sentence)
    print("Substituted sentences:")
    if not synonyms:
        print("No synonym found")
        return None
    for synonym, score in synonyms:
        if synonym == subject:
            continue
        new_sentence = sentence.replace(subject, synonym)
        print(new_sentence)
        new_embedding = model.encode(new_sentence)
        
        similarity = cosine_similarity([embedding_original], [new_embedding])[0][0]
        print("Similarity: ", similarity)

    return synonyms

def find_best_in_context(sentence):
    subject = find_subject_of_sentence(sentence)
    if not subject:
        print("Original sentence:", sentence)
        print("No subject found")
        return None
    results = embedding_nearest_neighbors(sentence, subject, 3)
    print("Original sentence: ", sentence)
    print("Candidates:")
    for candidate, similarity in results:
        print(candidate, similarity)


# sentence = "Two people are kickboxing and spectators are not watching."
# sentence = "A cat is near the red ball in the air"
sentence = "There is no biker jumping in the air"
# synonyms1 = find_best_match_with_cosine(sentence)
find_best_in_context(sentence)

# # Example usage
# neighbors = embedding_nearest_neighbors("people", top_k=10)
# for word, score in neighbors:
#     print(f"{word} ({score:.4f})")

dataset_path = '../data_files/contradiction_sick_data.txt'  # Adjust the path if necessary
# Fix: Use the correct separator
data = pd.read_csv(dataset_path, sep=",")
data = data.head(100)
# Fix: Trim column names to remove leading/trailing spaces
data.columns = data.columns.str.strip()
for index, row in data.iterrows():
    premise = row['sentence_A']
    hypothesis = row['sentence_B']
    true_judgment = row['entailment_judgment'].strip().lower()
    find_best_in_context(premise)