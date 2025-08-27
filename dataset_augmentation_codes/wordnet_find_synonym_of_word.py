import nltk 
from nltk.corpus import wordnet 
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import requests
import pandas as pd
import json
import spacy

def replaceSubjectWithSynonym(sentence, subject, synonym):
    if not subject or not synonym:
        return sentence  # Nothing to replace

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    
    new_tokens = []
    replaced = False

    for token in doc:
        # Match based on lowercase for robustness
        if token.text.lower() == subject.lower() and not replaced:
            new_tokens.append(synonym)
            replaced = True
        else:
            new_tokens.append(token.text)

    # Reconstruct the sentence
    return spacy.tokens.Doc(doc.vocab, words=new_tokens).text



def findBestSynonymMatch(givenWord):
    synonyms = [] 
    antonyms = [] 
    frequency = 0
    bestMatch = ""

    for syn in wordnet.synsets(givenWord, pos=wordnet.NOUN): 
        for l in syn.lemmas(): 
            synonyms.append(l.name()) 
            if l.antonyms(): 
                antonyms.append(l.antonyms()[0].name()) 
            if l.count()>=frequency and l.name() != givenWord:
                bestMatch = l.name()
                frequency = l.count()

    # print(set(synonyms)) 
    # print(set(antonyms)) 
    # print(f"The best match for the word: {givenWord} is {bestMatch}, with frequency of {frequency}")

    return bestMatch

def getFirstNN(premise):
    # Sample text
    text = "NLTK is a powerful library for natural language processing."

    # Tokenize the text
    words = word_tokenize(premise)

    # Performing PoS tagging
    pos_tags = pos_tag(words)

    # Displaying the PoS tagged result in separate lines
    # print("Original Text:")
    # print(premise)

    # print("\nPoS Tagging Result:")
    # for word, tag in pos_tags:
    #     print(f"{word}: {tag}")

    for word, tag in pos_tags:
        if tag in ['NN', 'NNS']:
            return word
    return None

def getSubjectOfSentence(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "nsubj":
            print("Subject:", token.text)
            return token.text
            

def substituteWordInPremise(dataset_path, output_file_path):  
    # Fix: Use the correct separator
    data = pd.read_csv(dataset_path, sep=",")
    # Fix: Trim column names to remove leading/trailing spaces
    data.columns = data.columns.str.strip()

    # Print column names for debugging
    print("Columns in dataset:", data.columns)

    #data = data.head(100)  # Limit to 100 samples

    correct = 0
    total = len(data)
    givenWord = ""
    bestSynonym = ""

    with open(output_file_path, 'w') as output_file:
        # Write headers
        output_file.write("Index,Premise,Original Word,Best Synonym\n")
        
        for index, row in data.iterrows():
            premise = row['sentence_A']
            hypothesis = row['sentence_B']
            true_judgment = row['entailment_judgment'].strip().lower()
            # givenWord = getFirstNN(premise)
            givenWord = getSubjectOfSentence(premise)
            # bestSynonym = findBestSynonymMatch(givenWord)
            # print(f"In the premise {index}: {premise} the word: {givenWord} should be substituted with {bestSynonym}")
            # Find the best synonym for the given word
            if givenWord:
                bestSynonym = findBestSynonymMatch(givenWord)
                
                # Write the results to the file
                output_file.write(f"{index},{premise},{givenWord},{bestSynonym}\n")
                # print(f"In the premise {index}: {premise} the word: {givenWord} should be substituted with {bestSynonym}")
                new_sentence = replaceSubjectWithSynonym(premise, givenWord, bestSynonym)
                output_file.write(f"{index},{premise},{"."},{new_sentence}\n")

            else:
                output_file.write(f"{index},{premise},None,None\n")
                # print(f"In the premise {index}: {premise} no noun found")
                # new_sentence = replaceSubjectWithSynonym(premise, givenWord, bestSynonym)
                # output_file.write(f"{index},{premise}\n")

dataset_path = 'neutral_sick_data.txt'  # Adjust the path if necessary
output_file_path = 'found_synonyms.txt'
substituteWordInPremise(dataset_path, output_file_path)
# print(f"Model Accuracy: {accuracy:.2f}")