import spacy
import requests
import pandas as pd
import json

def getSubjectOfSentence(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "nsubj":
            return token.text
    return None

def getUniqueSubjects(sentences):
    subjects = set()
    for sentence in sentences:
        subject = getSubjectOfSentence(sentence)
        if subject:
            subjects.add(subject.lower())
    return list(subjects)



def sentencesGen(dataset_path):
    # Fix: Use the correct separator
    data = pd.read_csv(dataset_path, sep=",")
    # Fix: Trim column names to remove leading/trailing spaces
    data.columns = data.columns.str.strip()
    subjects = set()
    for index, row in data.iterrows():
            premise = row['sentence_A']
            hypothesis = row['sentence_B']
            true_judgment = row['entailment_judgment'].strip().lower()
            subject = getSubjectOfSentence(premise)
            if subject:
                 subjects.add(subject.lower())
            print(f"Nr: {index}")
    return subjects



# unique_subjects = getUniqueSubjects(sentences)
# print("Unique subjects (case-insensitive):", unique_subjects)

dataset_path = '../neutral_sick_data.txt'  # Adjust the path if necessary
output_file_path = '../test_directory.txt'
subjects = sentencesGen(dataset_path)
print(f"Different subjects: {subjects}")
