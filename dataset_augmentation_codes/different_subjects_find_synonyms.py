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


def is_just_plural_form(word, synonym):
    return (
        synonym.lower() == p.singular_noun(word.lower()) or
        p.plural(synonym.lower()) == word.lower()
    )

def findBestSynonymMatch(givenWord):
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

    for synonym in synonyms:
        synonymDoc = nlp(synonym)
        similarity = givenDoc.similarity(synonymDoc)
        if similarity > bestScore:
            bestScore = similarity
            bestMatch = synonym

    return bestMatch


def makeSynonymHashMap(subjects):
    hashMapOfSynonyms = {}
    for subject in subjects:
        bestMatch = findBestSynonymMatch(subject)
        hashMapOfSynonyms[subject] = bestMatch
        # Check if subject is plural
        is_plural = p.singular_noun(subject)

        if bestMatch:
            # If subject is plural and the synonym is singular, pluralize the synonym
            if is_plural:
                bestMatch = p.plural(bestMatch)
        hashMapOfSynonyms[subject] = bestMatch

    return hashMapOfSynonyms


subjects = {'bike', 'guy', 'horses', 'men', 'hiker', 'squirrel', 'balls',
             'cheerleader', 'swimmers', 'she', 'surfer', 'singer', 'turtle', 'footballer', 'kids', 'ball', 
             'stands', 'cheetah', 'daschunds', 'children', 'groups', 'racers', 'herd', 'somebody', 'cyclist', 'crane', 
             'hikers', 'puppy', 'crocodiles', 'deer', 'sheepdog', 'boy', 'animals', 'ladies', 'skateboarder', 'girl', 'boat', 'kid', 
             'cat', 'performer', 'kittens', 'skier', 'women', 'ferrets', 'cheerleaders', 'couple', 'nobody', 'bride', 'motorcyclist', 
             'doctor', 'pandas', 'horse', 'pair', 'lemur', 'cub', 'person', 'racer', 'who', 'jockeys', 'band', 'wrestlers', 'lot', 
             'father', 'rollerbladers', 'gamer', 'blokes', 'teams', 'people', 'bunch', 'child', 'bikers', 'chef', 'rhino', 'cats', 
             'dog', 'snowboarder', 'guys', 'driver', 'games', 'cook', 'rider', 'group', 'dogs', 'reindeer', 'waterfall', 'swimming', 
             'lady', 'jumper', 'schoolgirl', 'potato', 'bells', 'tiger', 'boys', 'adults', 'rink', 'woman', 'poodles', 'family', 
             'runners', 'milk', 'writing', 'athletes', 'fish', 'biker', 'bee', 'machine', 'officer', 'toddlers', 'patient', 
             'someone', 'panda', 'slide', 'duck', 'monkey', 'teammates', 'cars', 'truck', 'baby', 'black', 'man', 'crowd', 
             'adult', 'pedestrians', 'elephant', 'girls', 'workers', 'friends', 'priest', 'gloves', 'water', 'lion', 'players', 
             'blonde', 'climber', 'ringers', 'students', 'interviewer', 'bird', 'football', 'bubble', 'sheep', 'llama', 'plane', 
             'seabird', 'subject', 'cowgirl', 'jet', 'bicyclist', 'player', 'animal', 'male', 'car', 'kitten', 'angels'}
print(f"Number of different subjects: {len(subjects)}")
result = makeSynonymHashMap(subjects)
# for subject, synonym in result.items():
#     print(f"{subject}: {synonym}")
print(f"Result: {result}")