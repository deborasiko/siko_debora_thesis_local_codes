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

def findClosestHypernym(word):
    # Look for the closest (distance=1) hypernym
    for syn in wordnet.synsets(word, pos=wordnet.NOUN):
        for hypernym, dist in syn.hypernym_distances():
            if dist == 1:
                return hypernym.lemmas()[0].name().replace("_", " ")
    return None

def makeHypernymMap(subjects):
    hypernymMap = {}
    for subject in subjects:
        #checkking if original word is plural
        singular_form = p.singular_noun(subject)
        is_plural = bool(singular_form)
        base_word = singular_form if is_plural else subject
        
        #gett hypernym for singular form
        hypernym = findClosestHypernym(base_word)

        if hypernym:
            hypernym = p.plural(hypernym) if is_plural else hypernym

        hypernymMap[subject] = hypernym
    return hypernymMap



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
result = makeHypernymMap(subjects)
# for subject, synonym in result.items():
#     print(f"{subject}: {synonym}")
print(f"Result: {result}")