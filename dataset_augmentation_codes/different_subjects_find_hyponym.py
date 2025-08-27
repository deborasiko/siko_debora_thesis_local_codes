from nltk.corpus import wordnet
import inflect

p = inflect.engine()

def findClosestHyponym(word):
    synsets = wordnet.synsets(word, pos=wordnet.NOUN)
    if synsets:
        first_synset = synsets[0]
        hyponyms = first_synset.hyponyms()
        if hyponyms:
            # Take the first hyponym and extract its lemma name
            return hyponyms[0].lemmas()[0].name().replace("_", " ")
    return None

def makeHyponymMap(subjects):
    hyponymMap = {}
    for subject in subjects:
        # Determine if the subject is plural
        singular_form = p.singular_noun(subject)
        is_plural = bool(singular_form)
        base_word = singular_form if is_plural else subject

        hyponym = findClosestHyponym(base_word)

        if hyponym:
            # Re-pluralize if original subject was plural
            hyponym = p.plural(hyponym) if is_plural else hyponym

        hyponymMap[subject] = hyponym
    return hyponymMap

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
result = makeHyponymMap(subjects)
# for subject, synonym in result.items():
#     print(f"{subject}: {synonym}")
print(f"Result: {result}")