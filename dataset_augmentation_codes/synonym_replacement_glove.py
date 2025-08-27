from gensim.models import KeyedVectors
fname = "glove.6B.300d.txt"
glove = KeyedVectors.load_word2vec_format(fname,no_header=True)
print(glove.vectors.shape)
# common noun
print(glove.most_similar("cactus"))
print(glove.most_similar("cactus")[0][0])

def findSynonyms(subjects):
    synonymMap = {}
    for subject in subjects:
        if subject in glove:  #check if word exists
            synonyms = glove.most_similar(subject)     
            if synonyms:
                bestsyn = synonyms[1][0]
                synonymMap[subject] = bestsyn
            else:
                synonymMap[subject] = ""
        else:
            synonymMap[subject] = ""  #word not in vocabulary
    return synonymMap



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

synonyms = findSynonyms(subjects)
print(synonyms)