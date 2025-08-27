# Importing the NLTK library
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Sample text
text = "NLTK is a powerful library for natural language processing."

# Tokenize the text
words = word_tokenize(text)

# Performing PoS tagging
pos_tags = pos_tag(words)

# Displaying the PoS tagged result in separate lines
print("Original Text:")
print(text)

print("\nPoS Tagging Result:")
for word, tag in pos_tags:
    print(f"{word}: {tag}")
