from collections import Counter

tweet = "Fire fire near the school"
tokens = tweet.lower().split()
frequencies = Counter(tokens) #Counter conta quantas vezes um token aparece

print("Tokens:", tokens)
print("Frequências:", frequencies)
print("Total de tokens: ", len(tokens))

