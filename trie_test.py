import random
from models.trie import Trie
from utils import preprocess

def test_predict(Trie, test_sentences):
    # per testare se il Trie permette di completare parole a partire da prefisso

    correct = 0
    total = 0 # parole testate

    tokenized_sentences = preprocess(test_sentences)

    for sentence in tokenized_sentences:
        for word in sentence:
            if len(word) < 3: # si ignorano le parole troppo corte
                continue

            prefix_len = 4 # scelta casuale di un prefisso di 2 o 3 lettere
            prefix = word[:prefix_len]
            total += 1

            predictions = Trie.autocomplete(prefix, top_n = 5)
            predicted_words = [w for w, _ in predictions]

            # la parola corretta compare tra i suggerimenti?
            if word.lower() in (w.lower() for w in predicted_words):
                correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Total words tested: {total}")
    print(f"Correctly predicted words: {correct}")
    print(f"Prefix prediction accuracy: {accuracy: .2%}")

