from collections import defaultdict, Counter
import math

# pour entrainement et prediction
class NGram_model:
    def __init__(self, n, top_k=10):
        self.n = n
        self.model = defaultdict(Counter) # mantenere frequenze di n-grammi
        self.start = "<s>"
        self.end = "</s>"
        self.vocab = set()
        self.top_predictions = {}
        self.top_k = top_k

    # applica faux tokens, aggiorna vocabolario e conta n-grammi
    def train(self, sentences):
        print(f"Training {self.n}-gram model...")
        for sentence in sentences:
            # aggiungi token di inizio e fine frase
            padded_sentence = [self.start] * (self.n - 1) + sentence + [self.end] # ripetizione di token iniziali in base al modello
            self.vocab.update(padded_sentence) # aggiorna vocabolario

            # conta tutti i n-grammi
            for i in range(len(padded_sentence) - self.n + 1):
                context = tuple(padded_sentence[i:i + self.n - 1]) # contesto
                next_word = padded_sentence[i + self.n - 1] # parola da predire
                self.model[context][next_word] += 1

        for context, counter in self.model.items():
            if sum(counter.values()) >= self.top_k:
                self.top_predictions[context] = counter.most_common(self.top_k)

    # aggiornamento incrementale
    def update(self, sentence):
        padded_sentence = [self.start] * (self.n - 1) + sentence + [self.end]
        for i in range(len(padded_sentence) - self.n + 1):
            context = tuple(padded_sentence[i:i + self.n - 1])
            next_word = padded_sentence[i + self.n - 1]
            self.model[context][next_word] += 1

    # implementazione dello smoothing di laplace
    def get_laplace_prob(self, context, word):
        vocab_size = len(self.vocab)
        if context not in self.model:
            return 1 / vocab_size
        
        word_count = self.model[context][word]
        context_count = sum(self.model[context].values())
        return (word_count + 1) / (context_count + vocab_size) # Laplace smoothing

    # restituisce le parole più probabili nel contesto dato
    def predict(self, context, top_k=5):
       context = tuple(word.lower() for word in context)
       candidates = self.vocab - {self.start}
       scores = {word: self.get_laplace_prob(context, word) for word in candidates}
       return sorted(scores.items(), key = lambda x: x[1], reverse= True)[:top_k]

# combina tre modelli n-gramma e li combina secondo l'interpolazione    
class NGram_predictor:
    # aggiungi gestione di input troppo corti
    def __init__(self, unigram, bigram, trigram, lambdas=(0.6, 0.3, 0.1)):
        self.models = {1: unigram, 2: bigram, 3: trigram}
        self.lambdas = lambdas
    
    def interpolated_predict(self, w1, w2, top_k=5):
        w1 = w1.lower()
        w2 = w2.lower()
        
        vocab = self.models[1].vocab
        scores = {}

        # Candidati unione delle top parole nei tre modelli
        candidates = set()
        candidates.update([w for w, _ in self.models[3].top_predictions.get((w1, w2), [])])
        candidates.update([w for w, _ in self.models[2].top_predictions.get((w2,), [])])
        candidates.update([w for w, _ in self.models[1].top_predictions.get((), [])])

        for word in candidates:
            if word in {"<s>", "</s>"}:
                continue

            p_tri = self.models[3].get_laplace_prob((w1,w2), word)
            p_bi = self.models[2].get_laplace_prob((w2,), word)
            p_uni = self.models[1].get_laplace_prob((), word)

            score = self.lambdas[0] * p_tri + self.lambdas[1] * p_bi + self.lambdas[2] * p_uni
            scores[word] = score

        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_k]
    
    # implemento interpolazione per coerenza con il modello che uso
    def perplexity(self, test_sentences):
        log_prob = 0
        N = 0

        for sentence in test_sentences:
            padded = ["<s>", "<s>"] + sentence + ["</s>"]
            for i in range(2, len(padded)):
                w1, w2 = padded[i-2], padded[i-1]
                word = padded[i]

                # probabilità con interpolazione
                p_tri = self.models[3].get_laplace_prob((w1, w2), word)
                p_bi = self.models[2].get_laplace_prob((w2,), word)
                p_uni = self.models[1].get_laplace_prob((), word)

                p = (
                    self.lambdas[0] * p_tri +
                    self.lambdas[1] * p_bi +
                    self.lambdas[2] * p_uni
                )

                # evita log(0)
                if p > 0:
                    log_prob += math.log(p)
                    N += 1
        
        if N == 0:
            return float("inf")
        return math.exp(-log_prob / N)
        