from utils import preprocess
from models.ngram import NGram_model, NGram_predictor

def test_prediction(predictor, test_sentences, top_k=5):
    correct = 0
    total = 0

    for sentence in test_sentences:
        padded = ["<s>", "<s>"] + sentence + ["</s>"]
        for i in range(2, len(padded) - 1):
            w1, w2 = padded[i-2], padded[i-1]
            true_next = padded[i]
            predictions = predictor.interpolated_predict(w1,w2,top_k = 10)
            predicted_words = [w for w, _ in predictions]
            total += 1

            if true_next in predicted_words:
                correct += 1

            accuracy = correct / total if total > 0 else 0
            print(f"Test prédiction: {total} cas testés")
            print(f"Correctement prédits: {correct}")
            print(f"Précision prédiction: {accuracy: .2%}")
            