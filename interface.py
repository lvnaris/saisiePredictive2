from utils import preprocess
from models.trie import Trie
#from models.ngram import(interpolated_predict, predict)

def interface_graphique(trie, predictor):
  print("\nTapez votre phrase: ('exit ou Crtl+C' pour quitter)\n")

  while True:
    try:
      user_input = input(">").strip()
      if user_input.lower() == "exit":
        print("Au revoir!")
        break
      
      if not user_input:
        continue
      
      words = user_input.lower().split()
      if not words:
        continue

      # Autocomplete: ultima parola
      prefix = words[-1]

      # parole precedenti
      prec_tokens = preprocess([" ".join(words[:-1])])[0] if len(words) > 1 else []

      completions = trie.autocomplete(prefix, top_n=5)

      if completions:
        print("Completions:", [w for w, _ in completions])
      else:
        print("No completions")

      if len(prec_tokens) >= 2:
        preds = predictor.interpolated_predict(prec_tokens[-2], prec_tokens[-1])
      elif len(prec_tokens) == 1:
        preds = predictor.interpolated_predict("", prec_tokens[-1])
      else:
        preds = []

      if preds:
        print("Predictions:", [w for w, _ in preds])
      else:
        print("No predictions available.")

    except KeyboardInterrupt:
      print("\nAu revoir!")
      break

# Interfaccia solo per NgramPredictor interpolato    
def interactive_prediction(model):
  print("\nTapez votre phrase: ('exit' pour quitter)\n")

  while True:
      user_input = input(">").strip()
      if user_input.lower() == "exit":
        print("Au revoir!")
        break
      
      if not user_input:
        continue
      
      words = user_input.lower().split()
      if not words:
        continue
      
      if len(words) >= 2:
        w1, w2 = words[-2], words[-1]
        predictions = model.interpolated_predict(w1,w2, top_k = 5)
      elif len(words) == 1:
        w2 = words[-1]
        predictions = model.interpolated_predict("", w2, top_k=5)
      else:
        print("Tapez au moins un mot")
        continue

      if predictions:
        contexte = ' '.join(words[-2:]) if len(words) >= 2 else words[-1]
        print(f"\nAvec le contexte: {contexte}")
        for i, (word, prob) in enumerate(predictions, 1):
          print(f"{i}.{word} (p = {prob:.4f})")
      else:
        print("Pas de predictions")
