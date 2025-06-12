from sklearn.model_selection import train_test_split
import sys

from utils import load_dataset, preprocess
from models.trie import Trie
from models.ngram import NGram_model, NGram_predictor
from interface import interface_graphique

#from trie_test import test_predict

# Carica dataset
file_path = "C:\\Users\\yangy\\Desktop\\saisiePredictiveTest\\88milSMS_88522.xlsx"
print("Chargement du dataset...")
df = load_dataset(file_path)

# Split dataset
train_df, test_df = train_test_split(df, random_state=42, test_size=0.1)
testset = test_df["SMS_ANON"].astype(str).tolist()
trainset = train_df["SMS_ANON"].astype(str).tolist()

# Preprocessing
print("Prétraitement des données...")
train_sentences = preprocess(trainset)
test_sentences = preprocess(testset)

# Afficher correctement les accents
sys.stdout.reconfigure(encoding='utf-8') # supporto dei caratteri speciali

# Flat list di token
token_list = [token for sublist in train_sentences for token in sublist]
  
# 2. Building the Trie
print("Construction du Trie...")

trie = Trie()
for token in token_list:
  trie.insert(token)

# Construction des modèles n-grammes
print("Apprentissage des modèles n-grammes...")

uni = NGram_model(1)
bi = NGram_model(2)
tri = NGram_model(3)

uni.train(train_sentences)
bi.train(train_sentences)
tri.train(train_sentences)

# Création du prédicteur
predictor = NGram_predictor(uni, bi, tri)

#Lancement de l'interface interactive
print("Lancement interface...")
interface_graphique(trie, predictor)


"""
def main():
  # 1. Caricamento e preprocessing del dataset
  file_path = "C:\\Users\\yangy\\Desktop\\saisiePredictiveTest\\88milSMS_88522.xlsx"
  #file_path = "88milSMS_88522.xlsx"

  print("Chargement du dataset...")
  df = load_dataset(file_path)

  # Split dataset and preprocess
  train_df, test_df = train_test_split(df, random_state=42, test_size=0.1)
  testset = test_df["SMS_ANON"].astype(str).tolist()
  trainset = train_df["SMS_ANON"].astype(str).tolist()
  
  print("Prétraitement des données...")

  train_sentences = preprocess(trainset)
  test_sentences = preprocess(testset)
  
  # Afficher correctement les accents
  sys.stdout.reconfigure(encoding='utf-8') # supporto dei caratteri speciali
  
  # Flat list di token
  token_list = [token for sublist in train_sentences for token in sublist]
  
  # 2. Building the Trie
  print("Construction du Trie...")

  trie = Trie()
  for token in token_list:
    trie.insert(token)

  # Construction des modèles n-grammes
  print("Apprentissage des modèles n-grammes...")
  uni = NGram_model(1)
  bi = NGram_model(2)
  tri = NGram_model(3)

  uni.train(train_sentences)
  bi.train(train_sentences)
  tri.train(train_sentences)

  # Création du prédicteur
  predictor = NGram_predictor(uni, bi, tri)

  #Lancement de l'interface interactive
  print("Lancement interface (exit pour quitter)...")
  interface_graphique(trie, uni, bi, tri)

# convenzione per eseguire un file come script principale
if __name__ == "__main__":
  main()
"""