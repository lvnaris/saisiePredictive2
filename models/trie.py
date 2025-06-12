class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        word = word.lower()
        for char in word:
            if char not in node.children:
                node.children[char] =  TrieNode()
            node = node.children[char]
        node.is_end = True
        node.count += 1

    # lento su grandi corpus
    def _collect(self, node, prefix):
        words = []
        if node.is_end:
            words.append((prefix, node.count))
        for char, child in node.children.items():
            words.extend(self._collect(child, prefix + char))
        return words

    def autocomplete(self, prefix, top_n = 5):
        node = self.root
        prefix = prefix.lower()
        for char in prefix:
            if char not in node.children:
              return []
            node = node.children[char]
        words = self._collect(node,prefix)
        if not words:
            return []
        total_count = sum(count for _ , count in words)
        words_scores = [(word, count / total_count) for word, count in words]
        words_scores.sort(key=lambda x: x[1], reverse = True)
        return words_scores[:top_n]
 