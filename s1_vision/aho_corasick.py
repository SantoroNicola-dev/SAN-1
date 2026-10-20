"""
High-Performance Aho-Corasick String Matching Automaton
======================================================
Provides O(N + M) simultaneous multi-pattern search across text.
Zero-dependency, pure Python, microsecond-level latency.
"""

from collections import deque
from typing import List, Tuple, Dict, Set, Optional


class AhoCorasickAutomaton:
    """
    Trie-based Aho-Corasick Automaton for simultaneous multi-keyword matching.
    """

    def __init__(self, keywords: Optional[List[str]] = None):
        self.goto: List[Dict[str, int]] = [{}]
        self.fail: List[int] = [0]
        self.output: List[List[str]] = [[]]
        if keywords:
            self.build_from_keywords(keywords)

    def add_word(self, word: str):
        if not word:
            return
        curr = 0
        for char in word:
            if char not in self.goto[curr]:
                nxt = len(self.goto)
                self.goto[curr][char] = nxt
                self.goto.append({})
                self.fail.append(0)
                self.output.append([])
            curr = self.goto[curr][char]
        if word not in self.output[curr]:
            self.output[curr].append(word)

    def build_from_keywords(self, keywords: List[str]):
        """Populates and compiles the failure transitions."""
        for kw in keywords:
            self.add_word(kw)

        queue = deque()
        for char, nxt in self.goto[0].items():
            queue.append(nxt)
            self.fail[nxt] = 0

        while queue:
            curr = queue.popleft()
            for char, nxt in self.goto[curr].items():
                queue.append(nxt)
                f = self.fail[curr]
                while f > 0 and char not in self.goto[f]:
                    f = self.fail[f]
                self.fail[nxt] = self.goto[f].get(char, 0)
                self.output[nxt].extend(self.output[self.fail[nxt]])

    def search(self, text: str, word_boundary: bool = False) -> List[Tuple[int, int, str]]:
        """
        Scans text for all occurrences of registered keywords.
        Returns list of (start_idx, end_idx, keyword).
        If word_boundary is True, only matches bounded by non-alphanumeric chars.
        """
        matches = []
        curr = 0
        text_len = len(text)

        for i, char in enumerate(text):
            while curr > 0 and char not in self.goto[curr]:
                curr = self.fail[curr]
            curr = self.goto[curr].get(char, 0)
            if self.output[curr]:
                for word in self.output[curr]:
                    start_idx = i - len(word) + 1
                    end_idx = i
                    if word_boundary:
                        # Check left boundary
                        if start_idx > 0 and text[start_idx - 1].isalnum():
                            continue
                        # Check right boundary
                        if end_idx + 1 < text_len and text[end_idx + 1].isalnum():
                            continue
                    matches.append((start_idx, end_idx, word))
        return matches
