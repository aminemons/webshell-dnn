import re
import numpy as np
from collections import defaultdict, Counter


_PHP_PRETOK = re.compile(r"[\s\(\)\{\}\[\];,]+")
_WHITESPACE_PRETOK = re.compile(r"\s+")
_MAX_TOKEN_LEN = 48
_MIN_TOKEN_FREQ = 2
_MAX_VOCAB_BEFORE_BPE = 30000


def pretokenize_php(text):
    raw = _PHP_PRETOK.split(text)
    return [tok for tok in raw if tok and len(tok) <= _MAX_TOKEN_LEN]


def pretokenize_whitespace(text):
    raw = _WHITESPACE_PRETOK.split(text)
    return [tok for tok in raw if tok and len(tok) <= _MAX_TOKEN_LEN]


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.
    Sennrich et al. 2016 - Neural Machine Translation of Rare Words with Subword Units.
    arXiv:1508.07909

    Optimizations over naive BPE:
    - Pre-token length cap (_MAX_TOKEN_LEN) eliminates huge base64/hex blobs.
    - Minimum frequency filter (_MIN_TOKEN_FREQ) before BPE drastically shrinks vocab.
    - Vocabulary cap (_MAX_VOCAB_BEFORE_BPE) bounds worst-case merge iterations.
    - List-based merge (no regex) avoids re.sub overhead and escape issues.
    - Merge lookup dict for O(1) pair-priority queries during transform.
    """

    def __init__(self, num_merges=1000, vocab_size=256, pretokenizer="php"):
        self.num_merges = num_merges
        self.vocab_size = vocab_size
        self.pretokenizer = pretokenizer
        self.merges = []
        self.vocab = []
        self._merge_rank = {}
        self._vocab_set = {}
        self._merge_cache = {}

    def _pretok(self, text):
        if self.pretokenizer == "php":
            return pretokenize_php(text)
        return pretokenize_whitespace(text)

    def _word_to_chars(self, word):
        chars = list(word)
        if chars:
            chars[-1] = chars[-1] + "</w>"
        return tuple(chars)

    def _build_vocab_freq(self, texts):
        raw_freq = defaultdict(int)
        for text in texts:
            for tok in self._pretok(text):
                raw_freq[tok] += 1

        raw_freq = {w: f for w, f in raw_freq.items() if f >= _MIN_TOKEN_FREQ}
        if len(raw_freq) > _MAX_VOCAB_BEFORE_BPE:
            top = sorted(raw_freq.items(), key=lambda x: -x[1])[:_MAX_VOCAB_BEFORE_BPE]
            raw_freq = dict(top)

        vocab_freq = {}
        for word, freq in raw_freq.items():
            key = self._word_to_chars(word)
            vocab_freq[key] = vocab_freq.get(key, 0) + freq
        return vocab_freq

    def _get_pair_counts(self, vocab_freq):
        pairs = defaultdict(int)
        for word, freq in vocab_freq.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, vocab_freq):
        a, b = pair
        merged = a + b
        new_vocab = {}
        for word_tuple, freq in vocab_freq.items():
            if len(word_tuple) < 2:
                new_vocab[word_tuple] = freq
                continue
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == a and word_tuple[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def fit(self, texts):
        print("    BPE: building initial vocab ...", flush=True)
        vocab_freq = self._build_vocab_freq(texts)
        print(f"    BPE: {len(vocab_freq)} unique pre-token types after filtering", flush=True)
        self.merges = []

        for step in range(self.num_merges):
            pairs = self._get_pair_counts(vocab_freq)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break
            self.merges.append(best_pair)
            vocab_freq = self._merge_vocab(best_pair, vocab_freq)
            if (step + 1) % 100 == 0:
                print(f"    BPE: merge {step+1}/{self.num_merges}", flush=True)

        self._merge_rank = {pair: idx for idx, pair in enumerate(self.merges)}
        print(f"    BPE: {len(self.merges)} merges learned", flush=True)

        print("    BPE: computing document frequencies for vocabulary ...", flush=True)
        doc_freq = defaultdict(int)
        for text in texts:
            seen = set()
            for tok in self._tokenize_text(text):
                if tok not in seen:
                    seen.add(tok)
                    doc_freq[tok] += 1

        sorted_tokens = sorted(doc_freq.items(), key=lambda x: -x[1])
        self.vocab = [tok for tok, _ in sorted_tokens[: self.vocab_size]]
        self._vocab_set = {tok: idx for idx, tok in enumerate(self.vocab)}
        print(f"    BPE: final vocab size = {len(self.vocab)}", flush=True)

    def _apply_merges(self, word_chars):
        cached = self._merge_cache.get(word_chars)
        if cached is not None:
            return cached
        word = list(word_chars)
        while len(word) > 1:
            best_rank = len(self.merges)
            best_idx = -1
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self._merge_rank.get(pair, len(self.merges))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            a, b = word[best_idx], word[best_idx + 1]
            word = word[:best_idx] + [a + b] + word[best_idx + 2:]
        self._merge_cache[word_chars] = word
        return word

    def _tokenize_text(self, text):
        for pretok in self._pretok(text):
            chars = self._word_to_chars(pretok)
            yield from self._apply_merges(chars)

    def transform(self, texts):
        for t in texts:
            yield self._tokenize_text(t)


class WhitespaceTokenizer:
    """Simple whitespace+punctuation tokenizer as baseline."""

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.vocab = []
        self._vocab_set = {}

    def fit(self, texts):
        doc_freq = defaultdict(int)
        for text in texts:
            tokens = set(pretokenize_whitespace(text))
            for tok in tokens:
                doc_freq[tok] += 1
        sorted_tokens = sorted(doc_freq.items(), key=lambda x: -x[1])
        self.vocab = [tok for tok, _ in sorted_tokens[: self.vocab_size]]
        self._vocab_set = {tok: idx for idx, tok in enumerate(self.vocab)}

    def transform(self, texts):
        return [pretokenize_whitespace(t) for t in texts]


class OneHotVectorizer:
    """
    Binary presence one-hot encoding.
    Vocabulary built from training set only (no data leakage).
    Fixed 256-dimensional output.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, texts):
        self.tokenizer.fit(texts)
        return self

    def transform(self, texts):
        vocab_set = self.tokenizer._vocab_set
        n = len(texts)
        d = len(self.tokenizer.vocab)
        X = np.zeros((n, d), dtype=np.float32)
        for i, tok_seq in enumerate(self.tokenizer.transform(texts)):
            for tok in tok_seq:
                idx = vocab_set.get(tok)
                if idx is not None:
                    X[i, idx] = 1.0
        return X

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


class BagOfWordsVectorizer:
    """
    Bag of Words with log(1+count) smoothing and L2 normalization.
    Robertson 2004 - Understanding Inverse Document Frequency.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, texts):
        self.tokenizer.fit(texts)
        return self

    def transform(self, texts):
        vocab_set = self.tokenizer._vocab_set
        n = len(texts)
        d = len(self.tokenizer.vocab)
        X = np.zeros((n, d), dtype=np.float32)
        for i, tok_seq in enumerate(self.tokenizer.transform(texts)):
            counts = Counter(tok_seq)
            for tok, cnt in counts.items():
                idx = vocab_set.get(tok)
                if idx is not None:
                    X[i, idx] = np.log1p(cnt)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return X / norms

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


class TFIDFVectorizer:
    """
    TF-IDF with sublinear TF scaling and smoothed IDF.

    Sublinear TF (Jones 1972):
        tf(t, d) = 1 + log(count(t, d))  if count > 0  else  0

    Smoothed IDF (Sparck Jones 1972, Salton & Buckley 1988):
        idf(t) = log(N / df(t)) + 1

    L2 normalization of final vectors.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.idf = None

    def fit(self, texts):
        self.tokenizer.fit(texts)
        vocab_set = self.tokenizer._vocab_set
        d = len(self.tokenizer.vocab)
        N = len(texts)
        df = np.zeros(d, dtype=np.float64)
        for tok_seq in self.tokenizer.transform(texts):
            seen = set()
            for tok in tok_seq:
                idx = vocab_set.get(tok)
                if idx is not None and idx not in seen:
                    df[idx] += 1
                    seen.add(idx)
        self.idf = np.log(N / np.maximum(df, 1.0)) + 1.0
        return self

    def transform(self, texts):
        vocab_set = self.tokenizer._vocab_set
        n = len(texts)
        d = len(self.tokenizer.vocab)
        X = np.zeros((n, d), dtype=np.float32)
        for i, tok_seq in enumerate(self.tokenizer.transform(texts)):
            counts = Counter(tok_seq)
            for tok, cnt in counts.items():
                idx = vocab_set.get(tok)
                if idx is not None:
                    tf = 1.0 + np.log(cnt) if cnt > 0 else 0.0
                    X[i, idx] = tf * self.idf[idx]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (X / norms).astype(np.float32)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


def build_vectorizers(tokenizer_type="bpe"):
    """Returns (one_hot, bow, tfidf) vectorizer instances sharing a tokenizer."""
    if tokenizer_type == "bpe":
        tok_oh = BPETokenizer(num_merges=1000, vocab_size=256, pretokenizer="php")
        tok_bow = BPETokenizer(num_merges=1000, vocab_size=256, pretokenizer="php")
        tok_tfidf = BPETokenizer(num_merges=1000, vocab_size=256, pretokenizer="php")
    else:
        tok_oh = WhitespaceTokenizer(vocab_size=256)
        tok_bow = WhitespaceTokenizer(vocab_size=256)
        tok_tfidf = WhitespaceTokenizer(vocab_size=256)
    return (
        OneHotVectorizer(tok_oh),
        BagOfWordsVectorizer(tok_bow),
        TFIDFVectorizer(tok_tfidf),
    )


def token_dropout(X, p=0.05, rng=None):
    """Input noise regularization: randomly zero 5% of features during training."""
    if rng is None:
        rng = np.random
    mask = rng.random(X.shape) > p
    return X * mask
