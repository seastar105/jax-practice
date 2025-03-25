import concurrent.futures
import gc
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import regex as re
from cs336_basics.utils import gpt2_bytes_to_unicode
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger.setLevel(logging.INFO)


@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{name} took {end - start:.4f}s")


class Node:
    """Node class for doubly linked list"""

    def __init__(self, value: int, freq: int, prev: "Node" = None, next: "Node" = None):
        self.value = value
        self.freq = freq

        self.prev = prev
        self.next = next

    def __repr__(self):
        if self.prev:
            prev = self.prev.value
        else:
            prev = "empty"
        if self.next:
            next = self.next.value
        else:
            next = "empty"
        return f"Node(prev={prev}, value={self.value}, next={next})"

    def delete(self):
        # delete this node, update prev and next
        if self.prev:
            self.prev.next = self.next
        if self.next:
            self.next.prev = self.prev

        self.prev = None
        self.next = None


class PriorityQueue:
    def __init__(self):
        self.heap: List[Tuple[Tuple[int, int], int]] = [((0, 0), 0)]  # 1-indexed heap
        self.index = dict()

    def swap(self, i: int, j: int):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.index[self.heap[i][0]] = i
        self.index[self.heap[j][0]] = j

    def upward(self, idx: int):
        while idx > 1:
            parent = idx // 2
            if self.heap[parent][1] < self.heap[idx][1]:
                self.swap(parent, idx)
                idx = parent
            else:
                break

    def downward(self, idx: int):
        while 2 * idx < len(self.heap):
            left = 2 * idx
            right = 2 * idx + 1
            if right < len(self.heap) and self.heap[right][1] > self.heap[left][1]:
                child = right
            else:
                child = left
            if self.heap[child][1] > self.heap[idx][1]:
                self.swap(child, idx)
                idx = child
            else:
                break

    def get_max(self, vocab):
        max_freq = self.top()[1]
        candidates = []

        while self.top()[1] == max_freq:
            candidates.append(self.heap[1])
            self.pop()

        max_pair = max(candidates, key=lambda x: (vocab[x[0][0]], vocab[x[0][1]]))[0]

        # repush
        for pair, freq in candidates:
            if pair != max_pair:
                self.push(pair, freq)
        return max_pair

    def pop(self):
        pair = self.heap[1][0]
        self.delete(pair)

    def top(self):
        assert len(self.heap) > 1
        return self.heap[1]

    def push(self, pair: Tuple[int, int], freq: int):
        self.heap.append((pair, freq))
        self.index[pair] = len(self.heap) - 1
        self.upward(len(self.heap) - 1)

    def delete(self, pair: Tuple[int, int]):
        assert pair in self.index
        idx = self.index[pair]

        self.swap(idx, len(self.heap) - 1)
        self.heap.pop()
        del self.index[pair]

        if idx < len(self.heap):
            self.upward(idx)
            self.downward(idx)

    def update(self, pair: Tuple[int, int], diff: int):
        if pair in self.index:
            idx = self.index[pair]
            new_freq = self.heap[idx][1] + diff
            assert new_freq >= 0, f"new_freq: {new_freq}, diff: {diff}, pair: {pair}, idx: {idx}"
            if new_freq == 0:
                self.delete(pair)
            else:
                self.heap[idx] = (pair, new_freq)
                self.upward(idx)
                self.downward(idx)
        else:
            if diff:
                self.push(pair, diff)

    def size(self):
        return len(self.heap) - 1

    def print(self, vocab):
        for pair, freq in self.heap[1:]:
            print(f"{vocab[pair[0]]} + {vocab[pair[1]]}: {freq}")


class Tokenizer:
    compiled_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def __init__(
        self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None
    ):
        self.vocab = vocab
        self.merges = []
        rev_vocab = {v: k for k, v in vocab.items()}
        for c1, c2 in merges:
            pair = (rev_vocab[c1], rev_vocab[c2])
            new_idx = rev_vocab[c1 + c2]
            self.merges.append((pair, new_idx))
        self.merges_dict = {pair: new_idx for pair, new_idx in self.merges}

        self.special_tokens = special_tokens or []
        self.special_tokens_map = {token: rev_vocab[token.encode("utf-8")] for token in special_tokens}
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(token) for token in sorted_tokens) + ")"
            self.special_regex = re.compile(pattern)
        else:
            self.special_regex = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] | None = None):
        unicode_to_bytes = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath) as f:
            unicode_vocab = json.load(f)
        vocab = dict()
        for idx, token in unicode_vocab.items():
            idx = int(idx)
            vocab[idx] = bytes([unicode_to_bytes[c] for c in token])

        merges = []
        with open(merges_filepath) as f:
            for line in f:
                c1, c2 = line.strip().split(" ")
                merges.append((bytes([unicode_to_bytes[c] for c in c1]), bytes([unicode_to_bytes[c] for c in c2])))
        return cls(vocab, merges, special_tokens)

    def encode_chunk(self, chunk: str):
        # some chunk could be really long, so need to improve this logic
        THRESHOLD = 100
        orig_chunk = chunk
        chunk = list(chunk.encode("utf-8"))
        max_merge = len(chunk)
        if max_merge <= THRESHOLD:
            # TODO: Improve this logic, currently it's O(n^2) and it can be improved to O(nlogn) with priority queue
            while len(chunk) >= 2:
                pairs = [(chunk[i], chunk[i + 1]) for i in range(len(chunk) - 1)]
                lowest_pair = min(pairs, key=lambda pair: self.merges_dict.get(pair, float("inf")))
                if lowest_pair in self.merges_dict:
                    i = pairs.index(lowest_pair)
                    chunk = chunk[:i] + [self.merges_dict[lowest_pair]] + chunk[i + 2 :]
                else:
                    break
        else:
            # use priority queue and doubly linked list
            pairs = defaultdict(int)

            # construct doubly linked list and their index
            index = defaultdict(list)
            start_node = Node(chunk[0], 1)
            prev_node = start_node
            for i in range(1, len(chunk)):
                cur_node = Node(chunk[i], 1, prev=prev_node)
                prev_node.next = cur_node
                pair = (prev_node.value, cur_node.value)
                index[pair].append(prev_node)
                prev_node = cur_node
                pairs[pair] += 1

            # construct priority queue, low index has high priority
            pq = PriorityQueue()
            for pair in pairs:
                if pair in self.merges_dict:
                    pq.push(pair, -self.merges_dict[pair])

            # merge until no pair to merge
            while pq.size() > 0:
                pair, new_idx = pq.top()
                new_idx = -new_idx
                pq.pop()  # after merge, that pair will no longer exist

                if pairs[pair] == 0:
                    # pair does not exist, skip
                    continue

                for node in index[pair]:
                    node: Node
                    if node.value != pair[0] or node.next is None or node.next.value != pair[1]:
                        # invalid node, skip
                        continue

                    new_pairs = []
                    pairs[pair] -= 1
                    # remove pair
                    if node.prev:
                        pairs[(node.prev.value, pair[0])] -= 1
                        pairs[(node.prev.value, new_idx)] += 1
                        new_pairs.append((node.prev.value, new_idx))
                    if node.next.next:
                        pairs[(pair[1], node.next.next.value)] -= 1
                        pairs[(new_idx, node.next.next.value)] += 1
                        new_pairs.append((new_idx, node.next.next.value))
                    node.next.delete()
                    node.value = new_idx
                    # update index
                    if node.prev:
                        index[(node.prev.value, new_idx)].append(node.prev)
                    if node.next:
                        index[(new_idx, node.next.value)].append(node)
                    # update priority queue
                    for new_pair in new_pairs:
                        if new_pair in self.merges_dict and pq.index.get(new_pair, None) is None:
                            pq.push(new_pair, -self.merges_dict[new_pair])
                chunk = []
                node = start_node
                while node:
                    chunk.append(node.value)
                    node = node.next
        return chunk

    def encode(self, text: str, num_proc: int = 4) -> List[int]:
        if len(text) > 10 * 1024 * 1024:
            return list(self.encode_iterable([text], num_proc))
        if self.special_regex:
            special_chunks = re.split(self.special_regex, text)
        else:
            special_chunks = [text]

        tokens = []
        for text in special_chunks:
            if len(text) == 0:
                continue
            if text in self.special_tokens_map:
                tokens.extend([self.special_tokens_map[text]])
            else:
                chunks = re.finditer(self.compiled_pattern, text)
                for chunk in chunks:
                    tokens.extend(self.encode_chunk(chunk.group()))
        return tokens

    def encode_iterable(self, iterable: Iterable[str], num_proc: int = 4, progress: bool = False) -> Iterator[int]:
        # NOTE: Throughput is about 10MB/s with num_proc=16, need to improve
        CLEAR_CHUNK_NUM = num_proc * num_proc
        CHUNK_SIZE = 1024 * 1024
        ADJUST_SIZE = 1024
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
            futures = []
            buf = ""
            if progress:
                pbar = tqdm(unit=" characters", unit_scale=True, desc="Encoding") if progress else None
            for text in iterable:
                buf += text
                if progress:
                    pbar.update(len(text))
                while len(buf) >= CHUNK_SIZE:
                    pos = CHUNK_SIZE
                    found = False
                    while not found:
                        # adjust boundary
                        if self.special_regex:
                            m = self.special_regex.search(buf, pos)
                            if m:
                                pos = m.end()
                                found = True
                            else:
                                ws = re.search(r"\s", buf[pos:])
                                if ws:
                                    pos = pos + ws.start()
                                found = True
                        else:
                            ws = re.search(r"\s", buf[pos:])
                            if ws:
                                pos = pos + ws.start()
                                found = True
                        if not found:
                            pos -= ADJUST_SIZE
                        if pos <= CHUNK_SIZE - 50 * ADJUST_SIZE:
                            break
                    if not found:
                        break
                    chunk = buf[:pos]
                    buf = buf[pos:]
                    futures.append(executor.submit(self.encode, chunk))
                if len(futures) >= CLEAR_CHUNK_NUM:
                    tokens = []
                    for future in concurrent.futures.as_completed(futures):
                        tokens.extend(future.result())
                    yield from tokens
                    futures.clear()
                    gc.collect()
            if buf:
                futures.append(executor.submit(self.encode, buf))
            if progress:
                pbar.close()
        tokens = []
        for future in concurrent.futures.as_completed(futures):
            tokens.extend(future.result())
        yield from tokens
        gc.collect()

    def decode(self, tokens: List[int]) -> str:
        cat_bytes = b"".join(self.vocab[token] for token in tokens)
        return cat_bytes.decode("utf-8", errors="replace")

    @staticmethod
    def from_pretrained(name: str):
        PRETRAINED = {
            "open_webtext": {
                "vocab_filepath": str(Path(__file__).parent.parent / "tokenizers" / "open_webtext_vocab.json"),
                "merges_filepath": str(Path(__file__).parent.parent / "tokenizers" / "open_webtext_merges.txt"),
                "special_tokens": ["<|endoftext|>"],
            },
            "tiny_stories": {
                "vocab_filepath": str(Path(__file__).parent.parent / "tokenizers" / "tiny_stories_vocab.json"),
                "merges_filepath": str(Path(__file__).parent.parent / "tokenizers" / "tiny_stories_merges.txt"),
                "special_tokens": ["<|endoftext|>"],
            },
        }
        if name not in PRETRAINED:
            raise ValueError(f"Unknown pretrained tokenizer: {name}")
        params = PRETRAINED[name]
        return Tokenizer.from_files(params["vocab_filepath"], params["merges_filepath"], params["special_tokens"])

    @classmethod
    def train(
        cls, input_path: str, vocab_size: int, special_tokens: List[str], progress: bool = False, num_proc: int = 4
    ):
        # NOTE: UTF-8 decoding consumes huge memory, so process in streaming manner. Here, <|endoftext|> is specific delimiter for this case.
        READ_CHUNK_SIZE = 1024 * 1024
        DOC_PROCESS_SIZE = 1024 * 1024 * 10
        chunk_freq = dict()
        doc_delimiter = "<|endoftext|>"
        buf = ""
        file_size = os.path.getsize(input_path)
        with timer("Reading corpus and counting frequencies"):
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
                futures = []
                pbar = (
                    tqdm(total=file_size, unit="B", unit_scale=True, desc="Reading file", leave=False)
                    if progress
                    else None
                )
                with open(input_path) as f:
                    while True:
                        chunk = f.read(READ_CHUNK_SIZE)
                        if not chunk:
                            break
                        buf += chunk
                        if pbar:
                            pbar.update(len(chunk.encode("utf-8")))
                        if len(buf) >= DOC_PROCESS_SIZE:
                            rem_idx = buf.rfind(doc_delimiter) + len(doc_delimiter)
                            corpus = buf[:rem_idx].replace(doc_delimiter, "")
                            futures.append(executor.submit(cls.pre_tokenize, corpus))
                            buf = buf[rem_idx:]
                if buf:
                    futures.append(executor.submit(cls.pre_tokenize, buf.replace(doc_delimiter, "")))
                for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures), desc="Processing chunks", leave=False
                ):
                    result = future.result()
                    for token, freq in result.items():
                        chunk_freq[token] = chunk_freq.get(token, 0) + freq

        gc.collect()
        logger.info(f"Number of chunks: {len(chunk_freq)}")
        vocab = dict()
        for i in range(256):
            vocab[i] = bytes([i])
        merges = []
        num_merges = vocab_size - len(vocab) - len(special_tokens)
        with timer("Contructing Data Structures"):
            # construct initial index and occurences from chunks
            # index key is pair, value is first node of pair
            # NOTE: invalid node will be NOT deleted for simplicity, this logic will be updated if memory is a concern
            occurences = defaultdict(int)
            index = defaultdict(list)

            for chunk, freq in chunk_freq.items():
                prev_node = Node(chunk[0], freq)
                for value in chunk[1:]:
                    cur_node = Node(value, freq, prev=prev_node)
                    prev_node.next = cur_node
                    pair = (prev_node.value, cur_node.value)
                    occurences[pair] += freq
                    index[pair].append(prev_node)
                    prev_node = cur_node
            pq = PriorityQueue()
            for pair, freq in occurences.items():
                pq.push(pair, freq)

            del occurences
        gc.collect()

        with timer("Merging"):
            iterator = tqdm(range(num_merges), leave=False) if progress else range(num_merges)
            for _ in iterator:
                p1, p2 = pq.get_max(vocab)
                c1 = vocab[p1]
                c2 = vocab[p2]

                # add to vocab and merge
                new_idx = len(vocab)
                merges.append((c1, c2))
                vocab[new_idx] = c1 + c2

                # update index and occurences
                changes = defaultdict(int)
                for node in index[(p1, p2)]:
                    node: Node
                    if node.value != p1 or node.next is None or node.next.value != p2:
                        # invalid node, skip
                        continue

                    freq = node.freq
                    changes[(p1, p2)] -= freq
                    if node.prev:
                        changes[(node.prev.value, p1)] -= freq
                        changes[(node.prev.value, new_idx)] += freq

                    if node.next.next:
                        changes[(p2, node.next.next.value)] -= freq
                        changes[(new_idx, node.next.next.value)] += freq

                    node.next.delete()
                    node.value = new_idx

                    # update index
                    if node.prev:
                        index[(node.prev.value, new_idx)].append(node.prev)
                    if node.next:
                        index[(new_idx, node.next.value)].append(node)

                for pair, diff in changes.items():
                    if diff:
                        pq.update(pair, diff)

        # add special tokens
        for i, token in enumerate(special_tokens):
            new_idx = len(vocab)
            vocab[new_idx] = token.encode("utf-8")

        return vocab, merges

    @classmethod
    def pre_tokenize(cls, text: str) -> Dict[Tuple[int, ...], int]:
        counter = Counter(re.findall(cls.compiled_pattern, text))
        chunks = {}
        to_tuple = lambda seq: tuple(c for c in seq.encode("utf-8"))
        for token, freq in counter.items():
            chunks[to_tuple(token)] = freq
        return chunks

    @staticmethod
    def export_vocab(vocab):
        decoder = gpt2_bytes_to_unicode()
        unicode_vocab = dict()
        for i, token in vocab.items():
            unicode_token = "".join([decoder[c] for c in token])
            unicode_vocab[i] = unicode_token
        return unicode_vocab

    @staticmethod
    def export_merges(merges):
        decoder = gpt2_bytes_to_unicode()
        unicode_merges = []
        for merge in merges:
            unicode_merge = tuple("".join([decoder[c] for c in token]) for token in merge)
            unicode_merges.append(unicode_merge)
        return unicode_merges
