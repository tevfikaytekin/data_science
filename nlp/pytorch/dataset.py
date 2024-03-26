import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import random

class Word2VecSentenceDataset(Dataset):
    def __init__(self, sentences, window_size=2, negative_samples=5, total_negative_samples=10000):
        self.tokens = [word for sentence in sentences for word in sentence]
        self.vocab = {word: i for i, word in enumerate(set(self.tokens))}
        self.index_to_word = {i: word for word, i in self.vocab.items()}
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.word_frequencies = np.array([freq for word, freq in Counter(self.tokens).items()])**0.75
        self.word_frequencies /= self.word_frequencies.sum()
        self.negative_sample_pool = np.random.choice(len(self.word_frequencies), size=total_negative_samples, replace=True, p=self.word_frequencies).tolist()

        self.data = self.generate_training_data(sentences)

    def generate_training_data(self, sentences):
        training_data = []
        for sentence in sentences:
            for i, target_word in enumerate(sentence):
                target_index = self.vocab[target_word]
                context_indices = range(max(0, i - self.window_size), min(len(sentence), i + self.window_size + 1))
                for j in context_indices:
                    if i != j:  # Exclude the target word itself
                        context_word = sentence[j]
                        context_index = self.vocab[context_word]

                        negative_samples = random.sample(self.negative_sample_pool, k=self.negative_samples)
                        #negative_samples = self.get_negative_samples(target_index, self.negative_samples)
                        training_data.append((target_index, context_index, negative_samples))
   
        return training_data

    def get_negative_samples(self, target, num_samples):
        negatives = []
        while len(negatives) < num_samples:
            neg_sample = np.random.choice(len(self.vocab), p=self.word_frequencies)
            if neg_sample != target:
                negatives.append(neg_sample)
        return negatives

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target, context, negatives = self.data[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long), torch.tensor(negatives, dtype=torch.long)
