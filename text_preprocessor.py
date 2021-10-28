from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
from torch.utils.data import DataLoader
import torch


class TextPreprocessor():
    def __init__(self):
        self.dataset = datasets.WikiText2(root="data/", split="valid")
        self.dataloader = DataLoader(self.dataset)
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(self.yield_tokens(self.dataset), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))

    def to_tensor(self, MAX_SEQ_LEN=200):
        #TODO: I wasn't able to iterate over dataloader,
        #TODO: So I initiliazied them again and it worked!
        #TODO: Check this bug again
        self.dataset = datasets.WikiText2(root="data/", split="valid")
        self.dataloader = DataLoader(self.dataset)
        sentence_list = []

        for i, X in enumerate(self.dataloader):
            this_sentence = X[0]
            this_numbers = self.text_pipeline(this_sentence)

            if this_numbers:
                seq_length = len(this_numbers)
                if seq_length > MAX_SEQ_LEN:
                    this_numbers = this_numbers[:MAX_SEQ_LEN]
                elif seq_length < MAX_SEQ_LEN:
                    this_numbers = this_numbers + [0] * (MAX_SEQ_LEN - seq_length)
                sentence_list.append(this_numbers)

        return torch.tensor(sentence_list)

    def yield_tokens(self, data_iter):
        for text in data_iter:
            yield self.tokenizer(text)