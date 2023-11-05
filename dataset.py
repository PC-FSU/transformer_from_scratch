import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):

        """
        Create a BilingualDataset for training a neural machine translation model.

        Args:
            ds: The dataset containing translation pairs.
            tokenizer_src: Tokenizer for the source language.
            tokenizer_tgt: Tokenizer for the target language.
            src_lang: Source language identifier.
            tgt_lang: Target language identifier.
            seq_len: Maximum sequence length for training examples.
        """

        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        """
        Get the total number of training examples in the dataset.

        Returns:
            int: The number of training examples.
        """
        return len(self.ds)

    def __getitem__(self, index):
        """
        Get a training example from the dataset at the specified index.

        Args:
            index (int): The index of the training example to retrieve.

        Returns:
            dict: A dictionary containing encoder and decoder inputs, masks, labels, and source/target text.
        """
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens. Tokenizer split the sentence by whitespace, and then return unique id's based on vocab created.
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add sos token on decoder. That's why minus 1 instead of 2. In label we add eos token, therfore there too, we will see minus 1.
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1


        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")


        # Add sos and eos token. The encoder reads the input sequence to construct an embedding representation of the sequence. 
        # Terminating the input in an end-of-sequence (EOS) token signals to the encoder that when it receives that input, 
        # the output needs to be the finalized embedding. We (normally) don't care about intermediate states of the embedding,
        #  and we don't want the encoder to have to guess as to whether or not the input sentence is complete or not.
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only sos token. Decoders next-word predictors are trained by shifting the output by one token.
        #  So you need to start with a 1-token context, which should be SOS.
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only eos token. The EOS token is important for the decoder as well: the explicit "end" token allows the decoder to emit
        #  arbitrary-length sequences. The decoder will tell us when it's done emitting tokens: without an "end" token, 
        # we would have no idea when the decoder is done talking to us and continuing to emit tokens will produce gibberish
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len), each word can look at all the non-padding word
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len), Each word can look at non-padding word((1, seq_len)) & each word can look at in past only((1, seq_len, seq_len)), not in future. 
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    """
    Generate a causal mask for use in self-attention mechanisms.

    Args:
        size (int): The size of the mask.

    Returns:
        torch.Tensor: A 3D binary tensor with a triangular shape to restrict attention to the past in self-attention mechanisms.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


