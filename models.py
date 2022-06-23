import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var
from torch import optim
from utils import *
from data import *
from lf_evaluator import *
import numpy as np
from typing import List

def add_models_args(parser):
    """
    Command-line arguments to the system related to your model.  Feel free to extend here.  
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes

class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap with Jaccard similarity
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # Note that this is a list of a single Derivation
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs

class Seq2SeqSemanticParser(nn.Module):
    def __init__(self, input_indexer, output_indexer, emb_dim, hidden_size, model_input_emb, model_enc, model_output_emb, model_dec, embedding_dropout=0.2, decoder_len_limit = 65, bidirect=True):
        # We've include some args for setting up the input embedding and encoder
        # You'll need to add code for output embedding and decoder
        super(Seq2SeqSemanticParser, self).__init__()
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.decoder_len_limit = decoder_len_limit
        
        self.input_emb = EmbeddingLayer(emb_dim, len(input_indexer), embedding_dropout)
        self.encoder = RNNEncoder(emb_dim, hidden_size, bidirect)
        #raise Exception("implement me!")

        self.model_input_emb = model_input_emb
        self.model_enc = model_enc
        self.model_output_emb = model_output_emb
        self.model_dec = model_dec

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True,dropout=0.2)
        self.ff = nn.Linear(hidden_size, len(output_indexer))
        self.softmax = nn.LogSoftmax(dim = 1)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        nn.init.xavier_uniform_(self.ff.weight)

    def forward(self, ex, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor, teacher_forcing_ratio, model_enc, model_dec, model_input_emb, model_output_emb, enc_optimizer, dec_optimizer, input_emb_optimizer, output_emb_optimizer):
        """
        :param x_tensor/y_tensor: either a non-batched input/output [sent len] vector of indices or a batched input/output
        [batch size x sent len]. y_tensor contains the gold sequence(s) used for training
        :param inp_lens_tensor/out_lens_tensor: either a vector of input/output length [batch size] or a single integer.
        lengths aren't needed if you don't batchify the training.
        :return: loss of the batch
        """

        #raise Exception("implement me!")

        model_enc.train()
        model_dec.train()
        model_input_emb.train()
        model_output_emb.train()

        model_enc.zero_grad()
        model_dec.zero_grad()
        model_input_emb.zero_grad()
        model_output_emb.zero_grad()

        loss = 0
        SOS_token = 1
        EOS_token = 2
        criterion = torch.nn.NLLLoss()

        (enc_output, enc_context_mask, enc_hidden, enc_bi_hidden) = self.encode_input(x_tensor, inp_lens_tensor, model_input_emb, model_enc)
        dec_hidden  = enc_hidden
        input_output_map = create_input_output_map(ex.x_indexed, self.input_indexer, self.output_indexer)
        context_vec = enc_bi_hidden[0]

        dec_input = torch.as_tensor([[SOS_token]])
        y_temp = []

        teacher_forcing = True if random.random() <= teacher_forcing_ratio else False

        for y_ex in range(len(y_tensor)):
            dec_output, dec_input, dec_input_val, dec_hidden, context_vec, for_inference = decode_attn(dec_input, dec_hidden, model_output_emb, model_dec, context_vec, enc_output, 1, input_output_map)
            loss += criterion(dec_output, y_tensor[y_ex].unsqueeze(0))
            y_temp.append(dec_input.item())
            if dec_input == EOS_token:
                break
            if teacher_forcing:
                dec_input = y_tensor[y_ex].unsqueeze(0).unsqueeze(0)
        
        loss.backward()
        input_emb_optimizer.step()
        output_emb_optimizer.step()
        enc_optimizer.step()
        dec_optimizer.step()

        return loss

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        #raise Exception("implement me!")
        self.model_input_emb.eval()
        self.model_enc.eval()
        self.model_output_emb.eval()
        self.model_dec.eval()

        self.model_input_emb.zero_grad()
        self.model_output_emb.zero_grad()
        self.model_enc.zero_grad()
        self.model_dec.zero_grad()

        SOS_token = 1
        EOS_token = self.output_indexer.index_of('<EOS>')
        SOS_label = self.output_indexer.get_object(SOS_token)
        beam_length = 1
        derivations = []
        print("EOS_token: ", EOS_token)

        for ex in test_data:
            count = 0
            y_toks =[]
            self.model_input_emb.zero_grad()
            self.model_output_emb.zero_grad()
            self.model_enc.zero_grad()
            self.model_dec.zero_grad()

            x_tensor = torch.as_tensor([ex.x_indexed])
            y_tensor = torch.as_tensor(ex.y_indexed)
            inp_lens_tensor = torch.as_tensor([len(ex.x_indexed)])

            enc_output, enc_context_mask, enc_hidden, enc_bi_hidden = self.encode_input(x_tensor, inp_lens_tensor, self.model_input_emb, self.model_enc)
            dec_hidden = enc_hidden
            context_vec = enc_bi_hidden[0]
            dec_input = torch.as_tensor([[SOS_token]])
            input_output_map = create_input_output_map(ex.x_indexed, self.input_indexer, self.output_indexer)

            while (dec_input.item() != EOS_token) and count <= self.decoder_len_limit:
                dec_output, dec_input, dec_input_val, dec_hidden, context_vec, for_inference = decode_attn(dec_input, dec_hidden, self.model_output_emb, self.model_dec, context_vec, enc_output, beam_length, input_output_map)
                top_val, top_ind = for_inference.topk(1)
                top_ind = top_ind.detach()
                if top_ind <len(self.output_indexer):
                    y_label = self.output_indexer.get_object(dec_input.item())
                else:
                    source_word_idx = top_ind.item() - len(self.output_indexer)
                    copy_word_idx = ex.x_indexed[source_word_idx]
                    y_label = self.input_indexer.get_object(copy_word_idx)
                if dec_input.item() != EOS_token:
                    y_toks.append(y_label)
                count = count + 1
            derivations.append([Derivation(ex, 1.0 , y_toks)])
        return derivations

    def encode_input(self, x_tensor, inp_lens_tensor, model_input_emb, model_enc):
        """
        Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
        inp_lens_tensor lengths.
        YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
        as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
        :param x_tensor: [batch size, sent len] tensor of input token indices
        :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
        :param model_input_emb: EmbeddingLayer
        :param model_enc: RNNEncoder
        :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
        are real and which ones are pad tokens), and the encoder final states (h and c tuple). ONLY THE ENCODER FINAL
        STATES are needed for the basic seq2seq model. enc_output_each_word is needed for attention, and
        enc_context_mask is needed to batch attention.

        E.g., calling this with x_tensor (0 is pad token):
        [[12, 25, 0],
        [1, 2, 3],
        [2, 0, 0]]
        inp_lens = [2, 3, 1]
        will return outputs with the following shape:
        enc_output_each_word = 3 x 3 x dim, enc_context_mask = [[1, 1, 0], [1, 1, 1], [1, 0, 0]],
        enc_final_states = 3 x dim
        """
        input_emb = model_input_emb.forward(x_tensor)
        (enc_output_each_word, enc_context_mask, enc_final_states, enc_bi_hidden) = model_enc.forward(input_emb, inp_lens_tensor)
        enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
        return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped, enc_bi_hidden)

class EmbeddingLayer(nn.Module):
    """
    Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
    Works for both non-batched and batched inputs
    """
    def __init__(self, input_dim: int, full_dict_size: int, embedding_dropout_rate: float):
        """
        :param input_dim: dimensionality of the word vectors
        :param full_dict_size: number of words in the vocabulary
        :param embedding_dropout_rate: dropout rate to apply
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    def forward(self, input):
        """
        :param input: either a non-batched input [sent len x voc size] or a batched input
        [batch size x sent len x voc size]
        :return: embedded form of the input words (last coordinate replaced by input_dim)
        """
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings

class RNNEncoder(nn.Module):
    """
    One-layer RNN encoder for batched inputs -- handles multiple sentences at once. To use in non-batched mode, call it
    with a leading dimension of 1 (i.e., use batch size 1)
    """
    def __init__(self, input_emb_dim: int, hidden_size: int, bidirect: bool):
        """
        :param input_emb_dim: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.hidden_size = hidden_size
        self.input_emb_dim = input_emb_dim
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_emb_dim, hidden_size, num_layers=1, batch_first=True,
                               dropout=0., bidirectional=self.bidirect)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def forward(self, embedded_words, input_lens):
        """
        Runs the forward pass of the LSTM
        :param embedded_words: [batch size x sent len x input dim] tensor
        :param input_lens: [batch size]-length vector containing the length of each input sentence
        :return: output (each word's representation), context_mask (a mask of 0s and 1s
        reflecting where the model's output should be considered), and h_t, a *tuple* containing
        the final states h and c from the encoder for each sentence.
        Note that output is only needed for attention, and context_mask is only used for batched attention.
        """
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True, enforce_sorted=False)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = max(input_lens.data).item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            bi_ht = (h_, c_)
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (output, context_mask, h_t, bi_ht)

class AttnRNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size_enc, hidden_size_dec, out):
        super(AttnRNNDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec
        self.out = out
        self.rnn = nn.LSTM(input_size, hidden_size_dec, num_layers=1, batch_first=True,dropout=0)
        self.attn = nn.Linear(hidden_size_enc, hidden_size_dec)
        self.attn_hid = nn.Linear(hidden_size_dec + hidden_size_enc, hidden_size_dec)
        self.ff = nn.Linear(hidden_size_dec, out)
        self.softmax = nn.Softmax(dim = 1)
        self.Wh = nn.Linear(hidden_size_enc, 1)
        self.Ws = nn.Linear(hidden_size_dec, 1)
        self.Wx = nn.Linear(input_size,1)
        self.bptr = nn.Linear(1,1)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.xavier_uniform_(self.attn_hid.weight)
        nn.init.xavier_uniform_(self.ff.weight)
        nn.init.xavier_uniform_(self.Wh.weight)
        nn.init.xavier_uniform_(self.Ws.weight)
        nn.init.xavier_uniform_(self.Wx.weight)
        nn.init.xavier_uniform_(self.bptr.weight)

    def forward(self, input, hidden, encoder_outputs, input_output_map):
        output, (h,c) = self.rnn(input, hidden)
        h_bar = h[0]
        encoder_outputs = encoder_outputs.squeeze()
        attn_weight = self.attn(encoder_outputs).squeeze()
        attn_weight = torch.transpose(attn_weight, 0, 1)
        attn_energy = torch.matmul(h_bar, attn_weight)
        attn_score = F.softmax(attn_energy, dim = 1)
        context = torch.matmul(attn_score, encoder_outputs)
        attn_hid_combined = torch.cat((context, h_bar), 1)
        attn_hid_transformed = self.attn_hid(attn_hid_combined)
        out = self.ff(attn_hid_transformed)
        p_vocab = self.softmax(out)
        p_gen = self.sigmoid(self.Wh(context) + self.Ws(h_bar))
        p_gen = p_gen.squeeze(0)
        ai = torch.matmul(attn_score, input_output_map)
        Pw = p_gen * p_vocab + (1-p_gen) * ai
        Pw = torch.log(Pw)
        for_inference = torch.cat((p_gen*p_vocab, (1-p_gen)*attn_score), 1)
        return Pw, (h,c), context, for_inference

def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])

def decode_attn(y_index, hidden, model_output_emb, model_dec, context_vec, encoder_output, beam_length, input_output_map):
    output_emb = model_output_emb.forward(y_index)
    output_emb= torch.cat((output_emb.squeeze(1), context_vec), dim = 1)
    output_emb = output_emb.unsqueeze(0)
    output, hidden, context_vec, for_inference = model_dec.forward(output_emb, hidden, encoder_output, input_output_map)
    top_val, top_ind = output.topk(beam_length)
    dec_input = top_ind.detach()
    dec_input_prob = top_val.detach()
    return output, dec_input, dec_input_prob, hidden, context_vec, for_inference

def create_input_output_map(input_seq, input_indexer, output_indexer):
    map = torch.zeros(len(input_seq), len(output_indexer))
    unk_index = output_indexer.index_of(UNK_SYMBOL)
    for i, idx in enumerate(input_seq):
        word = input_indexer.get_object(idx)
        out_index = output_indexer.index_of(word)
        if out_index == -1:
            map[i][unk_index] =  1.0
        else:
            map[i][out_index] =  1.0
    return map

def train_model_encdec(train_data: List[Example], dev_data: List[Example], input_indexer, output_indexer, args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param dev_data: Development set in case you wish to evaluate during training
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, reverse_input=False)
    all_test_input_data = make_padded_input_tensor(dev_data, input_indexer, input_max_len, reverse_input=False)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(dev_data, output_indexer, output_max_len)

    if args.print_dataset:
        print("Train length: %i" % input_max_len)
        print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
        print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # First create a model. Then loop over epochs, loop over examples, and given some indexed words
    # call your seq-to-seq model, accumulate losses, update parameters

    # raise Exception("Implement the rest of me to train your encoder-decoder model")

    epochs = 20
    teacher_forcing_ratio = 1.0
    emb_dim = 300
    hidden_size = 200

    model_input_emb = EmbeddingLayer(emb_dim, len(input_indexer), 0.2)
    model_output_emb = EmbeddingLayer(emb_dim, len(output_indexer), 0.2)

    model_enc = RNNEncoder(emb_dim, hidden_size, True)
    model_dec = AttnRNNDecoder(emb_dim + hidden_size*2, hidden_size * 2, hidden_size, len(output_indexer))

    decoder = Seq2SeqSemanticParser(input_indexer, output_indexer, emb_dim, hidden_size, model_input_emb, model_enc, model_output_emb, model_dec)

    enc_optimizer = optim.Adam(model_enc.parameters(), lr=.001)
    dec_optimizer = optim.Adam(model_dec.parameters(), lr=.001)

    input_emb_optimizer = optim.Adam(model_input_emb.parameters(), lr=args.lr)
    output_emb_optimizer = optim.Adam(model_output_emb.parameters(), lr=args.lr)

    for i in range(epochs):
        count = 0
        teacher_forcing_ratio = teacher_forcing_ratio**i        
        for ex in train_data:
            x_tensor = torch.as_tensor([ex.x_indexed])
            y_tensor = torch.as_tensor(ex.y_indexed)
            inp_lens_tensor = torch.as_tensor([len(ex.x_indexed)])
            out_lens_tensor = torch.as_tensor([len(ex.y_indexed)])
            
            loss = decoder(ex, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor, teacher_forcing_ratio, model_enc, model_dec, model_input_emb, model_output_emb, enc_optimizer, dec_optimizer, input_emb_optimizer, output_emb_optimizer)

            if count %25 == 0:
                print ("loss: ", loss)
            count += 1
        
        print("epoch: ", i)

    return decoder