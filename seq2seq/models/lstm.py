import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils
from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
from seq2seq.models import register_model, register_model_architecture


@register_model('lstm')
class LSTMModel(Seq2SeqModel):
    """ Defines the sequence-to-sequence model class. """
    
    def __init__(self,
                 encoder,
                 decoder):
        
        super().__init__(encoder, decoder)
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-embed-dim', type = int,
                            help = 'encoder embedding dimension')
        parser.add_argument('--encoder-embed-path',
                            help = 'path to pre-trained encoder embedding')
        parser.add_argument('--encoder-hidden-size',
                            type = int, help = 'encoder hidden size')
        parser.add_argument('--encoder-num-layers', type = int,
                            help = 'number of encoder layers')
        parser.add_argument('--encoder-bidirectional',
                            help = 'bidirectional encoder')  # oda: diff, default set to True in base_architecture
        parser.add_argument('--encoder-dropout-in',
                            help = 'dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out',
                            help = 'dropout probability for encoder output')
        
        parser.add_argument('--decoder-embed-dim', type = int,
                            help = 'decoder embedding dimension')
        parser.add_argument('--decoder-embed-path',
                            help = 'path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size',
                            type = int, help = 'decoder hidden size')
        parser.add_argument('--decoder-num-layers', type = int,
                            help = 'number of decoder layers')
        parser.add_argument('--decoder-dropout-in', type = float,
                            help = 'dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type = float,
                            help = 'dropout probability for decoder output')
        parser.add_argument('--decoder-use-attention',
                            help = 'decoder attention')  # diff, default true
        parser.add_argument('--decoder-use-lexical-model',
                            help = 'toggle for the lexical model')  # diff
    
    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """ Constructs the model. """
        base_architecture(args)
        encoder_pretrained_embedding = None
        decoder_pretrained_embedding = None
        
        # Load pre-trained embeddings, if desired
        if args.encoder_embed_path:
            encoder_pretrained_embedding = utils.load_embedding(
                args.encoder_embed_path, src_dict)
        if args.decoder_embed_path:
            decoder_pretrained_embedding = utils.load_embedding(
                args.decoder_embed_path, tgt_dict)
        
        # Construct the encoder
        encoder = LSTMEncoder(dictionary = src_dict,
                              embed_dim = args.encoder_embed_dim,
                              hidden_size = args.encoder_hidden_size,
                              num_layers = args.encoder_num_layers,
                              bidirectional = bool(
                                  eval(args.encoder_bidirectional)),
                              dropout_in = args.encoder_dropout_in,
                              dropout_out = args.encoder_dropout_out,
                              pretrained_embedding = encoder_pretrained_embedding)
        
        # Construct the decoder
        decoder = LSTMDecoder(dictionary = tgt_dict,
                              embed_dim = args.decoder_embed_dim,
                              hidden_size = args.decoder_hidden_size,
                              num_layers = args.decoder_num_layers,
                              dropout_in = args.decoder_dropout_in,
                              dropout_out = args.decoder_dropout_out,
                              pretrained_embedding = decoder_pretrained_embedding,
                              use_attention = bool(
                                  eval(args.decoder_use_attention)),
                              use_lexical_model = bool(eval(args.decoder_use_lexical_model)))
        return cls(encoder, decoder)


class LSTMEncoder(Seq2SeqEncoder):
    """ Defines the encoder class. """
    
    def __init__(self,
                 dictionary,
                 embed_dim = 64,
                 hidden_size = 64,
                 num_layers = 1,
                 bidirectional = True,
                 dropout_in = 0.25,
                 dropout_out = 0.25,
                 pretrained_embedding = None):
        
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.output_dim = 2 * hidden_size if bidirectional else hidden_size
        
        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(
                len(dictionary), embed_dim, dictionary.pad_idx)
        
        dropout_lstm = dropout_out if num_layers > 1 else 0.
        self.lstm = nn.LSTM(input_size = embed_dim,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout = dropout_lstm,
                            bidirectional = bool(bidirectional))
    
    def forward(self, src_tokens, src_lengths):
        """ Performs a single forward pass through the instantiated encoder sub-network. """
        # Embed tokens and apply dropout
        batch_size, src_time_steps = src_tokens.size()
        src_embeddings = self.embedding(src_tokens)  # bs x len_sentence x embed size
        _src_embeddings = F.dropout(
            src_embeddings, p = self.dropout_in, training = self.training)
        
        # Transpose batch: [batch_size, src_time_steps, num_features] -> [src_time_steps, batch_size, num_features]
        src_embeddings = _src_embeddings.transpose(0, 1)
        
        # Pack embedded tokens into a PackedSequence.
        # Oda: Originally it is a len_sent x bs x embed_size tensor.
        #   The problem is that not all sentences have the same length.
        #   So here the shorted sents are  converted into appropriate length.
        packed_source_embeddings = nn.utils.rnn.pack_padded_sequence(
            src_embeddings, src_lengths.data.tolist())
        
        # Pass source input through the recurrent layer(s)
        if self.bidirectional:
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
        else:
            state_size = self.num_layers, batch_size, self.hidden_size
        
        hidden_initial = src_embeddings.new_zeros(*state_size)
        context_initial = src_embeddings.new_zeros(*state_size)
        
        # Oda: packed_outputs: num_tokens x decoder hidden size
        packed_outputs, (final_hidden_states, final_cell_states) = self.lstm(packed_source_embeddings,
                                                                             (hidden_initial, context_initial))
        
        # Unpack LSTM outputs and optionally apply dropout (dropout currently disabled)
        # Oda: Return to len_sent x bs x es. 这也是为什么叫做pad，就是将短的后面pad上0
        #  lstm_output: lensent x bs x decode_hidden_size.
        #  The output of encoder on each time step. Used only in attetion.
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value = 0.)
        lstm_output = F.dropout(
            lstm_output, p = self.dropout_out, training = self.training)
        assert list(lstm_output.size()) == [
            src_time_steps, batch_size, self.output_dim]  # sanity check
        
        # ___QUESTION-1-DESCRIBE-A-START___
        '''
        ___QUESTION-1-DESCRIBE-A-START___
        Describe what happens when self.bidirectional is set to True. 
        What is the difference between final_hidden_states and final_cell_states?
        '''
        '''
                Answer:
                h_n of shape (num_layers * num_directions, batch, hidden_size)
                c_n of shape (num_layers * num_directions, batch, hidden_size)

                to separate two directions state ==>

                h_n of shape (num_layers, batch, hidden_size * num_directions)
                c_n of shape (num_layers, batch, hidden_size * num_directions)

                cell_states represents state of storage cell
                hidden_states represents state of hidden layer units
        '''
        
        if self.bidirectional:
            def combine_directions(outs):
                return torch.cat([outs[0: outs.size(0): 2], outs[1: outs.size(0): 2]], dim = 2)
            
            final_hidden_states = combine_directions(final_hidden_states)
            final_cell_states = combine_directions(final_cell_states)
        '''___QUESTION-1-DESCRIBE-A-END___'''
        
        # Generate mask zeroing-out padded positions in encoder inputs
        src_mask = src_tokens.eq(self.dictionary.pad_idx)
        
        return {'src_embeddings': _src_embeddings.transpose(0, 1),
                'src_out': (lstm_output, final_hidden_states, final_cell_states),
                'src_mask': src_mask if src_mask.any() else None}


class AttentionLayer(nn.Module):
    """ Defines the attention layer class. Uses Luong's global attention with the general scoring function. """
    
    def __init__(self, input_dims, output_dims):
        super().__init__()
        # Scoring method is 'general'
        self.src_projection = nn.Linear(input_dims, output_dims, bias = False)
        self.context_plus_hidden_projection = nn.Linear(
            input_dims + output_dims, output_dims, bias = False)
    
    def forward(self, tgt_input, encoder_out, src_mask):
        # tgt_input has shape = [batch_size, input_dims]
        # encoder_out has shape = [src_time_steps, batch_size, output_dims]
        # src_mask has shape = [src_time_steps, batch_size]
        
        # Get attention scores
        # [batch_size, src_time_steps, output_dims]
        encoder_out = encoder_out.transpose(1, 0)
        
        # [batch_size, 1, src_time_steps]
        attn_scores = self.score(tgt_input, encoder_out)
        
        # ___QUESTION-1-DESCRIBE-B-START___
        '''
        ___QUESTION-1-DESCRIBE-B-START___
        Describe how the attention context vector is calculated. Why do we need to apply a mask to the attention scores?
        
        Answer:
        unsqueeze: src_mask shape: [src_time_steps, batch_size] --> [src_time_steps,1 , batch_size]
        mask as '-inf', softmax make it nealy 0
        avoid model simply copy the next word when predict.
        
        '''
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(dim = 1)
            attn_scores.masked_fill_(src_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim = -1)
        attn_context = torch.bmm(attn_weights, encoder_out).squeeze(dim = 1)
        # Oda: tgt_input is actually tgt_hidden_states,
        #  so here concat hidden and context as stated in textbook
        context_plus_hidden = torch.cat([tgt_input, attn_context], dim = 1)
        attn_out = torch.tanh(
            self.context_plus_hidden_projection(context_plus_hidden))
        '''___QUESTION-1-DESCRIBE-B-END___'''
        
        return attn_out, attn_weights.squeeze(dim = 1)
    
    def score(self, tgt_input, encoder_out):
        """ Computes attention scores. """
        
        # ___QUESTION-1-DESCRIBE-C-START___
        '''
        ___QUESTION-1-DESCRIBE-C-START___
        How are attention scores calculated? What role does matrix multiplication (i.e. torch.bmm()) play 
        in aligning encoder and decoder representations?
        
        Answer:
        scr_projection: Linear layer
        
        transpose: shape of encoder_out 
        [batch_size, src_time_steps, output_dims] --> [batch_size, output_dims, src_time_steps]
        
        bmm:
        for each item in batch: do [1, input_dims]*[output_dims, src_time_steps]
        
        '''
        projected_encoder_out = self.src_projection(encoder_out).transpose(2, 1)
        attn_scores = torch.bmm(tgt_input.unsqueeze(dim = 1), projected_encoder_out)
        '''___QUESTION-1-DESCRIBE-C-END___'''
        
        return attn_scores


class LSTMDecoder(Seq2SeqDecoder):
    """ Defines the decoder class. """
    
    def __init__(self,
                 dictionary,
                 embed_dim = 64,
                 hidden_size = 128,
                 num_layers = 1,
                 dropout_in = 0.25,
                 dropout_out = 0.25,
                 pretrained_embedding = None,
                 use_attention = True,
                 use_lexical_model = False):
        
        super().__init__(dictionary)
        
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        
        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(
                len(dictionary), embed_dim, dictionary.pad_idx)
        
        # Define decoder layers and modules
        self.attention = AttentionLayer(
            hidden_size, hidden_size) if use_attention else None
        
        self.layers = nn.ModuleList([nn.LSTMCell(
            input_size = hidden_size + embed_dim if layer == 0 else hidden_size,
            hidden_size = hidden_size)
            for layer in range(num_layers)])
        
        self.final_projection = nn.Linear(hidden_size, len(dictionary))
        
        self.use_lexical_model = use_lexical_model
        if self.use_lexical_model:
            # __QUESTION-5: Add parts of decoder architecture corresponding to the LEXICAL MODEL here
            # TODO: --------------------------------------------------------------------- CUT
            self.Lexical_FFNN = nn.Linear(embed_dim, embed_dim, bias = False)
            self.Lexical_add = nn.Linear(embed_dim, len(dictionary))
            # TODO: --------------------------------------------------------------------- /CUT
    
    def forward(self, tgt_inputs, encoder_out, incremental_state = None):
        """ Performs the forward pass through the instantiated model. """
        # Optionally, feed decoder input token-by-token
        if incremental_state is not None:
            tgt_inputs = tgt_inputs[:, -1:]
        
        # __QUESTION-5 : Following code is to assist with the LEXICAL MODEL implementation
        # Recover encoder input
        src_embeddings = encoder_out['src_embeddings']
        
        src_out, src_hidden_states, src_cell_states = encoder_out['src_out']
        src_mask = encoder_out['src_mask']
        src_time_steps = src_out.size(0)
        
        # Embed target tokens and apply dropout
        batch_size, tgt_time_steps = tgt_inputs.size()
        tgt_embeddings = self.embedding(tgt_inputs)
        tgt_embeddings = F.dropout(
            tgt_embeddings, p = self.dropout_in, training = self.training)
        
        # Transpose batch: [batch_size, tgt_time_steps, num_features] -> [tgt_time_steps, batch_size, num_features]
        tgt_embeddings = tgt_embeddings.transpose(0, 1)
        
        # Initialize previous states (or retrieve from cache during incremental generation)
        
        # ___QUESTION-1-DESCRIBE-D-START___
        '''
        ___QUESTION-1-DESCRIBE-D-START___
        Describe how the decoder state is initialized. When is cached_state == None? What role does input_feed play?
        
        Ansewr:
        When cached_state == None, all hidden_units and cell_units will set to zero.
        input_feed the previous prediction, so model can depend on all input and previous predictions to predict.
        '''
        # Oda: incremental_state will always be None, thus cached_state will be None in get_incremental_state
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            tgt_hidden_states, tgt_cell_states, input_feed = cached_state
        else:
            # Oda: Save all hidden and memory output for each RNN block. Only one for default
            #  This is not time step, it is stacked RNN.
            tgt_hidden_states = [torch.zeros(
                tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            tgt_cell_states = [torch.zeros(
                tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            input_feed = tgt_embeddings.data.new(
                batch_size, self.hidden_size).zero_()
        '''___QUESTION-1-DESCRIBE-D-END___'''
        
        if torch.cuda.is_available():
            tgt_hidden_states = [s.cuda() for s in tgt_hidden_states]
            tgt_cell_states = [s.cuda() for s in tgt_cell_states]
        
        # Initialize attention output node
        attn_weights = tgt_embeddings.data.new(
            batch_size, tgt_time_steps, src_time_steps).zero_()
        rnn_outputs = []
        
        # __QUESTION-5 : Following code is to assist with the LEXICAL MODEL implementation
        # Cache lexical context vectors per translation time-step
        lexical_contexts = []
        
        for j in range(tgt_time_steps):  # oda: j = time step
            # Concatenate the current token embedding with output from previous time step (i.e. 'input feeding')
            lstm_input = torch.cat([tgt_embeddings[j, :, :], input_feed], dim = 1)
            
            for layer_id, rnn_layer in enumerate(self.layers):  # Oda: go through stacked RNN
                # Pass target input through the recurrent layer(s)
                tgt_hidden_states[layer_id], tgt_cell_states[layer_id] = \
                    rnn_layer(
                        lstm_input, (tgt_hidden_states[layer_id], tgt_cell_states[layer_id]))
                
                # Current hidden state becomes input to the subsequent layer; apply dropout
                lstm_input = F.dropout(
                    tgt_hidden_states[layer_id], p = self.dropout_out, training = self.training)
            
            # ___QUESTION-1-DESCRIBE-E-START___
            '''
            ___QUESTION-1-DESCRIBE-E-START___
            How is attention integrated into the decoder? Why is the attention function given the previous 
            target state as one of its inputs? What is the purpose of the dropout layer?
            
            Answer:
            Attention of decoder gives a weight to all input_hidden_states and previous target state.
            
            Because decoder should condition on previous prediction to predict.
            
            Avoid over-fitting.
            '''
            if self.attention is None:
                input_feed = tgt_hidden_states[-1]
            else:
                # Oda: tgt_hidden_states is hidden for one time step
                #  step_attn_weights: bs x len_input
                #  attn_weights: bs x len_out x len_in
                input_feed, step_attn_weights = self.attention(tgt_hidden_states[-1], src_out, src_mask)
                attn_weights[:, j, :] = step_attn_weights
                
                if self.use_lexical_model:
                    # __QUESTION-5: Compute and collect LEXICAL MODEL context vectors here
                    # TODO: --------------------------------------------------------------------- CUT
                    # weight_sum = torch.zeros(
                    #     (src_embeddings.shape[1], src_embeddings.shape[2]))
                    # if torch.cuda.is_available():
                    #     weight_sum = weight_sum.cuda()
                    # for i in range(src_embeddings.shape[0]):
                    #     # print(step_attn_weights[:,i].shape)
                    #     # print(src_embeddings[i].shape)
                    #
                    #     weight_sum += step_attn_weights[:, i].repeat(
                    #         src_embeddings.shape[2], 1).transpose(1, 0) * src_embeddings[i]
                    # weight_sum = torch.bmm(src_embeddings.transpose(0, 1).transpose(1, 2),
                    #                        step_attn_weights.unsqueeze(2)).squeeze() # matrix impl
                    # # ws.append(weight_sum) # debug
                    # flt = torch.tanh(weight_sum)
                    #
                    # htl = torch.tanh(self.Lexical_FFNN(flt)) + flt
                    # lexical_contexts.append(htl)
                    # TODO: --------------------------------------------------------------------- /CUT
                    pass
            
            input_feed = F.dropout(
                input_feed, p = self.dropout_out, training = self.training)
            rnn_outputs.append(input_feed)
            '''___QUESTION-1-DESCRIBE-E-END___'''
        
        # Cache previous states (only used during incremental, auto-regressive generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (tgt_hidden_states, tgt_cell_states, input_feed))
        
        # Collect outputs across time steps
        decoder_output = torch.cat(rnn_outputs, dim = 0).view(tgt_time_steps, batch_size, self.hidden_size)
        
        # Transpose batch back: [tgt_time_steps, batch_size, num_features] -> [batch_size, tgt_time_steps, num_features]
        decoder_output = decoder_output.transpose(0, 1)
        
        # Final projection
        # Oda: From RNN output (feature space) to vocab. Softmax is applied later.
        #  one reason is that torch's CrossEntropyLoss accepts raw data, but not prob after softmax
        decoder_output = self.final_projection(decoder_output)
        
        if self.use_lexical_model:
            # __QUESTION-5: Incorporate the LEXICAL MODEL into the prediction of target tokens here
            # TODO: --------------------------------------------------------------------- CUT
            weighted_sum = torch.bmm(attn_weights, src_embeddings.transpose(0, 1))
            bs, out, es = weighted_sum.shape
            weighted_sum = weighted_sum.view(-1, es)
            flt = torch.tanh(weighted_sum)
            htl = torch.tanh(self.Lexical_FFNN(flt)) + flt
            lexical_contexts = htl.view(bs, out, es)
            
            # lexical_contexts = torch.stack(lexical_contexts).transpose(1, 0)
            # print(decoder_output.shape)
            # print(lexical_contexts.shape)
            decoder_output += self.Lexical_add(lexical_contexts)
            # TODO: --------------------------------------------------------------------- /CUT
        return decoder_output, attn_weights


@register_model_architecture('lstm', 'lstm')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 64)
    args.encoder_num_layers = getattr(args, 'encoder_num_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', 'True')
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.25)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.25)
    
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 128)
    args.decoder_num_layers = getattr(args, 'decoder_num_layers', 1)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.25)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.25)
    args.decoder_use_attention = getattr(args, 'decoder_use_attention', 'True')
    args.decoder_use_lexical_model = getattr(
        args, 'decoder_use_lexical_model', 'False')
