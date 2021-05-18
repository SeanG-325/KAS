# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS
a = "/gpu:0"
a2 = "/gpu:1"
a3 = "/gpu:1"
class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab
    self.LSTM_num = hps.LSTM_num.value
    self.hiddenSize = hps.hidden_dim.value

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    
    #A
    self._enc_batch_a = tf.placeholder(tf.int32, [hps.batch_size.value, None], name='enc_batch_a')
    self._enc_lens_a = tf.placeholder(tf.int32, [hps.batch_size.value], name='enc_lens_a')
    self._enc_padding_mask_a = tf.placeholder(tf.float32, [hps.batch_size.value, None], name='enc_padding_mask_aa')
    self._enc_batch_aa = tf.placeholder(tf.int32, [hps.batch_size.value, None], name='enc_batch_aa')
    self._enc_lens_aa = tf.placeholder(tf.int32, [hps.batch_size.value], name='enc_lens_aa')
    self._enc_padding_mask_aa = tf.placeholder(tf.float32, [hps.batch_size.value, None], name='enc_padding_mask_aa')
    #\A
    self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size.value, None], name='enc_batch')
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size.value], name='enc_lens')
    self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size.value, None], name='enc_padding_mask')
    if FLAGS.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size.value, None], name='enc_batch_extend_vocab')
      self._enc_batch_extend_vocab_a = tf.placeholder(tf.int32, [hps.batch_size.value, None], name='enc_batch_extend_vocab_a')
      self._enc_batch_extend_vocab_aa = tf.placeholder(tf.int32, [hps.batch_size.value, None], name='enc_batch_extend_vocab_aa')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

    # decoder part
    if hps.mode.value=="decode":
        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size.value, hps.max_dec_steps_2.value], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size.value, hps.max_dec_steps_2.value], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size.value, hps.max_dec_steps_2.value], name='dec_padding_mask')
    else:
        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size.value, hps.max_dec_steps.value], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size.value, hps.max_dec_steps.value], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size.value, hps.max_dec_steps.value], name='dec_padding_mask')

    if hps.mode.value=="decode" and hps.coverage.value:
      self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size.value, None], name='prev_coverage')
      self.prev_coverage_2 = tf.placeholder(tf.float32, [hps.batch_size.value, None], name='prev_coverage_2')
      self.prev_coverage_aa = tf.placeholder(tf.float32, [hps.batch_size.value, None], name='prev_coverage_aa')

  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.
    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    feed_dict[self._enc_batch] = batch.enc_batch
    feed_dict[self._enc_lens] = batch.enc_lens
    feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
    feed_dict[self._enc_batch_a] = batch.enc_batch_2
    feed_dict[self._enc_lens_a] = batch.enc_lens_2
    feed_dict[self._enc_padding_mask_a] = batch.enc_padding_mask_2
    feed_dict[self._enc_batch_aa] = batch.enc_batch_aa
    feed_dict[self._enc_lens_aa] = batch.enc_lens_aa
    feed_dict[self._enc_padding_mask_aa] = batch.enc_padding_mask_aa
    if FLAGS.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed_dict[self._enc_batch_extend_vocab_a] = batch.enc_batch_extend_vocab_2
      feed_dict[self._enc_batch_extend_vocab_aa] = batch.enc_batch_extend_vocab_aa
      feed_dict[self._max_art_oovs] = batch.max_art_oovs
    if not just_enc:
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
    return feed_dict
    
  def _add_encoder(self, encoder_inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder'):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=-1, values=encoder_outputs) # concatenate the forwards and backwards states

    return encoder_outputs, fw_st, bw_st
  def _add_encoder_2(self, encoder_inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder_2'):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=-1, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, fw_st, bw_st
  def _add_encoder_aa(self, encoder_inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    
    with tf.variable_scope('encoder_aa'):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=-1, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, fw_st, bw_st
  def _reduce_states(self, fw_st, bw_st, name):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim.value
    with tf.variable_scope(name):
        # Define weights and biases to reduce the cell and reduce the state
        w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

        # Apply linear layer
        old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
        old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
        new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
        new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
        return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state
        
        
      
  
  
  def _add_decoder(self, inputs):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

    Args:
      inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of scalar tensors; the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim.value, state_is_tuple=True, initializer=self.rand_unif_init)

    prev_coverage = self.prev_coverage if hps.mode.value=="decode" and hps.coverage.value else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
    prev_coverage_2 = self.prev_coverage_2 if hps.mode.value=="decode" and hps.coverage.value else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
    prev_coverage_aa = self.prev_coverage_aa if hps.mode.value=="decode" and hps.coverage.value else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
    outputs, out_state, attn_dists, p_gens, coverage, coverage_2, attn_dists_2, lambdas, coverage_aa, attn_dists_aa, lambdas_2, lambdas_3 = attention_decoder(inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell, initial_state_attention=(hps.mode.value=="decode"), pointer_gen=hps.pointer_gen.value, use_coverage=hps.coverage.value, prev_coverage=prev_coverage, prev_coverage_2=prev_coverage_2, encoder_states_2=self._enc_a_states, enc_padding_mask_2=self._enc_padding_mask_a, prev_coverage_aa=prev_coverage_aa, encoder_states_aa=self._enc_aa_states, enc_padding_mask_aa=self._enc_padding_mask_aa)

    return outputs, out_state, attn_dists, p_gens, coverage, coverage_2, attn_dists_2, coverage_aa, attn_dists_aa, lambdas, lambdas_2, lambdas_3
    
  def _calc_final_dist(self, vocab_dists, attn_dists, attn_dists_2, attn_dists_aa):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

    Returns:
      final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    with tf.variable_scope('final_distribution'):
      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
      attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, attn_dists)]
      
      #A
      attn_dists = [lambda_ * dist for (lambda_,dist) in zip(self.lambdas, attn_dists)]
      attn_dists_2 = [lambda2_ * dist for (lambda2_,dist) in zip(self.lambdas_2, attn_dists_2)]
      attn_dists_aa = [lambda3_ * dist for (lambda3_,dist) in zip(self.lambdas_3, attn_dists_aa)]
      #\A
      

      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
      extended_vsize = self._vocab.size() + self._max_art_oovs # the maximum (over the batch) size of the extended vocabulary
      extra_zeros = tf.zeros((self._hps.batch_size.value, self._max_art_oovs))
      vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use tf.scatter_nd to do the projection
      batch_nums = tf.range(0, limit=self._hps.batch_size.value) # shape (batch_size)
      batch_nums_ = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
      attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
      attn_len_2 = tf.shape(self._enc_batch_extend_vocab_a)[1]
      attn_len_aa = tf.shape(self._enc_batch_extend_vocab_aa)[1]
      
      batch_nums = tf.tile(batch_nums_, [1, attn_len]) # shape (batch_size, attn_len)
      batch_nums_2 = tf.tile(batch_nums_, [1, attn_len_2]) # shape (batch_size, attn_len_2)
      batch_nums_aa = tf.tile(batch_nums_, [1, attn_len_aa]) # shape (batch_size, attn_len_2)
      
      indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
      indices_2 = tf.stack( (batch_nums_2, self._enc_batch_extend_vocab_a), axis=2) # shape (batch_size, enc_t_2, 2)
      indices_aa = tf.stack( (batch_nums_aa, self._enc_batch_extend_vocab_aa), axis=2) # shape (batch_size, enc_t_2, 2)
      
      shape = [self._hps.batch_size.value, extended_vsize]
      attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)
      
      
      #A
      attn_dists_projected_2 = [tf.scatter_nd(indices_2, copy_dist, shape) for copy_dist in attn_dists_2]
      attn_dists_projected_aa = [tf.scatter_nd(indices_aa, copy_dist, shape) for copy_dist in attn_dists_aa]
      #\A
      

      # Add the vocab distributions and the copy distributions together to get the final distributions
      # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
      # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
      final_dists = [vocab_dist + copy_dist + copy_dist_2 + copy_dist_aa for (vocab_dist,copy_dist,copy_dist_2,copy_dist_aa) in zip(vocab_dists_extended, attn_dists_projected, attn_dists_projected_2, attn_dists_projected_aa)]
      #final_dists = [vocab_dist + copy_dist_2 + copy_dist_aa for (vocab_dist,copy_dist_2,copy_dist_aa) in zip(vocab_dists_extended, attn_dists_projected_2, attn_dists_projected_aa)]

      return final_dists

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)

  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary

    with tf.variable_scope('seq2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag.value, hps.rand_unif_init_mag.value, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std.value)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', [vsize, hps.emb_dim.value], dtype=tf.float32, initializer=self.trunc_norm_init)
        if hps.mode.value=="train": self._add_emb_vis(embedding) # add to tensorboard
        emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        
        
        #A
        emb_enc_a_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch_a)
        emb_enc_aa_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch_aa)
        #\A
        
        
        emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)

      # Add the encoder.
      enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
      
      
      #A
      enc_a_outputs, fw_st_a, bw_st_a = self._add_encoder_2(emb_enc_a_inputs, self._enc_lens_a)
      enc_aa_outputs, fw_st_aa, bw_st_aa = self._add_encoder_aa(emb_enc_aa_inputs, self._enc_lens_aa)
      #\A
      
      
      self._enc_states = enc_outputs
      
      
      #A
      self._enc_a_states = enc_a_outputs
      self._enc_aa_states = enc_aa_outputs
      #\A

      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      _dec_in_state = self._reduce_states(fw_st, bw_st, 'reduce_final_st')
      
      
      #A
      _dec_in_state_a = self._reduce_states(fw_st_a, bw_st_a, 'reduce_final_st_2')
      _dec_in_state_aa = self._reduce_states(fw_st_aa, bw_st_aa, 'reduce_final_st_aa')
      #\A
      self._dec_in_state_a = self._reduce_states(_dec_in_state_a, _dec_in_state_aa, 'reduce_dec_st_2')
      self._dec_in_state = self._reduce_states(_dec_in_state, _dec_in_state_a, 'reduce_dec_st')
      self._dec_in_state = tf.tanh(self._dec_in_state)


      # Add the decoder.
      with tf.variable_scope('decoder'):
        decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage, self.coverage_2, self.attn_dists_2, self.coverage_aa, self.attn_dists_aa, self.lambdas, self.lambdas_2, self.lambdas_3  = self._add_decoder(emb_dec_inputs) #self.lambdas

      # Add the output projection to obtain the vocabulary distribution
      with tf.variable_scope('output_projection'):
        
        w = tf.get_variable('w', [hps.emb_dim.value, hps.hidden_dim.value], dtype=tf.float32, initializer=self.trunc_norm_init)
        w2 = tf.matmul(embedding, w)
        w2 = tf.tanh(w2)
        w2 = tf.transpose(w2)
        b = tf.get_variable('b', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        vocab_scores = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
        for i,output in enumerate(decoder_outputs):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          vocab_scores.append(tf.nn.xw_plus_b(output, w2, b)) # apply the linear layer

        vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.


      # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
      if FLAGS.pointer_gen:
        final_dists = self._calc_final_dist(vocab_dists, self.attn_dists, self.attn_dists_2, self.attn_dists_aa)
      else: # final distribution is just vocabulary distribution
        final_dists = vocab_dists



      if hps.mode.value in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss'):
          if FLAGS.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
            loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
            batch_nums = tf.range(0, limit=hps.batch_size.value) # shape (batch_size)
            for dec_step, dist in enumerate(final_dists):
              targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
              indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
              gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
              losses = -tf.log(gold_probs + 1e-10)
              loss_per_step.append(losses)

            # Apply dec_padding_mask and get loss
            self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

          else: # baseline model
            self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

          tf.summary.scalar('loss', self._loss)

          # Calculate coverage loss from the attention distributions
          if hps.coverage.value:
            with tf.variable_scope('coverage_loss'):
              self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
              self._coverage_loss_2 = _coverage_loss(self.attn_dists_2, self._dec_padding_mask)
              self._coverage_loss_aa = _coverage_loss(self.attn_dists_aa, self._dec_padding_mask)
              tf.summary.scalar('coverage_loss_aa', self._coverage_loss_aa)
              tf.summary.scalar('coverage_loss_2', self._coverage_loss_2)
              tf.summary.scalar('coverage_loss', self._coverage_loss)
            self._total_loss = self._loss + hps.cov_loss_wt.value * self._coverage_loss + hps.cov_loss_wt_2.value * self._coverage_loss_2 + hps.cov_loss_wt_aa.value * self._coverage_loss_aa
            tf.summary.scalar('total_loss', self._total_loss)

    if hps.mode.value == "decode":
      # We run decode beam search mode one decoder step at a time
      assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
      final_dists = final_dists[0]
      topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size.value*2) # take the k largest probs. note batch_size=beam_size in decode mode
      self._topk_log_probs = tf.log(topk_probs)


  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._total_loss if self._hps.coverage.value else self._loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device(a):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm.value)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(self._hps.lr.value, initial_accumulator_value=self._hps.adagrad_init_acc.value)
    with tf.device(a):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device(a):
      self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode.value == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage.value:
      to_return['coverage_loss_aa'] = self._coverage_loss_aa
      to_return['coverage_loss_2'] = self._coverage_loss_2
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage.value:
      to_return['coverage_loss_aa'] = self._coverage_loss_aa
      to_return['coverage_loss_2'] = self._coverage_loss_2
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    (enc_states, dec_in_state, enc_states_2, enc_states_aa, global_step) = sess.run([self._enc_states, self._dec_in_state, self._enc_a_states, self._enc_aa_states, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, dec_in_state, enc_states_2, enc_states_aa

  def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage, prev_coverage_2, enc_states_2, prev_coverage_aa, enc_states_aa):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      new_states: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

    beam_size = len(dec_init_states)

    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    feed = {
        self._enc_states: enc_states,
        self._enc_a_states: enc_states_2,
        self._enc_aa_states: enc_states_aa,
        self._enc_padding_mask: batch.enc_padding_mask,
        self._enc_padding_mask_a: batch.enc_padding_mask_2,
        self._enc_padding_mask_aa: batch.enc_padding_mask_aa,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens])),
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "attn_dists": self.attn_dists,
      "attn_dists_2": self.attn_dists_2,
      "attn_dists_aa": self.attn_dists_aa
    }
    if FLAGS.pointer_gen:
      feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self._enc_batch_extend_vocab_a] = batch.enc_batch_extend_vocab_2
      feed[self._enc_batch_extend_vocab_aa] = batch.enc_batch_extend_vocab_aa
      feed[self._max_art_oovs] = batch.max_art_oovs
      to_return['p_gens'] = self.p_gens
      to_return['lambdas'] = self.lambdas
      to_return['lambdas_2'] = self.lambdas_2
      to_return['lambdas_3'] = self.lambdas_3

    if self._hps.coverage.value:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      feed[self.prev_coverage_2] = np.stack(prev_coverage_2, axis=0)
      feed[self.prev_coverage_aa] = np.stack(prev_coverage_aa, axis=0)
      to_return['coverage'] = self.coverage
      to_return['coverage_2'] = self.coverage_2
      to_return['coverage_aa'] = self.coverage_aa

    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in range(beam_size)]

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['attn_dists'])==1
    attn_dists = results['attn_dists'][0].tolist()
    
    assert len(results['attn_dists_2'])==1
    attn_dists_2 = results['attn_dists_2'][0].tolist()
    
    assert len(results['attn_dists_aa'])==1
    attn_dists_aa = results['attn_dists_aa'][0].tolist()

    if FLAGS.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(results['p_gens'])==1
      p_gens = results['p_gens'][0].tolist()
      assert len(results['lambdas'])==1
      lambdas = results['lambdas'][0].tolist()
      assert len(results['lambdas_2'])==1
      lambdas_2 = results['lambdas_2'][0].tolist()
      assert len(results['lambdas_3'])==1
      lambdas_3 = results['lambdas_3'][0].tolist()
    else:
      p_gens = [None for _ in range(beam_size)]
      lambdas = [None for _ in range(beam_size)]
      lambdas_2 = [None for _ in range(beam_size)]
      lambdas_3 = [None for _ in range(beam_size)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      new_coverage_2 = results['coverage_2'].tolist()
      new_coverage_aa = results['coverage_aa'].tolist()
      assert len(new_coverage) == beam_size
      assert len(new_coverage_2) == beam_size
      assert len(new_coverage_aa) == beam_size
    else:
      new_coverage = [None for _ in range(beam_size)]
      new_coverage_2 = [None for _ in range(beam_size)]
      new_coverage_aa = [None for _ in range(beam_size)]

    return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage, new_coverage_2, new_coverage_aa, attn_dists_2, attn_dists_aa, lambdas, lambdas_2, lambdas_3


def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/(dec_lens + 1e-10) # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss
