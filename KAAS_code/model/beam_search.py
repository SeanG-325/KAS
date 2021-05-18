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

"""This file contains code to run beam search decoding"""

import tensorflow as tf
import numpy as np
import data

FLAGS = tf.app.flags.FLAGS

class Hypothesis(object):
  """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

  def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage, coverage_2, attn_dists_2, lambdas, coverage_aa, attn_dists_aa, lambdas_2, lambdas_3):
    """Hypothesis constructor.

    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
      attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
      p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
      coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
    """
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.attn_dists = attn_dists
    self.p_gens = p_gens
    self.coverage = coverage
    self.coverage_2 = coverage_2 
    self.attn_dists_2 = attn_dists_2
    self.lambdas=lambdas
    self.coverage_aa = coverage_aa 
    self.attn_dists_aa = attn_dists_aa
    self.lambdas_2=lambdas_2
    self.lambdas_3=lambdas_3

  def extend(self, token, log_prob, state, attn_dist, p_gen, coverage, coverage_2, attn_dist_2, lambda_, coverage_aa, attn_dist_aa, lambda2_, lambda3_):
    """Return a NEW hypothesis, extended with the information from the latest step of beam search.

    Args:
      token: Integer. Latest token produced by beam search.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
      attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
      p_gen: Generation probability on latest step. Float.
      coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
    Returns:
      New Hypothesis for next step.
    """
    return Hypothesis(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      attn_dists = self.attn_dists + [attn_dist],
                      p_gens = self.p_gens + [p_gen],
                      coverage = coverage,
                      coverage_2 = coverage_2,
                      attn_dists_2 = self.attn_dists_2 + [attn_dist_2],
                      lambdas = self.lambdas + [lambda_],
                      coverage_aa = coverage_aa,
                      attn_dists_aa = self.attn_dists_aa + [attn_dist_aa],
                      lambdas_2 = self.lambdas_2 + [lambda2_],
                      lambdas_3 = self.lambdas_3 + [lambda3_])

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def log_prob(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
    return sum(self.log_probs)

  @property
  def avg_log_prob(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
    return self.log_prob / len(self.tokens)


def run_beam_search(sess, model, vocab, batch):
  """Performs beam search decoding on the given example.

  Args:
    sess: a tf.Session
    model: a seq2seq model
    vocab: Vocabulary object
    batch: Batch object that is the same example repeated across the batch

  Returns:
    best_hyp: Hypothesis object; the best hypothesis found by beam search.
  """
  # Run the encoder to get the encoder hidden states and decoder initial state
  enc_states, dec_in_state, enc_states_2, enc_states_aa = model.run_encoder(sess, batch)
  # dec_in_state is a LSTMStateTuple
  # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

  # Initialize beam_size-many hyptheses
  hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                     log_probs=[0.0],
                     state=dec_in_state,
                     attn_dists=[],
                     p_gens=[],
                     coverage=np.zeros([batch.enc_batch.shape[1]]), # zero vector of length attention_length
                     coverage_2=np.zeros([batch.enc_batch_2.shape[1]]),
                     attn_dists_2=[],
                     lambdas=[],
                     coverage_aa=np.zeros([batch.enc_batch_aa.shape[1]]),
                     attn_dists_aa=[],
                     lambdas_2=[],
                     lambdas_3=[]
                     ) for _ in range(FLAGS.beam_size)]
  results = [] # this will contain finished hypotheses (those that have emitted the [STOP] token)

  steps = 0
  while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
    latest_tokens = [h.latest_token for h in hyps] # latest token produced by each hypothesis
    latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens] # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
    states = [h.state for h in hyps] # list of current decoder states of the hypotheses
    prev_coverage = [h.coverage for h in hyps] # list of coverage vectors (or None)
    prev_coverage_2 = [h.coverage_2 for h in hyps]
    prev_coverage_aa = [h.coverage_aa for h in hyps]

    # Run one step of the decoder to get the new info
    (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage, new_coverage_2, new_coverage_aa, attn_dists_2, attn_dists_aa, lambdas, lambdas_2, lambdas_3) = model.decode_onestep(sess=sess,
                        batch=batch,
                        latest_tokens=latest_tokens,
                        enc_states=enc_states,###
                        dec_init_states=states,
                        prev_coverage=prev_coverage,
                        prev_coverage_2=prev_coverage_2,
                        enc_states_2=enc_states_2,
                        prev_coverage_aa=prev_coverage_aa,
                        enc_states_aa=enc_states_aa
                        )

    # Extend each hypothesis and collect them all in all_hyps
    all_hyps = []
    num_orig_hyps = 1 if steps == 0 else len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
    for i in range(num_orig_hyps):
      h, new_state, attn_dist, p_gen, new_coverage_i, new_coverage_2_i, lambdas_i, attn_dist_2, new_coverage_aa_i, lambdas_2_i, attn_dist_aa, lambdas_3_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], new_coverage[i], new_coverage_2[i], lambdas[i], attn_dists_2[i], new_coverage_aa[i], lambdas_2[i], attn_dists_aa[i], lambdas_3[i]  # take the ith hypothesis and new decoder state info
      for j in range(FLAGS.beam_size * 2):  # for each of the top 2*beam_size hyps:
        # Extend the ith hypothesis with the jth option
        new_hyp = h.extend(token=topk_ids[i, j],
                           log_prob=topk_log_probs[i, j],
                           state=new_state,
                           attn_dist=attn_dist,
                           p_gen=p_gen,
                           lambda_=lambdas_i,
                           attn_dist_2=attn_dist_2,
                           coverage=new_coverage_i,
                           coverage_2=new_coverage_2_i,
                           lambda2_=lambdas_2_i,
                           lambda3_=lambdas_3_i,
                           attn_dist_aa=attn_dist_aa,
                           coverage_aa=new_coverage_aa_i)
        all_hyps.append(new_hyp)

    # Filter and collect any hypotheses that have produced the end token.
    hyps = [] # will contain hypotheses for the next step
    for h in sort_hyps(all_hyps): # in order of most likely h
      if h.latest_token == vocab.word2id(data.STOP_DECODING): # if stop token is reached...
        # If this hypothesis is sufficiently long, put in results. Otherwise discard.
        if steps >= FLAGS.min_dec_steps:
          results.append(h)
      else: # hasn't reached stop token, so continue to extend this hypothesis
        hyps.append(h)
      if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
        # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
        break

    steps += 1

  # At this point, either we've got beam_size results, or we've reached maximum decoder steps

  if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
    results = hyps

  # Sort hypotheses by average log probability
  hyps_sorted = sort_hyps(results)

  # Return the hypothesis with highest average log prob
  return hyps_sorted[0]

def sort_hyps(hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
