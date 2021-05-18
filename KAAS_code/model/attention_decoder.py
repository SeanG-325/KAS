import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
def attention_decoder(decoder_inputs, initial_state, encoder_states, enc_padding_mask, cell, initial_state_attention=False, pointer_gen=True, use_coverage=False, prev_coverage=None, encoder_states_2=None, enc_padding_mask_2=None, prev_coverage_2=None, encoder_states_aa=None, enc_padding_mask_aa=None, prev_coverage_aa=None):
  with variable_scope.variable_scope("attention_decoder") as scope:
    batch_size = encoder_states.get_shape()[0].value # if this line fails, it's because the batch size isn't defined
    attn_size = encoder_states.get_shape()[2].value # if this line fails, it's because the attention length isn't defined
    encoder_states = tf.expand_dims(encoder_states, axis=2) # now is shape (batch_size, attn_len, 1, attn_size)
    if encoder_states_2 is not None:
        encoder_states_2 = tf.expand_dims(encoder_states_2, axis=2)
    if encoder_states_aa is not None:
        encoder_states_aa = tf.expand_dims(encoder_states_aa, axis=2)
    attention_vec_size = attn_size
    W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
    encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME") # shape (batch_size,attn_length,1,attention_vec_size)
    #A
    if encoder_states_2 is not None:
        W_h_2 = variable_scope.get_variable("W_h_2", [1, 1, attn_size, attention_vec_size])
        encoder_features_2 = nn_ops.conv2d(encoder_states_2, W_h_2, [1, 1, 1, 1], "SAME") # attn_length is diff
    if encoder_states_aa is not None:
        W_h_aa = variable_scope.get_variable("W_h_aa", [1, 1, attn_size, attention_vec_size])
        encoder_features_aa = nn_ops.conv2d(encoder_states_aa, W_h_aa, [1, 1, 1, 1], "SAME") # attn_length is diff
    W_h_c = variable_scope.get_variable("W_h_c", [1, 1, attn_size, attention_vec_size])
    #\A
    # Get the weight vectors v and w_c (w_c is for coverage)
    v = variable_scope.get_variable("v", [attention_vec_size])
    if use_coverage:
      with variable_scope.variable_scope("coverage"):
        w_c = variable_scope.get_variable("w_c", [1, 1, 1, attention_vec_size])
        w_c_2 = variable_scope.get_variable("w_c_2", [1, 1, 1, attention_vec_size])
        w_c_aa = variable_scope.get_variable("w_c_aa", [1, 1, 1, attention_vec_size])

    if prev_coverage is not None: # for beam search mode with coverage
      # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
      prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage,2),3)
    if prev_coverage_2 is not None:
      # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
      prev_coverage_2 = tf.expand_dims(tf.expand_dims(prev_coverage_2,2),3)
    if prev_coverage_aa is not None:
      # reshape from (batch_size, attn_length) to (batch_size, attn_len, 1, 1)
      prev_coverage_aa = tf.expand_dims(tf.expand_dims(prev_coverage_aa,2),3)
    def attention(decoder_state, coverage=None):
      with variable_scope.variable_scope("Attention"):
        decoder_features = linear(decoder_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)
        

        def masked_attention(e):
          """Take softmax of e then apply enc_padding_mask and re-normalize"""
          attn_dist = nn_ops.softmax(e) # take softmax. shape (batch_size, attn_length)
          attn_dist *= enc_padding_mask # apply mask
          masked_sums = tf.reduce_sum(attn_dist, axis=1) # shape (batch_size)
          return attn_dist / tf.reshape(masked_sums, [-1, 1]) # re-normalize

        if use_coverage and coverage is not None: # non-first step of coverage
          # Multiply coverage vector by w_c to get coverage_features.
          coverage_features = nn_ops.conv2d(coverage, w_c, [1, 1, 1, 1], "SAME") # c has shape (batch_size, attn_length, 1, attention_vec_size)

          # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
          e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features + coverage_features), [2, 3])  # shape (batch_size,attn_length)

          # Calculate attention distribution
          attn_dist = masked_attention(e)

          # Update coverage vector
          coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
        else:
          # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
          e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features), [2, 3]) # calculate e

          # Calculate attention distribution
          attn_dist = masked_attention(e)

          if use_coverage: # first step of training
            coverage = tf.expand_dims(tf.expand_dims(attn_dist,2),2) # initialize coverage

        # Calculate the context vector from attn_dist and encoder_states
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2]) # shape (batch_size, attn_size).
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

      return context_vector, attn_dist, coverage
      
      
    def attention_2(decoder_state, coverage_2=None):
      with variable_scope.variable_scope("Attention_2"):
        # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
        decoder_features = linear(decoder_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)
        

        def masked_attention(e):
          """Take softmax of e then apply enc_padding_mask and re-normalize"""
          attn_dist = nn_ops.softmax(e) # take softmax. shape (batch_size, attn_length)
          attn_dist *= enc_padding_mask_2 # apply mask
          masked_sums = tf.reduce_sum(attn_dist, axis=1) # shape (batch_size)
          return attn_dist / tf.reshape(masked_sums, [-1, 1]) # re-normalize
        
        if use_coverage and coverage_2 is not None: # non-first step of coverage
          # Multiply coverage vector by w_c to get coverage_features.
          coverage_features_2 = nn_ops.conv2d(coverage_2, w_c_2, [1, 1, 1, 1], "SAME") # c has shape (batch_size, attn_length, 1, attention_vec_size)

          # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
          e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features_2 + decoder_features + coverage_features_2), [2, 3])  # shape (batch_size,attn_length)

          # Calculate attention distribution
          attn_dist = masked_attention(e)

          # Update coverage vector
          coverage_2 += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
        else:
          # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
          e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features_2 + decoder_features), [2, 3]) # calculate e

          # Calculate attention distribution
          attn_dist = masked_attention(e)

          if use_coverage: # first step of training
            coverage_2 = tf.expand_dims(tf.expand_dims(attn_dist,2),2) # initialize coverage       
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states_2, [1, 2]) # shape (batch_size, attn_size).
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

      return context_vector, attn_dist, coverage_2
      
    def attention_aa(decoder_state, coverage_aa=None):
      with variable_scope.variable_scope("Attention_aa"):
        # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
        decoder_features = linear(decoder_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)
        

        def masked_attention(e):
          attn_dist = nn_ops.softmax(e) # take softmax. shape (batch_size, attn_length)
          attn_dist *= enc_padding_mask_aa # apply mask
          masked_sums = tf.reduce_sum(attn_dist, axis=1) # shape (batch_size)
          return attn_dist / tf.reshape(masked_sums, [-1, 1]) # re-normalize
        
        if use_coverage and coverage_aa is not None: # non-first step of coverage
          # Multiply coverage vector by w_c to get coverage_features.
          coverage_features_aa = nn_ops.conv2d(coverage_aa, w_c_aa, [1, 1, 1, 1], "SAME") # c has shape (batch_size, attn_length, 1, attention_vec_size)

          # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
          e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features_aa + decoder_features + coverage_features_aa), [2, 3])  # shape (batch_size,attn_length)

          # Calculate attention distribution
          attn_dist = masked_attention(e)

          # Update coverage vector
          coverage_aa += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
        else:
          # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
          e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features_aa + decoder_features), [2, 3]) # calculate e

          # Calculate attention distribution
          attn_dist = masked_attention(e)

          if use_coverage: # first step of training
            coverage_aa = tf.expand_dims(tf.expand_dims(attn_dist,2),2) # initialize coverage       
            
            
            
        # Calculate the context vector from attn_dist and encoder_states
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states_aa, [1, 2]) # shape (batch_size, attn_size).
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

      return context_vector, attn_dist, coverage_aa
      
    def attention_con(decoder_state, context, coverage_aa=None):
        context = tf.expand_dims(context, axis=2)
      
        context_features = nn_ops.conv2d(context, W_h_c, [1, 1, 1, 1], "SAME") # attn_length is diff
        with variable_scope.variable_scope("Attention_con"):
            # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
            decoder_features = linear(decoder_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
            decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)
        

   
        if use_coverage and coverage_aa is not None: # non-first step of coverage
            # Multiply coverage vector by w_c to get coverage_features.
            coverage_features_aa = nn_ops.conv2d(coverage_aa, w_c_aa, [1, 1, 1, 1], "SAME") # c has shape (batch_size, attn_length, 1, attention_vec_size)

            # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
            e = math_ops.reduce_sum(v * math_ops.tanh(context_features + decoder_features), [2, 3])  # shape (batch_size,attn_length)
            attn_dist = nn_ops.softmax(e)
            attn_dist= attn_dist / tf.reshape(tf.reduce_sum(attn_dist, axis=1), [-1, 1])
            # Calculate attention distribution

            # Update coverage vector
            coverage_aa += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
        else:
            # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
            e = math_ops.reduce_sum(v * math_ops.tanh(context_features + decoder_features), [2, 3]) # calculate e
            attn_dist = nn_ops.softmax(e)
            attn_dist = attn_dist / tf.reshape(tf.reduce_sum(attn_dist, axis=1), [-1, 1])
            # Calculate attention distribution

            if use_coverage: # first step of training
                coverage_aa = tf.expand_dims(tf.expand_dims(attn_dist,2),2) # initialize coverage       
        # Calculate the context vector from attn_dist and encoder_states
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * context, [1, 2]) # shape (batch_size, attn_size).
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

        return context_vector, attn_dist, coverage_aa
    outputs = []
    attn_dists = []
    
    
    #A
    outputs_2 = []
    attn_dists_2 = []
    lambdas = []
    
    outputs_aa = []
    attn_dists_aa = []
    lambdas_2 = []
    #\A
    lambdas_3 = []
    p_gens = []
    state = initial_state
    coverage = prev_coverage # initialize coverage to None or whatever was passed in
    coverage_2 = prev_coverage_2
    coverage_aa = prev_coverage_aa
    
    context_vector = array_ops.zeros([batch_size, attn_size])
    context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
    #A
    context_vector_2 = array_ops.zeros([batch_size, attn_size])
    context_vector_2.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
    
    context_vector_aa = array_ops.zeros([batch_size, attn_size])
    context_vector_aa.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
    
    context_ = array_ops.zeros([batch_size, attn_size])
    context_.set_shape([None, attn_size])
    #\A
    if initial_state_attention: # true in decode mode
      # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
      context_vector, _, coverage = attention(initial_state, coverage) # in decode mode, this is what updates the coverage vector
      #A
      context_vector_2, _, coverage_2 = attention_2(initial_state, coverage_2)
      context_vector_aa, _, coverage_aa = attention_aa(initial_state, coverage_aa)
      context = tf.stack([context_vector, context_vector_2, context_vector_aa], axis=1)
      
      context_, _, coverage_ = attention_con(initial_state, context, coverage)
      
    for i, inp in enumerate(decoder_inputs):
      tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(decoder_inputs))
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      
      # Merge input and previous attentions into one vector x of the same size as inp
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      #x = linear([inp] + [context_vector] + [context_vector_2] + [context_vector_aa], input_size, True)

      #A
      x = linear([inp] + [context_], input_size, True)
      #\A
      

      # Run the decoder RNN cell. cell_output = decoder state
      cell_output, state = cell(x, state)
      

      # Run the attention mechanism.
      if i == 0 and initial_state_attention:  # always true in decode mode
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True): # you need this because you've already run the initial attention(...) call
          context_vector, attn_dist, _ = attention(state, coverage) # don't allow coverage to update
          #A
          context_vector_2, attn_dist_2, _ = attention_2(state, coverage_2)
          context_vector_aa, attn_dist_aa, _ = attention_aa(state, coverage_aa)
          context = tf.stack([context_vector, context_vector_2, context_vector_aa], axis=1)
          context_, _, coverage_ = attention_con(state, context, coverage)
          #\A
      else:
        context_vector, attn_dist, coverage = attention(state, coverage)
        #A
        context_vector_2, attn_dist_2, coverage_2 = attention_2(state, coverage_2)
        context_vector_aa, attn_dist_aa, coverage_aa = attention_aa(state, coverage_aa)
        context = tf.stack([context_vector, context_vector_2, context_vector_aa], axis=1)
        context_, _, coverage_ = attention_con(state, context, coverage)
        
        #\A
      attn_dists.append(attn_dist)
      #A
      attn_dists_2.append(attn_dist_2)
      attn_dists_aa.append(attn_dist_aa)
      if pointer_gen:
        with tf.variable_scope('calculate_pgen'):
          p_gen = linear([context_, state.c, state.h, x], 1, True) # a scalar
          p_gen = tf.sigmoid(p_gen)
          p_gens.append(p_gen)
      
      
      with tf.variable_scope('calculate_lambda'):
        _lambda_ = linear([state.c, state.h, inp, context_vector], 1, True)
        _lambda_ = tf.sigmoid(_lambda_)
        
      with tf.variable_scope('calculate_lambda2'):
        _lambda2_ = linear([state.c, state.h, inp, context_vector_2], 1, True)
        _lambda2_ = tf.sigmoid(_lambda2_)
        
      with tf.variable_scope('calculate_lambda3'):
        _lambda3_ = linear([state.c, state.h, inp, context_vector_aa], 1, True)
        _lambda3_ = tf.sigmoid(_lambda3_)
      
      lambda_ = _lambda_ / (_lambda_ + _lambda2_ + _lambda3_) * (1 - p_gen)
      lambda2_ = _lambda2_ / (_lambda_ + _lambda2_ + _lambda3_) * (1 - p_gen)
      lambda3_ = _lambda3_ / (_lambda_ + _lambda2_ + _lambda3_) * (1 - p_gen)
      #\A
      
      lambdas.append(lambda_)
      lambdas_2.append(lambda2_)
      lambdas_3.append(lambda3_)
      # Calculate p_gen
      

      # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
      # This is V[s_t, h*_t] + b in the paper
      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + [context_vector] + [context_vector_2] + [context_vector_aa], cell.output_size, True)
        #output = linear([cell_output] + [context_vector] + [context_vector_aa], cell.output_size, True)
        
      outputs.append(output)
      
    #A
    '''
    if decoder_inputs_2 is not None:
        for i, inp in enumerate(decoder_inputs_2):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x_2 = linear([inp] + [context_vector_2], input_size, True)
            # Run the decoder RNN cell. cell_output = decoder state
            cell_output_2, state_2 = cell(x_2, initial_state_2)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:  # always true in decode mode
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True): # you need this because you've already run the initial attention(...) call
                    context_vector_2, attn_dist_2, _ = attention(state_2, coverage_2) # don't allow coverage to update
            else:
                context_vector_2, attn_dist_2, coverage_2 = attention(state_2, coverage_2)
            attn_dists_2.append(attn_dist_2)
            
            with variable_scope.variable_scope("AttnOutputProjection"):
                output_2 = linear([cell_output_2] + [context_vector_2], cell_2.output_size, True)
            outputs_2.append(output_2) 
            '''
    #\A
    
    # If using coverage, reshape it
    if coverage is not None:
      coverage = array_ops.reshape(coverage, [batch_size, -1])
    if coverage_2 is not None:
      coverage_2 = array_ops.reshape(coverage_2, [batch_size, -1])
    if coverage_aa is not None:
      coverage_aa = array_ops.reshape(coverage_aa, [batch_size, -1])

    return outputs, state, attn_dists, p_gens, coverage, coverage_2, attn_dists_2, lambdas, coverage_aa, attn_dists_aa, lambdas_2, lambdas_3



def linear(args, output_size, bias, bias_start=0.0, scope=None):
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return res + bias_term
