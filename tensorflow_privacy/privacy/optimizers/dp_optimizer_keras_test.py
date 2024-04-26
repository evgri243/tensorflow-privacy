# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras_vectorized


class DPOptimizerComputeGradientsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for _compute_gradients method."""

  def _loss(self, val0, val1):
    """Loss function whose derivative w.r.t val1 is val1 - val0."""
    return 0.5 * tf.reduce_sum(
        input_tensor=tf.math.squared_difference(val0, val1), axis=1)

  @parameterized.named_parameters(
      ('DPGradientDescent_1', dp_optimizer_keras.DPKerasSGDOptimizer, 1),
      ('DPGradientDescent_None', dp_optimizer_keras.DPKerasSGDOptimizer, None),
      ('DPAdam_2', dp_optimizer_keras.DPKerasAdamOptimizer, 2),
      ('DPAdagrad _4', dp_optimizer_keras.DPKerasAdagradOptimizer, 4),
      ('DPGradientDescentVectorized_1',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 1),
      ('DPAdamVectorized_2',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdamOptimizer, 2),
      ('DPAdagradVectorized_4',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdagradOptimizer, 4),
      ('DPAdagradVectorized_None',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdagradOptimizer, None),
  )
  def testBaselineWithCallableLossNoNoise(self, optimizer_class,
                                          num_microbatches):
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0])
    data0 = tf.Variable([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0]])
    data1 = tf.Variable([[8.0], [2.0], [3.0], [1.0]])
    expected_grad0 = [-2.5, -2.5]
    expected_grad1 = [-0.5]

    optimizer = optimizer_class(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        learning_rate=2.0)

    loss = lambda: self._loss(data0, var0) + self._loss(data1, var1)

    grads_and_vars = optimizer._compute_gradients(loss, [var0, var1])

    self.assertAllCloseAccordingToType(expected_grad0, grads_and_vars[0][0])
    self.assertAllCloseAccordingToType(expected_grad1, grads_and_vars[1][0])

  def testKerasModelBaselineSaving(self):
    """Tests that DP optimizers work with tf.keras.Model."""

    model = tf.keras.models.Sequential(layers=[
        tf.keras.layers.Dense(
            1,
            activation='linear',
            name='dense',
            kernel_initializer='zeros',
            bias_initializer='zeros')
    ])

    optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=None,
        learning_rate=0.05)
    loss = tf.keras.losses.MeanSquaredError(reduction='none')
    model.compile(optimizer, loss)

    true_weights = np.array([[-5], [4], [3], [2]]).astype(np.float32)
    true_bias = np.array([6.0]).astype(np.float32)
    train_data = np.random.normal(scale=3.0, size=(1000, 4)).astype(np.float32)
    train_labels = np.matmul(train_data,
                             true_weights) + true_bias + np.random.normal(
                                 scale=0.0, size=(1000, 1)).astype(np.float32)

    model.fit(train_data, train_labels, batch_size=8, epochs=1, shuffle=False)

    tempdir = self.create_tempdir()
    model.save(tempdir, save_format='tf')

  def testKerasModelBaselineAfterSavingLoading(self):
    """Tests that DP optimizers work with tf.keras.Model."""

    model = tf.keras.models.Sequential(layers=[
        tf.keras.layers.Dense(
            1,
            activation='linear',
            name='dense',
            kernel_initializer='zeros',
            bias_initializer='zeros')
    ])

    optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=None,
        learning_rate=0.05)
    loss = tf.keras.losses.MeanSquaredError(reduction='none')
    model.compile(optimizer, loss)

    true_weights = np.array([[-5], [4], [3], [2]]).astype(np.float32)
    true_bias = np.array([6.0]).astype(np.float32)
    train_data = np.random.normal(scale=3.0, size=(1000, 4)).astype(np.float32)
    train_labels = np.matmul(train_data,
                             true_weights) + true_bias + np.random.normal(
                                 scale=0.0, size=(1000, 1)).astype(np.float32)

    model.predict(train_data, batch_size=8)
    tempdir = self.create_tempdir()
    model.save(tempdir, save_format='tf')
    model.load_weights(tempdir)

    model.fit(train_data, train_labels, batch_size=8, epochs=1, shuffle=False)

  @parameterized.named_parameters(('1', 1), ('None', None))
  def testKerasModelBaselineNoNoise(self, num_microbatches):
    """Tests that DP optimizers work with tf.keras.Model."""

    model = tf.keras.models.Sequential(layers=[
        tf.keras.layers.Dense(
            1,
            activation='linear',
            name='dense',
            kernel_initializer='zeros',
            bias_initializer='zeros')
    ])

    optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        learning_rate=0.05)
    loss = tf.keras.losses.MeanSquaredError(reduction='none')
    model.compile(optimizer, loss)

    true_weights = np.array([[-5], [4], [3], [2]]).astype(np.float32)
    true_bias = np.array([6.0]).astype(np.float32)
    train_data = np.random.normal(scale=3.0, size=(1000, 4)).astype(np.float32)
    train_labels = np.matmul(train_data,
                             true_weights) + true_bias + np.random.normal(
                                 scale=0.0, size=(1000, 1)).astype(np.float32)

    model.fit(train_data, train_labels, batch_size=8, epochs=1, shuffle=False)

    self.assertAllClose(model.get_weights()[0], true_weights, atol=0.05)
    self.assertAllClose(model.get_weights()[1], true_bias, atol=0.05)

  @parameterized.named_parameters(
      ('DPGradientDescent_1', dp_optimizer_keras.DPKerasSGDOptimizer, 1),
      ('DPGradientDescent_None', dp_optimizer_keras.DPKerasSGDOptimizer, None),
      ('DPAdam_2', dp_optimizer_keras.DPKerasAdamOptimizer, 2),
      ('DPAdagrad_4', dp_optimizer_keras.DPKerasAdagradOptimizer, 4),
      ('DPGradientDescentVectorized_1',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 1),
      ('DPAdamVectorized_2',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdamOptimizer, 2),
      ('DPAdagradVectorized_4',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdagradOptimizer, 4),
      ('DPAdagradVectorized_None',
       dp_optimizer_keras_vectorized.VectorizedDPKerasAdagradOptimizer, None),
  )
  def testBaselineWithTensorLossNoNoise(self, optimizer_class,
                                        num_microbatches):
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0])
    data0 = tf.Variable([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0]])
    data1 = tf.Variable([[8.0], [2.0], [3.0], [1.0]])
    expected_grad0 = [-2.5, -2.5]
    expected_grad1 = [-0.5]

    optimizer = optimizer_class(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        learning_rate=2.0)

    tape = tf.GradientTape()
    with tape:
      loss = self._loss(data0, var0) + self._loss(data1, var1)

    grads_and_vars = optimizer._compute_gradients(loss, [var0, var1], tape=tape)
    self.assertAllCloseAccordingToType(expected_grad0, grads_and_vars[0][0])
    self.assertAllCloseAccordingToType(expected_grad1, grads_and_vars[1][0])

  @parameterized.named_parameters(
      ('DPGradientDescent_False', dp_optimizer_keras.DPKerasSGDOptimizer,
       False),
      ('DPGradientDescentVectorized_False',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, False),
      ('DPFTRLTreeAggregation_True',
       dp_optimizer_keras.DPFTRLTreeAggregationOptimizer, True),
  )
  def testClippingNorm(self, optimizer_class, requires_varlist):
    var0 = tf.Variable([0.0, 0.0])
    data0 = tf.Variable([[3.0, 4.0], [6.0, 8.0]])

    varlist_kwarg = {'var_list_or_model': [var0]} if requires_varlist else {}

    optimizer = optimizer_class(
        l2_norm_clip=1.0,
        noise_multiplier=0.0,
        num_microbatches=1,
        **varlist_kwarg,
        learning_rate=2.0)

    loss = lambda: self._loss(data0, var0)
    # Expected gradient is sum of differences.
    grads_and_vars = optimizer._compute_gradients(loss, [var0])
    self.assertAllCloseAccordingToType([-0.6, -0.8], grads_and_vars[0][0])

  @parameterized.named_parameters(
      ('DPGradientDescent_1', dp_optimizer_keras.DPKerasSGDOptimizer, 1, False),
      ('DPGradientDescent_2', dp_optimizer_keras.DPKerasSGDOptimizer, 2, False),
      ('DPGradientDescent_4', dp_optimizer_keras.DPKerasSGDOptimizer, 4, False),
      ('DPGradientDescentVectorized',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 1, False),
      ('DPFTRLTreeAggregation_4',
       dp_optimizer_keras.DPFTRLTreeAggregationOptimizer, 4, True))
  def testClippingNormMultipleVariables(self, cls, num_microbatches,
                                        requires_varlist):
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0])
    data0 = tf.Variable([[3.0, 6.0], [5.0, 6.0], [4.0, 8.0], [-1.0, 0.0]])
    data1 = tf.Variable([[8.0], [2.0], [3.0], [1.0]])
    l2_clip_norm = 2.5

    varlist_kwarg = {
        'var_list_or_model': [var0, var1]
    } if requires_varlist else {}

    opt = cls(
        l2_norm_clip=l2_clip_norm,
        noise_multiplier=0.0,
        num_microbatches=num_microbatches,
        learning_rate=2.0,
        **varlist_kwarg)

    loss = lambda: self._loss(data0, var0) + self._loss(data1, var1)

    # Expected gradient is sum of differences.
    grads_and_vars = opt._compute_gradients(loss, [var0, var1])

    # Compute expected gradients.
    batch_size = data0.shape[0]
    grad0 = (data0 - var0).numpy()
    grad1 = (data1 - var1).numpy()
    grads = np.concatenate([grad0, grad1], axis=1)

    grads = np.reshape(
        grads, (num_microbatches, int(batch_size / num_microbatches), -1))
    grads = np.mean(grads, axis=1)

    norms = np.apply_along_axis(np.linalg.norm, axis=1, arr=grads)
    grad_factors = l2_clip_norm / np.maximum(l2_clip_norm, norms)

    scaled_grads = grads * grad_factors[:, None]
    mean_scaled_grads = -np.mean(scaled_grads, axis=0)
    expected0, expected1 = np.split(mean_scaled_grads, [2], axis=0)

    # Compare expected with actual gradients.
    self.assertAllCloseAccordingToType(expected0, grads_and_vars[0][0])
    self.assertAllCloseAccordingToType(expected1, grads_and_vars[1][0])

  @parameterized.named_parameters(
      ('DPGradientDescent_2_4_1', dp_optimizer_keras.DPKerasSGDOptimizer, 2.0,
       4.0, 1, False),
      ('DPGradientDescent_4_1_4', dp_optimizer_keras.DPKerasSGDOptimizer, 4.0,
       1.0, 4, False),
      ('DPGradientDescentVectorized_2_4_1',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, 2.0, 4.0, 1,
       False), ('DPGradientDescentVectorized_4_1_4',
                dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer,
                4.0, 1.0, 4, False),
      ('DPFTRLTreeAggregation_2_4_1',
       dp_optimizer_keras.DPFTRLTreeAggregationOptimizer, 2.0, 4.0, 1, True))
  def testNoiseMultiplier(self, optimizer_class, l2_norm_clip, noise_multiplier,
                          num_microbatches, requires_varlist):
    var0 = tf.Variable(tf.zeros([1000], dtype=tf.float32))
    data0 = tf.Variable(tf.zeros([16, 1000], dtype=tf.float32))

    varlist_kwarg = {'var_list_or_model': [var0]} if requires_varlist else {}

    optimizer = optimizer_class(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        **varlist_kwarg,
        learning_rate=2.0)

    loss = lambda: self._loss(data0, var0)
    grads_and_vars = optimizer._compute_gradients(loss, [var0])
    grads = grads_and_vars[0][0].numpy()

    # Test standard deviation is close to l2_norm_clip * noise_multiplier.

    self.assertNear(
        np.std(grads), l2_norm_clip * noise_multiplier / num_microbatches, 0.5)


class SimpleEmbeddingModel(tf.keras.Model):
  """Simple embedding model."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.embed_layer = tf.keras.layers.Embedding(
        name='embedding',
        input_dim=10,  # vocabulary size.
        output_dim=6,  # embedding size.
        embeddings_initializer='uniform',
        input_length=4)  # sequence length.
    self.pool_layer = tf.keras.layers.Dense(
        name='pooler',
        units=6,
        activation='tanh',
        kernel_initializer='zeros',
        bias_initializer='zeros')
    self.probs_layer = tf.keras.layers.Dense(
        units=1, activation='softmax', name='classification')

  def call(self, inputs, training=None):
    # The shape of the sequence output from the embedding layer is
    # [batch_size, sequence_length, embedding_size]
    sequence_output = self.embed_layer(inputs)
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    # The shape of the pooled output from the embedding layer is
    # [batch_size, embedding_size]
    pooled_output = self.pool_layer(first_token_tensor)
    return sequence_output, pooled_output


def keras_embedding_model_fn(optimizer_class,
                             l2_norm_clip: float,
                             noise_multiplier: float,
                             num_microbatches: int,
                             learning_rate: float,
                             use_sequence_output: bool = False,
                             unconnected_gradients_to_zero: bool = False):
  """Construct a simple embedding model with a classification layer."""

  # Every sample has 4 tokens (sequence length=4).
  x = tf.keras.layers.Input(shape=(4,), dtype=tf.float32, name='input')
  sequence_output, pooled_output = SimpleEmbeddingModel()(x)
  if use_sequence_output:
    embedding = sequence_output
  else:
    embedding = pooled_output
  probs = tf.keras.layers.Dense(
      units=1, activation='softmax', name='classification')(
          embedding)
  model = tf.keras.Model(inputs=x, outputs=probs, name='model')

  optimizer = optimizer_class(
      l2_norm_clip=l2_norm_clip,
      noise_multiplier=noise_multiplier,
      num_microbatches=num_microbatches,
      unconnected_gradients_to_zero=unconnected_gradients_to_zero,
      learning_rate=learning_rate)

  model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.MeanSquaredError(
          # Return per-sample loss
          reduction=tf.keras.losses.Reduction.NONE),
      metrics=['accuracy'])
  return model


class DPVectorizedOptimizerUnconnectedNodesTest(tf.test.TestCase,
                                                parameterized.TestCase):
  """Tests for vectorized optimizers when there are unconnected nodes.

  Subclassed Keras models can have layers that are defined in the graph, but
  not connected to the input or output. Or a condition expression could
  determine if the layer in question was connected or not. In such cases, the
  gradients are not present for that unconnected layer. The vectorized DP
  optimizers compute the per-microbatch losses using the Jacobian. The Jacobian
  will contain 'None' values corresponding to that layer. This causes an error
  in the gradient computation.
  This error can be mitigated by setting those unconnected gradients to 0
  instead of 'None'. This is done using the 'unconnected_gradients' flag of the
  tf.GradientTape.jacobian() method.
  This class of tests tests the possible combinations of presence/absence of
  unconnected layers and setting unconnected gradients to 'None' or 0. In these
  tests, this is done by setting 'unconnected_gradients_to_zero' to True if the
  gradients are to be set to zero, or False if they are to be set to None.
  """

  # Parameters for testing: optimizer.
  @parameterized.named_parameters(
      ('DPSGDVectorized_SeqOutput_UnconnectedGradients',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer),)
  def testSeqOutputUnconnectedGradientsAsNoneFails(self, optimizer_class):
    """Tests that DP vectorized optimizers with 'None' unconnected gradients fail.

    Sequence models that have unconnected gradients (with
    'tf.UnconnectedGradients.NONE' passed to tf.GradientTape.jacobian) will
    return a 'None' in the corresponding entry in the Jacobian. To mitigate this
    the 'unconnected_gradients_to_zero' flag is added to the differentially
    private optimizers to support setting these gradients to zero.

    These tests test the various combinations of this flag and the model.

    Args:
      optimizer_class: The DP optimizer class to test.
    """

    embedding_model = keras_embedding_model_fn(
        optimizer_class,
        l2_norm_clip=1.0,
        noise_multiplier=0.5,
        num_microbatches=1,
        learning_rate=1.0,
        use_sequence_output=True,
        unconnected_gradients_to_zero=False)

    train_data = np.random.randint(0, 10, size=(1000, 4), dtype=np.int32)
    train_labels = np.random.randint(0, 2, size=(1000, 1), dtype=np.int32)

    def train_data_input_fn():
      return tf.data.Dataset.from_tensor_slices(
          (train_data, train_labels)).batch(8)

    self.assertRaisesRegex(
        ValueError,
        'None values not supported',
        embedding_model.fit,
        x=train_data_input_fn(),
        epochs=1,
        verbose=0)

  # Parameters for testing: optimizer.
  @parameterized.named_parameters(
      ('DPSGDVectorized_PooledOutput_UnconnectedGradients',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer),)
  def testPooledOutputUnconnectedGradientsAsNonePasses(self, optimizer_class):
    """Tests that DP vectorized optimizers with 'None' unconnected gradients fail.
    """

    embedding_model = keras_embedding_model_fn(
        optimizer_class,
        l2_norm_clip=1.0,
        noise_multiplier=0.5,
        num_microbatches=1,
        learning_rate=1.0,
        use_sequence_output=False,
        unconnected_gradients_to_zero=False)

    train_data = np.random.randint(0, 10, size=(1000, 4), dtype=np.int32)
    train_labels = np.random.randint(0, 2, size=(1000, 1), dtype=np.int32)

    def train_data_input_fn():
      return tf.data.Dataset.from_tensor_slices(
          (train_data, train_labels)).batch(8)

    try:
      embedding_model.fit(x=train_data_input_fn(), epochs=1, verbose=0)
    except ValueError:
      # For a 'ValueError' exception the test should record a failure. All
      # other exceptions are errors.
      self.fail('ValueError raised by model.fit().')

  # Parameters for testing: optimizer, use sequence output flag.
  @parameterized.named_parameters(
      ('DPSGDVectorized_SeqOutput_UnconnectedGradientsAreZero',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, True),
      ('DPSGDVectorized_PooledOutput_UnconnectedGradientsAreZero',
       dp_optimizer_keras_vectorized.VectorizedDPKerasSGDOptimizer, False),
  )
  def testUnconnectedGradientsAsZeroPasses(self, optimizer_class,
                                           use_sequence_output):
    """Tests that DP vectorized optimizers with 'Zero' unconnected gradients pass.
    """

    embedding_model = keras_embedding_model_fn(
        optimizer_class,
        l2_norm_clip=1.0,
        noise_multiplier=0.5,
        num_microbatches=1,
        learning_rate=1.0,
        use_sequence_output=use_sequence_output,
        unconnected_gradients_to_zero=True)

    train_data = np.random.randint(0, 10, size=(1000, 4), dtype=np.int32)
    train_labels = np.random.randint(0, 2, size=(1000, 1), dtype=np.int32)

    def train_data_input_fn():
      return tf.data.Dataset.from_tensor_slices(
          (train_data, train_labels)).batch(8)

    try:
      embedding_model.fit(x=train_data_input_fn(), epochs=1, verbose=0)
    except ValueError:
      # For a 'ValueError' exception the test should record a failure. All
      # other exceptions are errors.
      self.fail('ValueError raised by model.fit().')


class DPTreeAggregationOptimizerComputeGradientsTest(tf.test.TestCase,
                                                     parameterized.TestCase):
  """Tests for _compute_gradients method."""

  def _loss(self, val0, val1):
    """Loss function whose derivative w.r.t val1 is val1 - val0."""
    return 0.5 * tf.reduce_sum(
        input_tensor=tf.math.squared_difference(val0, val1), axis=1)

  @parameterized.named_parameters(
      ('1_None_None', 1, None, None),
      ('2_2_1', 2, 2, 1),
      ('4_1_None', 4, 1, None),
      ('4_4_2', 4, 4, 2),
  )
  def testBaselineWithCallableLossNoNoise(self, num_microbatches,
                                          restart_period, restart_warmup):
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0])
    data0 = tf.Variable([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0]])
    data1 = tf.Variable([[8.0], [2.0], [3.0], [1.0]])
    expected_grad0 = [-2.5, -2.5]
    expected_grad1 = [-0.5]

    optimizer = dp_optimizer_keras.DPFTRLTreeAggregationOptimizer(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        var_list_or_model=[var0, var1],
        num_microbatches=num_microbatches,
        restart_period=restart_period,
        restart_warmup=restart_warmup,
        learning_rate=2.0)

    loss = lambda: self._loss(data0, var0) + self._loss(data1, var1)

    grads_and_vars = optimizer._compute_gradients(loss, [var0, var1])

    self.assertAllCloseAccordingToType(expected_grad0, grads_and_vars[0][0])
    self.assertAllCloseAccordingToType(expected_grad1, grads_and_vars[1][0])

  @parameterized.named_parameters(
      ('1_None_None', 1, None, None),
      ('2_2_1', 2, 2, 1),
      ('4_1_None', 4, 1, None),
      ('4_4_2', 4, 4, 2),
  )
  def testBaselineWithTensorLossNoNoise(self, num_microbatches, restart_period,
                                        restart_warmup):
    var0 = tf.Variable([1.0, 2.0])
    var1 = tf.Variable([3.0])
    data0 = tf.Variable([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [-1.0, 0.0]])
    data1 = tf.Variable([[8.0], [2.0], [3.0], [1.0]])
    expected_grad0 = [-2.5, -2.5]
    expected_grad1 = [-0.5]

    optimizer = dp_optimizer_keras.DPFTRLTreeAggregationOptimizer(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        var_list_or_model=[var0, var1],
        num_microbatches=num_microbatches,
        restart_period=restart_period,
        restart_warmup=restart_warmup,
        learning_rate=2.0)

    tape = tf.GradientTape()
    with tape:
      loss = self._loss(data0, var0) + self._loss(data1, var1)

    grads_and_vars = optimizer._compute_gradients(loss, [var0, var1], tape=tape)
    self.assertAllCloseAccordingToType(expected_grad0, grads_and_vars[0][0])
    self.assertAllCloseAccordingToType(expected_grad1, grads_and_vars[1][0])

  def testRaisesOnNoCallOfComputeGradients(self):
    """Tests that assertion fails when DP gradients are not computed."""
    variables = [tf.Variable([0.0])]
    optimizer = dp_optimizer_keras.DPFTRLTreeAggregationOptimizer(
        l2_norm_clip=100.0,
        noise_multiplier=0.0,
        num_microbatches=1,
        learning_rate=2.0,
        restart_period=None,
        restart_warmup=None,
        var_list_or_model=variables)

    with self.assertRaises(AssertionError):
      optimizer.apply_gradients(variables)

    # Expect no exception if _compute_gradients is called.
    data0 = tf.Variable([[0.0]])
    loss = lambda: self._loss(data0, variables[0])
    grads_and_vars = optimizer._compute_gradients(loss, variables[0])
    optimizer.apply_gradients(grads_and_vars)


if __name__ == '__main__':
  tf.test.main()
