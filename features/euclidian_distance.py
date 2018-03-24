from __future__ import print_function
import tensorflow as tf

class MeanEuclidianDistance(object):
   '''
   '''

   def __init__(self, session=None):

      self.set_session()
      self._v1_placeholder = tf.placeholder(tf.float64, [None, 1, None])
      self._m1_placeholder = tf.placeholder(tf.float64, [None, None])
      self._build_mean_euclidian_distance()
      self._build_mean_angle()


   def calculate(self, v1, m1):
      '''

      :param v1: A tensorflow (None, 1, None) containing new points to calculate distance
      :param m1: A tensorflow (None, None) containing all points representing a standard behaviour
      :return: A Tensor (None, 1) containing mean euclidian distance for each
      '''
      med, man = self._session.run([self._mean_euclidian_distance, self._mean_angle], feed_dict = {self._v1_placeholder: v1, self._m1_placeholder: m1})
      return med, man


   def _build_mean_euclidian_distance(self):
      '''
      '''
      self._mean_euclidian_distance = \
          tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(
             self._v1_placeholder - self._m1_placeholder), axis=2)), axis=1)


   def _build_mean_angle(self):
      '''
      '''
      v1_unit_vector = self._v1_placeholder / tf.norm(self._v1_placeholder, axis=2, keep_dims=True)
      m1_unit_vector = self._m1_placeholder / tf.norm(self._m1_placeholder, axis=1, keep_dims=True)

      self._mean_angle = tf.acos(tf.clip_by_value(tf.reduce_sum(tf.multiply(v1_unit_vector, m1_unit_vector), axis=2), -1., 1.))
      self._mean_angle = tf.reduce_mean(self._mean_angle, axis=1) * 57.2958 / 300.0


   def set_session(self, session=None):
      if session is None:
         #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333, allocator_type = 'BFC')
         #config=tf.ConfigProto(gpu_options=gpu_options)
         self._session = tf.Session()
      else:
         self._session = session