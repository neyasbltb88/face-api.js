import * as tf from '@tensorflow/tfjs-core';

export function normalize(x: tf.Tensor4D, meanRgb: number[]): tf.Tensor4D {
  return tf.tidy(() => {
    const [r, g, b] = meanRgb
    const avg_r = tf.fill([...x.shape.slice(0, 3), 1], r, x.dtype)
    const avg_g = tf.fill([...x.shape.slice(0, 3), 1], g, x.dtype)
    const avg_b = tf.fill([...x.shape.slice(0, 3), 1], b, x.dtype)
    const avg_rgb = tf.concat([avg_r, avg_g, avg_b], 3)

    return tf.sub(x, avg_rgb)
  })
}