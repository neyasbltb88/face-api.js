/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import { reshape, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Rank } from '@tensorflow/tfjs-core';


/**
 * Converts a `tf.Tensor` to a `tf.Tensor1D`.
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
export function as1D<T extends Tensor>(thiz: T): Tensor1D {
  thiz.throwIfDisposed();
  return reshape(thiz, [thiz.size]) as Tensor1D;
};

export function as2D<T extends Tensor>(thiz: T,
  rows: number, columns: number): Tensor2D {
  thiz.throwIfDisposed();
  return reshape(thiz, [rows, columns]) as Tensor2D;
};

export function as3D<T extends Tensor>(thiz: T,
  rows: number, columns: number, depth: number): Tensor3D {
  thiz.throwIfDisposed();
  return reshape(thiz, [rows, columns, depth]) as Tensor3D;
};

export function as4D<T extends Tensor>(thiz: T,
  rows: number, columns: number, depth: number, depth2: number): Tensor4D {
  thiz.throwIfDisposed();
  return reshape(thiz, [rows, columns, depth, depth2]) as Tensor4D;
}
