# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import scipy.stats as osp_stats
from jax import lax
from jax._src.numpy.util import _wraps
from jax._src.numpy.lax_numpy import _promote_args_inexact
from jax._src.lax.lax import _const as _lax_const
from jax._src.typing import Array, ArrayLike


@_wraps(osp_stats.gennorm.logpdf, update_doc=False)
def logpdf(x: ArrayLike, p: ArrayLike) -> Array:
  x, p = _promote_args_inexact("gennorm.logpdf", x, p)
  return lax.log(.5 * p) - lax.lgamma(1/p) - lax.abs(x)**p

@_wraps(osp_stats.gennorm.cdf, update_doc=False)
def cdf(x: ArrayLike, p: ArrayLike) -> Array:
  x, p = _promote_args_inexact("gennorm.cdf", x, p)
  half = _lax_const(x, 0.5)
  one = _lax_const(x, 1)
  return lax.mul(half, (lax.mul(lax.sum(one, lax.sign(x)), lax.igamma(lax.div(one,p), lax.pow(lax.abs(x),p)))))

@_wraps(osp_stats.gennorm.pdf, update_doc=False)
def pdf(x: ArrayLike, p: ArrayLike) -> Array:
  return lax.exp(logpdf(x, p))

# Need inverse of incomplete gamma function in lax
# @_wraps(osp_stats.gennorm.ppf, update_doc=False)
# def ppf(q: ArrayLike, p: ArrayLike) -> Array:
#     q, p = _promote_args_inexact("gennorm.ppf", q, p)
#     c = lax.sign(lax.sub(q, 0.5))
#     one = _lax_const(q, 1)
#     two = _lax_const(q, 2)
#     return lax.sign(q) * lax.pow(lax.igammacinv(lax.div(one, p), lax.sub(lax.add(one, c), lax.mul(two,c,q))), lax.div(one, p))
