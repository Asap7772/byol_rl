import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

class RandomRewards(hk.Module):
  
  def __init__(self, output_size, name=None):
    super().__init__(name=name)
    self.output_size = output_size
    self.normalized = False

  def __call__(self, x):
    j, k = x.shape[-1], self.output_size
    w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))
    w = hk.get_state("w", shape=[j, k], dtype=x.dtype, init=w_init)
    
    # run gram-schmidt on w
    if not self.normalized:
        w = jnp.linalg.qr(w)[0]
        hk.set_state("w", w)
        self.normalized = True
    
    out = jnp.dot(x, w)
    return jax.lax.stop_gradient(out)