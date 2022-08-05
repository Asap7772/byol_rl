# Copyright 2020 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation loops for an experiment."""

import time
from typing import Any, Mapping, Text, Type, Union

from absl import app
from absl import flags
from absl import logging
import jax
import numpy as np

from byol import byol_experiment
from byol import eval_experiment
from byol.configs import byol as byol_config
from byol.configs import eval as eval_config
import wandb

import tqdm

flags.DEFINE_string('experiment_mode',
                    'pretrain', 'The experiment, pretrain or linear-eval')
flags.DEFINE_string('worker_mode', 'train', 'The mode, train or eval')
flags.DEFINE_string('worker_tpu_driver', '', 'The tpu driver to use')
flags.DEFINE_integer('pretrain_epochs', 1000, 'Number of pre-training epochs')
flags.DEFINE_integer('batch_size', 4096, 'Total batch size')
flags.DEFINE_string('checkpoint_root', '/tmp/byol',
                    'The directory to save checkpoints to.')
flags.DEFINE_integer('log_tensors_interval', 100, 'Log tensors every n steps.')
flags.DEFINE_string('wandb_project', 'byol', 'The wandb project to use.')
flags.DEFINE_string('run_name', 'test', 'Name of the run.')
flags.DEFINE_integer('use_imagenette', 1, 'To use Imagenette dataset')
flags.DEFINE_integer('rl_update', 0, 'Use RL Update')
flags.DEFINE_integer('num_samples', 20, 'Number of Samples')
flags.DEFINE_integer('update_type', 0, 'Use RL Update')
FLAGS = flags.FLAGS


Experiment = Union[
    Type[byol_experiment.ByolExperiment],
    Type[eval_experiment.EvalExperiment]]


def train_loop(experiment_class: Experiment, config: Mapping[Text, Any]):
  """The main training loop.

  This loop periodically saves a checkpoint to be evaluated in the eval_loop.

  Args:
    experiment_class: the constructor for the experiment (either byol_experiment
    or eval_experiment).
    config: the experiment config.
  """
  wandb.init(
    project=FLAGS.wandb_project if FLAGS.run_name != 'test' else 'test', 
    config=config, 
    reinit=True, 
    settings=wandb.Settings(start_method="thread")
  )
  wandb.run.name = FLAGS.run_name

  experiment = experiment_class(**config)
  rng = jax.random.PRNGKey(0)
  step = 0

  host_id = jax.host_id()
  if config['checkpointing_config']['use_checkpointing']:
    checkpoint_data = experiment.load_checkpoint()
    if checkpoint_data is None:
      step = 0
    else:
      step, rng = checkpoint_data

  local_device_count = jax.local_device_count()

  print('Starting training loop.', step, config['max_steps'])
  
  for step in tqdm.tqdm(range(step, config['max_steps']), desc='Step'):
    step_rng, rng = tuple(jax.random.split(rng))
    # Broadcast the random seeds across the devices
    step_rng_device = jax.random.split(step_rng, num=jax.device_count())
    step_rng_device = step_rng_device[
        host_id * local_device_count:(host_id + 1) * local_device_count]
    step_device = np.broadcast_to(step, [local_device_count])

    # Perform a training step and get scalars to log.
    scalars = experiment.step(global_step=step_device, rng=step_rng_device)

    # Checkpointing and logging.
    if config['checkpointing_config']['use_checkpointing']:
      experiment.save_checkpoint(step, rng)
      current_time = time.time()
      
      if step % FLAGS.log_tensors_interval == 0:
        wandb.log({k:v.item() for k,v in scalars.items()}, step=step)
        logging.info('Step %d: %s', step, scalars)
        last_logging = current_time

  try:
    logging.info('Saving final checkpoint')
    logging.info('Step %d: %s', step, scalars)
    experiment.save_checkpoint(step, rng)
    wandb.log({k:v.item() for k,v in scalars.items()}, step=step)
  except Exception as e:
    pass

def eval_loop(experiment_class: Experiment, config: Mapping[Text, Any]):
  """The main evaluation loop.

  This loop periodically loads a checkpoint and evaluates its performance on the
  test set, by calling experiment.evaluate.

  Args:
    experiment_class: the constructor for the experiment (either byol_experiment
    or eval_experiment).
    config: the experiment config.
  """
  experiment = experiment_class(**config)
  last_evaluated_step = -1
  while True:
    checkpoint_data = experiment.load_checkpoint()
    if checkpoint_data is None:
      logging.info('No checkpoint found. Waiting for 10s.')
      time.sleep(10)
      continue
    step, _ = checkpoint_data
    if step <= last_evaluated_step:
      logging.info('Checkpoint at step %d already evaluated, waiting.', step)
      time.sleep(10)
      continue
    host_id = jax.host_id()
    local_device_count = jax.local_device_count()
    step_device = np.broadcast_to(step, [local_device_count])
    scalars = experiment.evaluate(global_step=step_device)
    if host_id == 0:  # Only perform logging in one host.
      logging.info('Evaluation at step %d: %s', step, scalars)
    last_evaluated_step = step
    if last_evaluated_step >= config['max_steps']:
      return


def main(_):
  if FLAGS.worker_tpu_driver:
    jax.config.update('jax_xla_backend', 'tpu_driver')
    jax.config.update('jax_backend_target', FLAGS.worker_tpu_driver)
    logging.info('Backend: %s %r', FLAGS.worker_tpu_driver, jax.devices())

  if FLAGS.experiment_mode == 'pretrain':
    experiment_class = byol_experiment.ByolExperiment
    config = byol_config.get_config(FLAGS.pretrain_epochs, FLAGS.batch_size, 
                                    use_imagenette=FLAGS.use_imagenette, 
                                    rl_update=FLAGS.rl_update,
                                    num_samples=FLAGS.num_samples,
                                    update_type=FLAGS.update_type)
  elif FLAGS.experiment_mode == 'linear-eval':
    experiment_class = eval_experiment.EvalExperiment
    config = eval_config.get_config(f'{FLAGS.checkpoint_root}/pretrain.pkl',
                                    FLAGS.batch_size)
  else:
    raise ValueError(f'Unknown experiment mode: {FLAGS.experiment_mode}')
  config['checkpointing_config']['checkpoint_dir'] = FLAGS.checkpoint_root  # pytype: disable=unsupported-operands  # dict-kwargs

  if FLAGS.worker_mode == 'train':
    train_loop(experiment_class, config)
  elif FLAGS.worker_mode == 'eval':
    eval_loop(experiment_class, config)


if __name__ == '__main__':
  app.run(main)
