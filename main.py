from run_lib import train, eval
import datetime
import os
from absl import app, flags
from ml_collections.config_flags import config_flags
import warnings

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory to store files.")
flags.DEFINE_enum("mode", None, ['train', 'eval', 'tune'], "Running mode: train, eval or finetune.")
flags.mark_flags_as_required(['mode', 'config'])


def main(argv):
    warnings.filterwarnings('ignore')

    config = FLAGS.config

    if FLAGS.workdir is not None:
        work_dir = os.path.join('workspace', FLAGS.workdir)
    else:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        work_dir = os.path.join('workspace', f'run_{time_str}')

    if FLAGS.mode == 'train':
        train(config=config, workdir=work_dir)
    elif FLAGS.mode == 'eval':
        eval(config=config, workdir=work_dir)


if __name__ == '__main__':
    app.run(main)
