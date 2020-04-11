import os
import platform
import sys
from multiprocessing import current_process


def print_conditional(*args, **kwargs):
    if current_process().name == 'MainProcess':
        print(*args, **kwargs)


class EnvironmentInfo:
    def __init__(self, machine_name, data_root_path, system_include_paths, wine_include_root_path, tensorflow_version,
                 gpus_to_use, gpu_id_to_use):
        self.machine_name = machine_name
        # The code project's root path. The following line assumes that the current file is located in the root of the
        # code project!
        self.code_root_path = os.path.dirname(os.path.abspath(__file__))
        # ensure absolute paths:
        self.data_root_path = os.path.abspath(data_root_path)
        self.system_include_paths = [os.path.abspath(sys_inc_path) for sys_inc_path in system_include_paths]
        self.wine_include_root_path = os.path.abspath(wine_include_root_path)
        # Can be None if the environment does not support tensorflow
        self.tensorflow_version = tensorflow_version
        self.gpu_count = gpus_to_use
        self.gpu_id_to_use = gpu_id_to_use if self.gpu_count > 0 else -1
        assert (self.gpu_count == 0) == (self.gpu_id_to_use == -1)

    def __repr__(self):
        return "\n" + "\n".join([attr + ": " + str(value) for attr, value in self.__dict__.items()])


# Define possible environments:
environments = [
    EnvironmentInfo("DesktopUbuntu",
                    "/home/simon/MasterThesisData/",
                    ["/usr/local/include", "/usr/include/x86_64-linux-gnu", "/usr/include",
                     "/usr/lib/llvm-7/lib/clang/7.0.0/include"],
                    "/usr/include/wine",
                    "1.13.1",
                    1,
                    0
                    ),
    EnvironmentInfo("laptopUbuntu",
                    "/home/simon/MasterThesisData/",
                    ["/usr/local/include", "/usr/include/x86_64-linux-gnu", "/usr/include",
                     "/usr/lib/llvm-7/lib/clang/7.0.0/include"],
                    "/usr/include/wine",
                    "1.13.1",
                    0,
                    -1
                    ),
]

# Will hold the current environment info
ENVIRONMENT = None

if sys.flags.optimize > 0:
    print("\nWARNING: ASSERTIONS ARE DISABLED!\n")

# Try Mounting Google Drive As Directory
try:
    # Load the Drive helper and mount
    # noinspection PyUnresolvedReferences
    from google.colab import drive

    print_conditional("On Google Colab. Mounting Google Drive ...")

    # This will prompt for authorization.
    drive.mount('/content/drive', force_remount=True)

    curr_machine_name = "colab"
except ModuleNotFoundError:
    # Local instance
    curr_machine_name = platform.node()

print_conditional("Determining environment config for current machine \"" + curr_machine_name + "\" ...")
matching_environments = [environment for environment in environments
                         if environment.machine_name == curr_machine_name]
if len(matching_environments) == 1:
    # Set  the environment
    ENVIRONMENT = matching_environments[0]
elif len(matching_environments) == 0:
    assert False, "No environment config for current machine with name \"" + curr_machine_name \
                  + "\". Please add an environment config for it."
else:
    assert False, ("Duplicate entry in configs.", environments)

# Remove environments to make sure that non-matching environments can not be accessed
environments = None

# Print selected environment:
print_conditional("Set environment to", ENVIRONMENT, "\n")

# Seed and Version Fixing
import random

random.seed(133742)

import numpy as np

np.random.seed(133742)

# Can the current environment import tensorflow? tensorflow/Keras import has side effects (which produce SIGABRT on non
# AVX-CPUs)
if ENVIRONMENT.tensorflow_version is not None:
    if ENVIRONMENT.gpu_count == 0:
        # CPU mode:
        #  - Define CuDNNLSTM as normal LSTM
        #  - CUDA_VISIBLE_DEVICES will be set to -1 below
        from tensorflow.python.keras.layers import LSTM

        # noinspection PyRedeclaration
        CuDNNLSTM = LSTM
        print("WARNING: CPU MODE")
    else:
        from tensorflow.python.keras.layers import CuDNNLSTM

        CuDNNLSTM = CuDNNLSTM
    if ENVIRONMENT.gpu_count < 2:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ENVIRONMENT.gpu_id_to_use)
    import tensorflow

    tf_config = tensorflow.ConfigProto()
    # Do not allocate most of the GPU memory on startup but instead allocate if needed:
    tf_config.gpu_options.allow_growth = True
    tensorflow.enable_eager_execution(config=tf_config)
    tensorflow.set_random_seed(133742)

# Check versions:
assert sys.version.startswith("3.6.7"), "Python version changed!"
if ENVIRONMENT.tensorflow_version is not None:
    assert tensorflow.__version__ == ENVIRONMENT.tensorflow_version, \
        "Tensorflow version changed to " + str(tensorflow.__version__)
    from tensorflow.python import keras as tfkeras

    assert tfkeras.__version__ == "2.2.4-tf", "Keras version changed to " + str(tfkeras.__version__)

    print_conditional("Eager Execution:", tensorflow.executing_eagerly())

    # Monkey patch tf-keras predict_generator to also return the corresponding labels:
    from tensorflow.python.keras.engine import training_generator as tfkeras_training_generator


    def custom_make_execution_function(model, mode, class_weight=None):
        # Change behaviour for predicting:
        if mode == "predict":
            # noinspection PyUnusedLocal
            def custom_predict_on_batch(x, y=None, sample_weights=None):  # pylint: disable=unused-argument
                if y is not None:
                    return model.predict_on_batch(x), y
                else:
                    return model.predict_on_batch(x)

            return custom_predict_on_batch
        else:
            # Do not change other behaviour and call the non-patched version:
            return old_make_execution_function(model, mode, class_weight)


    # noinspection PyProtectedMember
    old_make_execution_function = tfkeras_training_generator._make_execution_function
    tfkeras_training_generator._make_execution_function = custom_make_execution_function

    from tensorflow.python.keras.engine import training_utils as tfkeras_training_utils
    from tensorflow.python.keras.engine.training_utils import Aggregator


    class CustomOutputsAggregator(Aggregator):
        """Aggregator that concatenates outputs."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.label_results = []
            self.no_labels = False

        def create(self, batch_outs):
            if self.use_steps:
                # Cannot pre-allocate the returned NumPy arrays bc
                # batch sizes are unknown. Concatenate batches at the end.
                for batch_out in batch_outs:
                    self.results.append([])
                    if isinstance(batch_out, tuple):
                        assert not self.no_labels
                        self.label_results.append([])
                    else:
                        self.no_labels = True
            else:
                assert False
                # Pre-allocate NumPy arrays.
                # noinspection PyUnreachableCode
                for batch_out in batch_outs:
                    shape = (self.num_samples_or_steps,) + batch_out.shape[1:]
                    self.results.append(np.zeros(shape, dtype=batch_out.dtype))

        def aggregate(self, batch_outs, batch_start=None, batch_end=None):
            if self.use_steps:
                for i, batch_out in enumerate(batch_outs):
                    if not self.no_labels:
                        self.results[i].append(batch_out[0])
                        self.label_results[i].append(batch_out[1])
                    else:
                        self.results[i].append(batch_out)
            else:
                assert False
                # noinspection PyUnreachableCode
                for i, batch_out in enumerate(batch_outs):
                    self.results[i][batch_start:batch_end] = batch_out

        def finalize(self):
            if self.use_steps:
                self.results = [np.concatenate(result, axis=0) for result in self.results]
                if not self.no_labels:
                    self.results = [self.results, [np.concatenate(label_result, axis=0)
                                                   for label_result in self.label_results]]
                    if len(self.results[0]) == 1:
                        self.results[0] = self.results[0][0]
                        self.results[1] = self.results[1][0]


    OldOutputsAggregator = tfkeras_training_utils.OutputsAggregator
    tfkeras_training_utils.OutputsAggregator = CustomOutputsAggregator

    print_conditional("Monkey-patched tensorflow keras predict_generator.")

    # Print new line:
    print_conditional()
