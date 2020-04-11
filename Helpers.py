# from tensorflow.python.keras.utils import Sequence, OrderedEnqueuer, GeneratorEnqueuer, Progbar
# from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite
# from tensorflow.python.keras.utils.generic_utils import to_list
import atexit
import glob
import gzip
import hashlib
import importlib.util
import json
import linecache
import math
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import tracemalloc
import typing
import urllib.request
import warnings
import zipfile
from contextlib import contextmanager
from io import TextIOWrapper
from os import listdir
from pathlib import Path
from typing import List, Dict

import dill as pickle
import psutil
import six

from Environment import *


class Tee(object):
    """
    Allows to enable the writing of each print to a file
    """
    sinks: List[TextIOWrapper] = None

    def __init__(self, sinks):
        self.sinks = sinks

    def write(self, obj):
        for sink in self.sinks:
            sink.write(obj)
        self.flush()

    def flush(self):
        for sink in self.sinks:
            sink.flush()

    def close(self):
        for sink in self.sinks:
            if not sink.closed and not sink.name.startswith("<std"):
                sink.flush()
                sink.close()

    def __repr__(self):
        return repr(self.sinks)

    @classmethod
    def enable_single(cls, std_name: str, file_obj):
        sys_io = getattr(sys, std_name)
        assert type(sys_io) is not cls, type(sys_io)
        tee = cls([sys_io, file_obj])
        setattr(sys, std_name, tee)
        print("Enabled Tee for", tee, "...")

    @classmethod
    def enable(cls, file_path, enable_syserr=True):
        # Disable previous Tee if any:
        cls.disable()
        # Make sure Tee is disabled on programm end:
        atexit.register(cls.disable)
        # Enable:
        file_obj = open(file_path, "a")
        cls.enable_single("stdout", file_obj)
        if enable_syserr:
            cls.enable_single("stderr", file_obj)

    @classmethod
    def disable_single(cls, std_name: str):
        tee = getattr(sys, std_name)
        if type(tee) is cls:
            print("Disabling Tee for", tee, "...")
            setattr(sys, std_name, [sink for sink in tee.sinks if sink.name == "<" + std_name + ">"][0])
            tee.close()
        # Tee does not need to be disabled at programm end anymore
        atexit.unregister(cls.disable)

    @classmethod
    def disable(cls):
        cls.disable_single("stdout")
        cls.disable_single("stderr")


def print_histogram(hist_data: Dict[int, int], max_rows: int = 30, max_columns: int = 20, topic: str = None):
    ordered_keys = sorted(hist_data.keys())
    min_key = ordered_keys[0]
    max_key = ordered_keys[-1]
    key_interval_length = max_key - min_key
    bin_size = max([math.ceil(key_interval_length / max_rows), 1])
    bin_count = int(key_interval_length / bin_size) + 1
    # Fill bins:
    bins = {i: 0 for i in range(bin_count)}
    for key in ordered_keys:
        bins[int((key - min_key) / bin_size)] += hist_data[key]
    # Display bins:
    max_bin_value = max(bins.values())
    # Determine the maximum bin end digit count for zero filling:
    max_bin_end_digits = int(math.log10(bin_size * bin_count)) + 1
    print("Histogram" + ((" of " + topic) if topic is not None else ""),
          "(bin size: " + str(bin_size) + ", bin count: " + str(bin_count) + ", bin value sum: " + str(
              sum(bins.values())) + ")")
    for bin_nr, bin_val in bins.items():
        bin_start = min_key + (bin_nr * bin_size)
        bin_end = bin_start + bin_size - 1  # exclusive upper bound
        print("[" + str(bin_start).zfill(max_bin_end_digits) +
              (("," + str(bin_end).zfill(max_bin_end_digits)) if bin_start != bin_end else "") + "]" +
              ("=" * math.ceil(bin_val / max_bin_value * max_columns)) + "> " + str(bin_val))
    print()


def argmax(iterable, also_return_max_value=False):
    max_index, max_value = max(enumerate(iterable), key=lambda index_val_pair: index_val_pair[1])
    return (max_index, max_value) if also_return_max_value else max_index


def as_safe_filename(string):
    """
    Generates a safe filename from the given string
    :param string:
    :return:
    """
    # do not keep "{", "}", because they interfere with str.format
    # do not keep "[", "]", because tensorflow checkpoints fail on these
    #   (https://github.com/tensorflow/tensorflow/issues/6082#issuecomment-265055615 and
    #   https://stackoverflow.com/questions/49669695/errortensorflowcouldnt-match-files-for-checkpoint/49749781)
    # do not keep ",", ":" because Tensorboard's --logdir argument parses these as separator of multiple (named) logdirs
    #   (https://github.com/tensorflow/tensorflow/issues/6082#issuecomment-265055615 and
    #   https://stackoverflow.com/questions/49669695/errortensorflowcouldnt-match-files-for-checkpoint/49749781)
    keep_characters = (".", "_", "+", "(", ")", "#", "-", "!", "$")
    return "".join(c if (c.isalnum() or c in keep_characters) else "_" for c in string).rstrip()


class ExtendedEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy types and simple custom types
    """

    def default(self, obj):
        """
        Is only called when an object can not be serialized otherwise
        :param obj:
        :return:
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif hasattr(obj, "__dict__"):
            # For simple classes like FeatVecAnnInputInfo
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)


def to_json(obj: object) -> str:
    return json.dumps(obj, cls=ExtendedEncoder)


def from_json(json_string: str, class_obj: typing.Type = None) -> typing.Union[Dict, object]:
    # object_hook is called for each json dict object.
    return json.loads(json_string,
                      object_hook=(lambda json_dict: class_obj(json_dict)) if class_obj is not None else None)


def from_legacy_dc_info_json_string(legacy_json_string: str):
    """
    Get json object from legacy info json string
    (e.g. "['Juliet', 'C++', ['ClangTooling'], ['CWE121_Stack_Based_Buffer_Overflow', 'CWE00_NO_FLAW_FOUND'], {'test_fraction': 0.15, 'validation_fraction': 0.15}, '/home/schroeder/MasterThesisData/Juliet/Juliet_Test_Suite_v1.3_for_C_Cpp', None, None, 10, False, ['c', 'cpp']]_20190516-114719")
    :param legacy_json_string: Either a string containing the legacy json string (i.e. first char is quote)
    or a string containing the legacy json string content (i.e. first char is opening square bracket)
    :return:
    """
    if legacy_json_string[0] == "\"":
        # parse legacy json string:
        legacy_json_string = from_json(legacy_json_string)
        assert isinstance(legacy_json_string, str), legacy_json_string
    # Remove timestamp at end:
    legacy_json_string = re.sub(r"_\d{8}-\d{6}$", "", legacy_json_string)
    # Replace single quotes with double quotes, True with true, False with false, None with null:
    assert "\"" not in legacy_json_string, legacy_json_string
    legacy_json_string = legacy_json_string \
        .replace("'", "\"") \
        .replace("None", "null") \
        .replace("False", "false") \
        .replace("True", "true")
    # String should be valid json now:
    return from_json(legacy_json_string)


def serialize_data(data, file_path, verbose=True, compress=True):
    if verbose:
        print("Serializing data " + ("compressed " if compress else "") + "to", file_path, "...")
    open_func = open if not compress else gzip.open
    with open_func(file_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print("Serialized data" + (" compressed" if compress else "") + ".")  # to " + file_path)


def deserialize_data(file_path, verbose=True, is_compressed=True):
    if verbose:
        print("Deserializing " + ("compressed " if is_compressed else "") + "data from " + file_path + " ...")
    open_func = open if not is_compressed else gzip.open
    try:
        with open_func(file_path, "rb") as handle:
            data = pickle.load(handle)
            if verbose:
                print("Deserialized " + ("compressed " if is_compressed else "") + "data.")  # from " + file_path)
            return data
    except OSError as os_err:
        if is_compressed and isinstance(os_err.args[0], str) and os_err.args[0].startswith("Not a gzipped file"):
            # Try without compression (print file if not alrady printed in previous print):
            print("File " + ((file_path + " ") if not verbose else "") + "is not compressed."
                  + " Trying to deserialize as uncompressed file ...")
            return deserialize_data(file_path, verbose, False)
        else:
            raise


if ENVIRONMENT.tensorflow_version is not None:
    from tensorflow.python.training.checkpointable.data_structures import _DictWrapper


    def save_model_to_path(model, path, model_name, add_timestamp=True, verbose=True):
        assert "[" not in path + model_name and "]" not in path + model_name, \
            "[ and ] are not supported in path or model name due to a tensorflow bug"
        # Add timestamp if requested
        if add_timestamp:
            model_name = model_name + "_" + get_timestamp()

        if tensorflow.executing_eagerly():
            dir_path = os.path.join(path, model_name)
            create_dir_if_necessary(dir_path, verbose=False)
            checkpoint = tensorflow.train.Checkpoint(model=model, optimizer=model.optimizer)
            actual_saved_path = checkpoint.save(os.path.join(dir_path, "model"))

            # Recursevily replace DictWrapper objects with normal dict objects. Otherwise it can not be unpickled later
            def replace_dictwrapper(obj):
                if isinstance(obj, _DictWrapper):
                    obj = dict(obj)
                    for key, value in obj.items():
                        obj[key] = replace_dictwrapper(value)
                return obj

            kwargs_dict = replace_dictwrapper(model.create_model_func_kwargs)
            # Serialize args for create model function:
            serialize_data(kwargs_dict, os.path.join(dir_path, "kwargs.pkl"), verbose=False)
            # Save create model function and surrounding module
            write_file_content(os.path.join(dir_path, "create_model_module.py"), model.create_model_func_source)

            # Test whether the model can be loaded again:
            # model = load_model_from_path(dir_path)
            return actual_saved_path
        else:
            # serialize model structure to JSON
            model_json = model.to_json()
            write_file_content(os.path.join(path, model_name + ".json"), model_json)

            # serialize weights to HDF5
            model.save_weights(os.path.join(path, model_name + ".h5"))

            # Save entire model (structure, weights, optimizer states, ...) to a HDF5 file
            model.save(os.path.join(path, model_name + "_complete.hdf5"))
        if verbose:
            print("Saved model to " + os.path.join(path, model_name))


    def load_model_from_path(dir_path):
        initial_dir_path = dir_path
        if tensorflow.executing_eagerly():
            # Copy the directory to local temp directory to speed up loading in case of remote paths. Otherwise building
            # the model takes very long (probably tensorflow does heavy IO stuff then)
            # Do not use "with tempfile.TemporaryDirectory() as ..." because the model files need to be available
            # until model.fit(_generator) call (probably because the weights are loaded from file then). Instead,
            # register cleanup function which is called at the end of the current script.
            temp_model_dir_path = tempfile.mkdtemp()

            def temp_model_dir_path_cleanup():
                # print("Removing temporary model directory", temp_model_dir_path, "...")
                remove_dir(temp_model_dir_path, verbose=True)

            atexit.register(temp_model_dir_path_cleanup)

            # Just copy the model related files (and not Visualizations and __pycache__ directories):
            model_file_wildcard_names_to_copy = ["kwargs.pkl", "create_model_module.py", "model*"]
            for model_file_wildcard_name_to_copy in model_file_wildcard_names_to_copy:
                model_file_wildcard_path_to_copy = os.path.join(dir_path, model_file_wildcard_name_to_copy)
                # Use glob to get all file paths that match the wildcarded file path:
                file_paths_to_copy = glob.glob(model_file_wildcard_path_to_copy)
                assert len(file_paths_to_copy) > 0, model_file_wildcard_path_to_copy + " does not exist."
                for file_path_to_copy in file_paths_to_copy:
                    # print(file_path_to_copy)
                    copy_file(file_path_to_copy, temp_model_dir_path)
            # Use the temp path in the following:
            # TODO: Results in Error 2019-05-23 22:33:52.599129: W tensorflow/core/framework/op_kernel.cc:1401] OP_REQUIRES failed at save_restore_tensor.cc:175 : Invalid argument: Unsuccessful TensorSliceReader constructor: Failed to get matching files on /tmp/tmpaunnwsnp/model_dir/model-1: Not found: /tmp/tmpaunnwsnp/model_dir; No such file or directory
            # Unsuccessful TensorSliceReader constructor: Failed to get matching files on /tmp/tmpaunnwsnp/model_dir/model-1: Not found: /tmp/tmpaunnwsnp/model_dir; No such file or directory [Op:RestoreV2] name: VARIABLE_VALUE_checkpoint_read
            dir_path = temp_model_dir_path

            # Load arguments for create model function:
            create_model_func_kwargs = deserialize_data(os.path.join(dir_path, "kwargs.pkl"), verbose=False)
            # Load module with which the model was created:
            create_model_module_spec = importlib.util.spec_from_file_location("create_model",
                                                                              os.path.join(dir_path,
                                                                                           "create_model_module.py"))
            create_model_module = importlib.util.module_from_spec(create_model_module_spec)
            create_model_module_spec.loader.exec_module(create_model_module)

            # Create the model using its initial create function and arguments:
            model = create_model_module.create(**create_model_func_kwargs)

            # Workaround
            if getattr(model, "create_model_func_source", None) is None:
                model.create_model_func_source = read_file_content(os.path.join(dir_path, "create_model_module.py"))

            # Restore the model's weights:
            checkpoint = tensorflow.train.Checkpoint(model=model, optimizer=model.optimizer)
            # Find latest checkpoint (do not use tensorflow.train.latest_checkpoint because it just reads the
            # "checkpoint" file which contains absolute paths. Absolute paths fail when the model has been copied or
            # when the model is loaded through a (sftp) mount.)
            model_index = 1  # start with 1 (seems not to be zero-based)
            model_file_stub_path = None
            while True:
                # "model" must be the same as specified in save_model_to_path func
                # Only the stub of the actual ".index" file path must be passed to restore. E.g. if there is the model
                # ".index" file ".../model-1.index" then ".../model-1" must be used as argument
                model_file_path_stub_to_check = os.path.join(dir_path, "model" + "-" + str(model_index))
                # Check for ".index" file existence:
                if os.path.exists(model_file_path_stub_to_check + ".index"):
                    # Remember file path stub ...:
                    model_file_stub_path = model_file_path_stub_to_check
                    # ...and continue search for latest model:
                    model_index += 1
                else:
                    break
            assert model_file_stub_path is not None, model_file_path_stub_to_check
            # print(model_file_stub_path)
            restore_status = checkpoint.restore(save_path=model_file_stub_path)

            # Weights are actually loaded/restored deferred, i.e. on the next fit or predict call. Therefore, when
            # called now, assert_consumed will assert. After fit or predict it should not.
            # restore_status.assert_consumed()
            # restore.run_restore_ops() may be necessary in graph mode

            # The following checks whether the list of weights in the checkpoint matches the list of weights in the
            # model and should therefore run without an error.
            # A reason for an error here, could be the loading in gpu mode of a model with LSTMs, which was created in
            # CPU mode, or vice versa.
            restore_status.assert_existing_objects_matched()

            # Add references to restore status and model directory cleanup to model:
            model.restore_status = restore_status
            model.temp_model_dir_path_cleanup = temp_model_dir_path_cleanup
        else:
            model = tfkeras.models.load_model(dir_path)
            print(model.summary())
        print("Loaded model from", initial_dir_path, "(" + dir_path + ")")
        return model


    def get_model_with_other_layer_activation(model, layer_index, new_activation):
        assert not tensorflow.executing_eagerly()
        print("Replacing activation " + str(model.layers[layer_index].activation) + " of layer " + str(
            model.layers[layer_index]) + " with new activation " + str(new_activation) + " ...")
        if model.layers[layer_index].activation != new_activation:
            model.layers[layer_index].activation = new_activation
            with tempfile.NamedTemporaryFile("w", suffix=".h5") as temp_model_file:
                model.save(temp_model_file.name)
                new_model = tfkeras.models.load_model(temp_model_file.name)  # TODO: use load_model from Helpers?
                print("Done. New activation: " + str(new_model.layers[layer_index].activation))
                return new_model
        else:
            print("Nothing to be done.")
            return model


@contextmanager
def expect_warnings(expected_warnings: List[typing.Type], verbose: bool = True):
    """
    Catches the warnings of the given types and prints their class names when leaving the context
    :param expected_warnings:
    :param verbose:
    :return:
    """
    occurred_warnings = []
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            yield None
        occurred_warnings = caught_warnings
    finally:
        # Do the handling of warnings outside the catch_warnings block to allow raising warnings (otherwise
        # catch_warnings catches the warnings again)
        if len(occurred_warnings) > 0:
            warn_category_strings = []
            for occured_warning in occurred_warnings:
                if occured_warning.category not in expected_warnings:
                    # "Raise" warning:
                    warnings.warn_explicit(occured_warning.message, occured_warning.category,
                                           occured_warning.filename, occured_warning.lineno,
                                           source=occured_warning.source)

                else:
                    # Remember WarningName
                    warn_category_strings.append(str(occured_warning.category.__name__))
            if verbose:
                # Remove duplicates:
                warn_category_strings_distinct = list(set(warn_category_strings))
                print("(" + ", ".join(warn_category_strings_distinct) + "), ", end="")


def get_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def get_md5_string(string_to_hash: str) -> str:
    return str(hashlib.md5(string_to_hash.encode('utf-8')).hexdigest())


def get_latex_escaped(text):
    """
    Based on https://stackoverflow.com/questions/16259923/how-can-i-escape-latex-special-characters-inside-django-templates#answer-25875504
    :param text:
    :return:
    """
    escape_mapping = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
        '\r\n': r'\\',  # try CRLF first
        '\r': r'\\',
        '\n': r'\\',
    }
    # Create regex with ORs:
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(escape_mapping.keys(),
                                                                      key=lambda item: -len(item))))
    # Replace based on index:
    return regex.sub(lambda match: escape_mapping[match.group()], text)


def run_lualatex(latex_file_path):
    """
    Runs lualatex for the given file with enabled shell escape
    :param latex_file_path:
    :return:
    """
    lualatex_command = ["lualatex", "-interaction=nonstopmode", "--shell-escape",
                        os.path.basename(latex_file_path)]
    return subprocess.run(lualatex_command,
                          stdout=subprocess.DEVNULL,
                          cwd=os.path.dirname(latex_file_path))


def compile_latex(latex_file_path, runs=2):
    """
    Compiles the given latex file using lualatex.
    :param latex_file_path:
    :param runs:
    :return:
    """
    assert latex_file_path[-4:] == ".tex", latex_file_path
    print("Compiling", latex_file_path, "...")
    try:
        # Copy file to temporary dir to avoid long compilation time for remote files
        with tempfile.TemporaryDirectory() as temp_dir_path:
            # print(temp_dir_path)
            copied_latex_file_path = copy_file(latex_file_path, temp_dir_path)
            print(" - Copied latex file to temp directory.")
            for run in range(runs):
                print(" - Run", run + 1, "...")
                result_object = run_lualatex(copied_latex_file_path)
                if result_object.returncode != 0:
                    print(" - Failed to generate pdf with exit code", result_object)
                    return False
            result_file = copy_file(copied_latex_file_path[:-4] + ".pdf", os.path.dirname(latex_file_path))
    except Exception as ex:
        print(" - Failed to generate pdf with error", ex)
        return False
    print(" - Generated pdf file", result_file)
    return True


def read_file_content(file_path):
    with open(file_path, "r") as file:
        return file.read()


def write_file_content(file_path, content, check_written_content=False):
    with open(file_path, "w") as file:
        file.write(content)
    if check_written_content and read_file_content(file_path) != content:
        raise Exception("Faulty write detected!")


def copy_file(source_file_path, destination_file_or_dir_path) -> str:
    return shutil.copy(source_file_path, destination_file_or_dir_path)


def copy_dir(source_dir_path, destination_dir_path):
    shutil.copytree(source_dir_path, destination_dir_path)


def create_dir_if_necessary(dir_path: str, verbose=True) -> bool:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        if verbose:
            print("Created directory " + dir_path)
        return True
    else:
        return False


def remove_dir(dir_path, verbose=True):
    if verbose:
        print("Removing " + dir_path + " ...")
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        if verbose:
            print("Removed directory " + dir_path)


def remove_file(file_path, verbose=True):
    if verbose:
        print("Removing " + file_path + " ...")
    file_path_obj = Path(file_path)
    if file_path_obj.exists():
        file_path_obj.unlink()
        if verbose:
            print("Removed file " + file_path)


def recreate_dir(dir_path):
    remove_dir(dir_path)
    create_dir_if_necessary(dir_path)


def print_python_ram_usage():
    process = psutil.Process(os.getpid())
    python_used_bytes = int(process.memory_info().rss)  # in bytes
    print("Python used RAM: " + str(int(python_used_bytes / 1024 / 1024)) + "MiB")


def get_non_rec_dir_content_names(dir_path):
    return [(os.path.join(dir_path, f_or_d), f_or_d) for f_or_d in listdir(dir_path)]


def download_file(from_url, to_file_name, to_dir, force_download=False):
    to_file_path = os.path.join(to_dir, to_file_name)
    if not os.path.isfile(to_file_path) or force_download:
        print("Downloading " + from_url + " to " + to_file_path + " ...")
        urllib.request.urlretrieve(from_url, to_file_path)
    return to_file_path


def unzip_file(zip_file_path, destination_dir):
    with zipfile.ZipFile(zip_file_path, "r") as zip_file:
        zip_file.extractall(destination_dir)


def backup_dir_to_archive(dir_path_to_archive: str,
                          dir_path_to_put_archive_to: str = None,
                          exclude_files_dirs_startswith: typing.Tuple = None,
                          verbose=True) -> None:
    """
    Backups the given directory recursively into a tar.gz archive.
    :param dir_path_to_archive: Path to directory which should be backuped
    :param dir_path_to_put_archive_to: Path to directory, in which the backup archive should be placed. If None,
    dir_path_to_archive's parent directory is used.
    :param exclude_files_dirs_startswith: Tuple of strings. Only archive files and directories whose name does not start
    with one of the string in exclude_files_dirs_startswith.
    :param verbose: Whether to output all archived files and directories
    :return:
    """
    if dir_path_to_put_archive_to is None:
        # Use the parent directories as directory to save the archive to:
        dir_path_to_put_archive_to = os.path.join(dir_path_to_archive, os.pardir)
    dir_path_to_put_archive_to = os.path.abspath(dir_path_to_put_archive_to)

    if exclude_files_dirs_startswith is None:
        exclude_files_dirs_startswith = ()
    else:
        # Make sure exclude_files_dirs_startswith is a tuple (as requested by str.startswith)
        exclude_files_dirs_startswith = tuple(exclude_files_dirs_startswith)

    # Generate the archive name from the directory to be archived and a timestamp:
    dir_name_to_archive = os.path.basename(dir_path_to_archive)
    archive_name = dir_name_to_archive + "_backup_" + get_timestamp() + ".tar.gz"
    archive_file_path = os.path.join(dir_path_to_put_archive_to, archive_name)

    # Callback for filtering and progress printing:
    def filter_callback(tarinfo):
        file_or_dir_name = os.path.basename(tarinfo.name)
        if verbose:
            print(" ", tarinfo.name, "...")
        if file_or_dir_name.startswith(exclude_files_dirs_startswith):
            return None
        else:
            return tarinfo

    print("Backing up", dir_path_to_archive, "to", archive_file_path, "...")
    start_time = time.time()
    with tarfile.open(archive_file_path, mode='w:gz') as archive:
        archive.add(dir_path_to_archive, filter=filter_callback, arcname=dir_name_to_archive)
    print("Backup done in", "{:.4f}".format(time.time() - start_time), "seconds.")


def reset_print_options():
    # from https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.set_printoptions.html
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False,
                        threshold=1000, formatter=None)


def array_to_string(arr: np.ndarray) -> str:
    return np.array2string(arr, max_line_width=999999, threshold=9999999)


def int_to_onehot(integer: int, length: int) -> np.ndarray:
    onehot = np.zeros((length,))
    onehot[integer] = 1
    return onehot


# Copied from Keras because Keras import has side effects (which produce SIGABRT on non AVX-CPUs)
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def to_numpy_array(list_of_samples):
    return np.array(list_of_samples, dtype=np.float32)


def check_and_make_finite(arr: np.ndarray, print_prefix: str = "") -> bool:
    """
    Checks for NaN, inf, -inf. If found, print a warning and replace them with finite numbers.
    :param print_prefix:
    :param arr:
    :return:
    """
    # np.isfinite(arr) allocates new boolean array of shape arr.shape which is not ideal. Afaik, there is no faster way.
    # (Do not use np.isfinite(np.sum(arr)) as np.sum may create new infinite values (but it would work with np.nan))
    # (Do not use np.isfinite(np.min(arr)) as np.min ditches infinite values. np.max analogously ditches -inf values)
    if not np.isfinite(arr).all():
        # Replace NaN with zero and (-)inf with arr.dtype's smallest/largest value
        # (copy=False == > change arr values in place)
        finite_arr = np.nan_to_num(arr, copy=False)
        # Make sure copy in place worked:
        assert finite_arr is arr, (finite_arr, arr, finite_arr.dtype, arr.dtype)
        print(print_prefix, "Warning: Detected NaN, inf, and/or -inf and replaced them with finite numbers.", sep="")
        return True
    return False


@contextmanager
def memory_tracing(key_type: str = "lineno", limit: int = 15):
    """
    Traces memory consumption and prints memory-usage statistics when leaving the context
    :param key_type:
    :param limit:
    :return:
    """
    tracemalloc.start()
    print("--- Tracing memory... ---")
    try:
        # Do computation ...
        yield None
    finally:
        snapshot = tracemalloc.take_snapshot()
        # snapshot = snapshot.filter_traces((
        #     tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        #     tracemalloc.Filter(False, "<unknown>"),
        # ))
        top_stats = snapshot.statistics(key_type)
        print("--- Memory usage statistics: ---")
        print("Top %s lines:" % limit)
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback[0]
            # replace "/path/to/module/file.py" with "module/file.py"
            filename = os.sep.join(frame.filename.split(os.sep)[-2:])
            print("#%s: %s:%s: %.1f KiB"
                  % (index, filename, frame.lineno, stat.size / 1024))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print('    %s' % line)

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            print("%s other: %.1f KiB" % (len(other), size / 1024))
        total = sum(stat.size for stat in top_stats)
        print("\nTotal allocated size: %.1f KiB" % (total / 1024))

        # May also be useful:
        import objgraph
        print("\nTypes of most common instances:")
        objgraph.show_most_common_types(limit=limit)
        print("\nObjects that do not have any referents:")
        objgraph.get_leaking_objects()
        print("\nIncrease in peak object counts since last call:")
        objgraph.show_growth(limit=limit)
        print("\ntuple objects tracked by the garbage collector:")
        objgraph.by_type('tuple')
        print("\ndict objects tracked by the garbage collector:")
        objgraph.by_type('dict')
        print("--- End of memory tracing ---")
