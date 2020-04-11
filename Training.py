"""
Training and Testing component
"""

import csv
import itertools
import multiprocessing
import pathlib
import traceback
from time import sleep
from typing import Tuple, Callable, Union, Optional

from numpy import Inf
from scipy.special import softmax
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score, \
    roc_auc_score, f1_score, cohen_kappa_score, matthews_corrcoef, roc_curve, auc, balanced_accuracy_score
from tensorboard import summary as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorflow.python.eager.context import graph_mode
from tensorflow.python.eager.execution_callbacks import errstate, ExecutionCallback
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.keras.callbacks import TensorBoard

import EagerModel
from DataPreparation import DatasetCustodian, PREPARED_DATASET_PATH_ROOT
from Helpers import *

PARAM_COMB_BEST_SCORE_JSON_NAME = "best_score.json"
REPETITIONS_SUMMARY_JSON_NAME = "repetition_scores_summary.json"
OVERALL_BEST_SCORE_JSON_NAME = "overall_best_score.json"
SCORE_FORMAT_STRING = "{0:.6f}"


class GridSearchCheckpointInfo:
    """
    Class for encapsulating all information of a grid search checkpoint. Such a checkpoint is created from a given
    model and contains information about the model and its associated dataset custodian. Based on a
    GridSearchCheckpointInfo, a saved model's scores and attributions can be calculated.

    Additionally, a different dataset custodian can be specified. As a result, a model can be tested on a dataset on
    which it was not trained.
    """

    def __init__(self, saved_model_dir_path: str, saved_dataset_custodian_dir_name_or_path: str = None,
                 saved_dc_is_path: bool = False):
        """
        Creates a GridSearchCheckpointInfo from a saved model.
        :param saved_model_dir_path: Path to the model which should be loaded.
        :param saved_dataset_custodian_dir_name_or_path: If specified, the given dataset custodian is used instead of
        the model's associated one. Both the directory name of a dataset custodian or an absolute path are accepted (see
        saved_dc_is_path)
        :param saved_dc_is_path: Whether the dataset custodian is specified by its absolute path or its directory name
        """
        self.saved_model_dir_path = saved_model_dir_path
        print("Reading model from", self.saved_model_dir_path, "...")
        # Load model itself:
        self.model: tfkeras.Model = load_model_from_path(self.saved_model_dir_path)
        # Read associated data/information:
        print("Model info:")
        saved_model_dir_path_obj = pathlib.Path(saved_model_dir_path)
        self.model_params = from_json(read_file_content(saved_model_dir_path_obj.parents[0].joinpath("params.json")))
        self.model_scores = from_json(read_file_content(
            saved_model_dir_path_obj.parents[0].joinpath(saved_model_dir_path_obj.name + "_scores.json")))
        self.train_threshold = self.model_scores["train"]["scores"].get("threshold", None)
        self.validation_threshold = self.model_scores["validation"]["scores"].get("threshold", None)
        assert saved_model_dir_path_obj.parents[1].name == "GridSearch"

        print("  Parameters:", self.model_params)
        print("  Scores:", self.model_scores)
        print("  Validation threshold:", self.validation_threshold)

        # Load model's associated dataset custodian since is necessary in both cases:
        model_dc = DatasetCustodian.load(saved_model_dir_path_obj.parents[2], is_path=True)

        if saved_dataset_custodian_dir_name_or_path is None:
            # No dataset is given. Use model's dataset:
            self.dataset_custodian: DatasetCustodian = model_dc
            self.model_to_dataset_mapping = None
        else:
            # Dataset is given. Load it:
            dataset_dc: DatasetCustodian = DatasetCustodian.load(saved_dataset_custodian_dir_name_or_path,
                                                                 is_path=saved_dc_is_path)
            # Check if mapping is necessary
            if dataset_dc.get_labels() != model_dc.get_labels():
                if len(dataset_dc.get_labels()) == len(model_dc.get_labels()):
                    # Len matches and score calculation is possible. However, results may be senseless if labels do not
                    #  match.
                    print("WARNING: Model labels", model_dc.get_labels(), "do not match dataset labels",
                          dataset_dc.get_labels())
                elif len(dataset_dc.get_labels()) < len(model_dc.get_labels()):
                    # Model can distinguish more classes than dataset has. Try to automatically create mapping:
                    self.model_to_dataset_mapping = {}
                    for model_label in model_dc.get_labels():
                        if model_label in dataset_dc.get_labels():
                            self.model_to_dataset_mapping[model_dc.label_to_int_mapping[model_label]] \
                                = dataset_dc.label_to_int_mapping[model_label]
                        else:
                            self.model_to_dataset_mapping[model_dc.label_to_int_mapping[model_label]] \
                                = dataset_dc.label_to_int_mapping[dataset_dc.get_no_bug_label()]
                    print("WARNING: Model predicts more classes than present in the dataset.",
                          "Created model-to-dataset mapping, which maps each non-present class to non-vulnerable",
                          "class:", self.model_to_dataset_mapping)
                elif len(dataset_dc.get_labels()) > len(model_dc.get_labels()):
                    print("WARNING: Model predicts less classes than present in the dataset.")

            # Use requested dataset:
            self.dataset_custodian: DatasetCustodian = dataset_dc

        self.dataset_custodian_info \
            = from_json(read_file_content(os.path.join(self.dataset_custodian.prepared_data_dir_path, "info.json")))
        print("Dataset info:", self.dataset_custodian_info)
        self.dataset_custodian.print_dataset_statistics()


# Custom Tensorboard which can be used if model.fit is called with epochs=1 in a loop
class CustomTensorBoard(TensorBoard):
    # Remember the actual epoch count:
    epoch_number: int
    # Remember whether set_model has been already called
    set_model_called: bool

    def __init__(self, model_graph, **kwargs):
        self.model_graph = model_graph
        self.epoch_number = 0
        self.set_model_called = False
        super().__init__(**kwargs)

    # Enable the Custom Scalar page of TensorBoard by specifying the layout for that page
    # This needs to be called when self.writer is opened (i.e. after set_model has been called)
    def add_custom_scalar_layout(self):
        # This action does not have to be performed at every step, so the action is not
        # taken care of by an op in the graph. We only need to specify the layout once.
        # We only need to specify the layout once (instead of per step).
        layout = layout_pb2.Layout(
            category=[
                layout_pb2.Category(
                    title='train, validation, and test scores',
                    chart=[
                        layout_pb2.Chart(
                            title='train scores',
                            multiline=layout_pb2.MultilineChartContent(
                                tag=[r'train_*'],
                            )),
                        layout_pb2.Chart(
                            title='validation scores',
                            multiline=layout_pb2.MultilineChartContent(
                                tag=[r'val_*'],
                            )),
                        layout_pb2.Chart(
                            title='test scores',
                            multiline=layout_pb2.MultilineChartContent(
                                tag=[r'test_*'],
                            )),

                    ]),
                layout_pb2.Category(
                    title='trig functions',
                    chart=[
                        layout_pb2.Chart(
                            title='wave trig functions',
                            multiline=layout_pb2.MultilineChartContent(
                                tag=[r'train_*', r'val_*'],
                            )),
                        # The range of tangent is different. Let's give it its own chart.
                        layout_pb2.Chart(
                            title='tan',
                            multiline=layout_pb2.MultilineChartContent(
                                tag=[r'trigFunctions/tangent'],
                            )),
                        layout_pb2.Chart(
                            title='baz',
                            margin=layout_pb2.MarginChartContent(
                                series=[
                                    layout_pb2.MarginChartContent.Series(
                                        value='loss/baz/scalar_summary',
                                        lower='baz_lower/baz/scalar_summary',
                                        upper='baz_upper/baz/scalar_summary'),
                                ],
                            )),
                    ],
                    # This category we care less about. Let's make it initially closed.
                    closed=True),
            ])
        layout_summary = summary_lib.custom_scalar_pb(layout)

        if not tensorflow.executing_eagerly():
            self.writer.add_summary(layout_summary)
        else:
            with graph_mode():
                tensorflow.summary.FileWriter(logdir=self.log_dir).add_summary(layout_summary)

    def set_model(self, model):
        # This is called on each model.fit call and therefore for each actual epoch-loop. Make sure this is only called
        # once to avoid that the graph and other data on is written multiple times
        if not self.set_model_called:
            super().set_model(model)
            self.set_model_called = True
            # Add Custom Scalar layout and graph because self.writer is set now:
            self.add_custom_scalar_layout()
            with self.writer.as_default():
                tensorflow.contrib.summary.graph(self.model_graph, step=self.epoch_number, name="network_graph")

    def add_scores(self, scores):
        self._write_custom_summaries(self.epoch_number, scores)
        # self._write_logs(scores, self.epoch_number)

    def on_epoch_end(self, epoch, logs=None):
        assert epoch == 0, epoch
        super().on_epoch_end(self.epoch_number, logs)
        self.epoch_number += 1

    def on_train_end(self, logs=None):
        # This is called on the end of each model.fit call. super.on_train_end closes the underlying writer object.
        # Therefore dont call super.on_train_end here
        pass

    def call_super_on_train_end(self):
        super().on_train_end(None)


class DatasetCustodianSequence(tfkeras.utils.Sequence):
    """
    "Sequence" for yielding training sequences to fit_generator, predict_generator and so on.
    This yields batches of the given size. Each batch contains sequences generated by the given dataset custodian.
    Either all sequences of the given split are yielded or the sequences specified by the given list of indices. The
    indices should match the dataset custodian's possible indices. Shuffling of all sequences should be done here and
    not by fit_generator's shuffle argument. Shuffling in here really shuffles all sequences and not only the order
    of the batches.
    """

    actual_max_sequence_length: int = None
    get_sequence_from_index_func: Callable = None
    traintestval_split_indices_to_use: List[int]  # indices (e.g. train, test or validation split indices)
    traintestval_split_indices_to_use_length: int
    batch_size: int
    shuffle_on_end: bool

    # returned_sequence_count = None

    # Variables for testing of sequence (slows training down and only works with use_multiprocessing=False)
    # indices = None
    # indices2 = None

    def __init__(self, dataset_custodian: DatasetCustodian, split_name_or_indices: Union[str, List[int], np.ndarray],
                 batch_size, shuffle_on_end=True):
        # Handle split name or indices:
        if isinstance(split_name_or_indices, str):
            # Get indices from dataset custodian:
            self.traintestval_split_indices_to_use = dataset_custodian.traintestval_split_indices[split_name_or_indices]
        elif isinstance(split_name_or_indices, (list, np.ndarray)):
            self.traintestval_split_indices_to_use = split_name_or_indices
        else:
            assert False, (type(split_name_or_indices), split_name_or_indices)
        # Make sure there are only ints:
        assert all(isinstance(index, (int, np.integer)) for index in self.traintestval_split_indices_to_use), \
            self.traintestval_split_indices_to_use
        self.traintestval_split_indices_to_use_length = len(self.traintestval_split_indices_to_use)
        self.class_weights = dataset_custodian.get_class_weights()
        self.get_sequence_from_index_func = dataset_custodian.get_sequence_tuple_for_tensorflow_from_index
        self.actual_max_sequence_length = dataset_custodian.actual_max_sequence_length
        self.batch_size = batch_size
        self.shuffle_on_end = shuffle_on_end
        self.reset_sequence()
        # print("Initialized DatasetCustodianSequence, ", end="")
        # Do not store the dataset custodian because the whole DatasetCustodianSequence is copied when multiprocessing
        # is used.

    def reset_sequence(self):
        # self.returned_sequence_count = 0
        # self.indices = []
        # self.indices2 = []
        if self.shuffle_on_end:
            # print("Shuffling split indices ...")
            random.shuffle(self.traintestval_split_indices_to_use)

        # print(" [", end="")

    def __len__(self):
        """
        :return: number of batches needed to feed each sequence to model exactly once
        """
        # print("__len__ called")
        return int(np.ceil(self.traintestval_split_indices_to_use_length / float(self.batch_size)))

    def __getitem__(self, batch_index) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param batch_index: index of the batch to return in [0, len(self)[
        :return: the batch_index'th batch, i.e. a tuple of the batch' sequence and its target labels
        """
        # print("  batch_index" + str(batch_index) + "/" + str(len(self)) + "  ")
        # print(self.traintestval_split_indices_to_use)
        # if batch_index in self.indices:
        #     assert False, str(batch_index) + " in " + str(self.indices)
        # else:
        #     self.indices.append(batch_index)

        batch_samples, batch_labels, batch_weights = [], [], []
        split_index_batch_start = batch_index * self.batch_size
        for i in range(self.batch_size):
            split_index = split_index_batch_start + i
            if split_index < self.traintestval_split_indices_to_use_length:
                sequence_index = self.traintestval_split_indices_to_use[split_index]

                # assert sequence_index not in self.indices2
                # self.indices2.append(sequence_index)
                # print("split_index" + str(split_index))
                # Get sample data:
                sequence_as_array, label_int_as_array, label_onehot_as_array, corresponding_ast_nodes \
                    = self.get_sequence_from_index_func(sequence_index)
                batch_samples.append(sequence_as_array)
                batch_labels.append(label_onehot_as_array)
                batch_weights.append(self.class_weights[label_int_as_array])
                # self.returned_sequence_count += 1
            else:
                # Batch which contains the last sequences. Therefore, this batch might not be complete.
                # As Tensorflow/Keras does not query the batches in order, this may happen anywhere during the epoch.
                # Therefore, nothing special needs to be done.
                assert len(batch_samples) > 0
                # Do not try to read behind the end of sequences again:
                break
        # Progress indicator:
        # print("#", end="")
        # if self.returned_sequence_count == self.traintestval_split_indices_to_use_length:
        #     print("]")

        # batch_samples_as_array = to_numpy_array(batch_samples)
        # print(batch_samples_as_array.shape)
        # print_histogram(Counter([len(batch_sample) for batch_sample in batch_samples]))

        # Pad batch of sequences:
        if tensorflow.executing_eagerly():
            # pad_sequences will determine the length from the batch
            maxlen = None
        else:
            # Pad sequences to max sequence length determined by dataset
            maxlen = self.actual_max_sequence_length
            assert maxlen is not None
        batch_samples_as_array = pad_sequences(batch_samples, maxlen=maxlen, dtype=np.float32)
        # batch_samples_as_array = trim_zeros_3d(batch_samples_as_array, axis=2)
        # print("batch shape:", batch_samples_as_array.shape)
        # print(to_numpy_array(batch_labels))
        # print(to_numpy_array(batch_weights))
        return batch_samples_as_array, to_numpy_array(batch_labels), to_numpy_array(batch_weights)

    def on_epoch_end(self):
        """
        Called at the end of every epoch.
        :return:
        """
        # printing does not work well in here. print outputs are missing sometimes (probably because of multi
        # threading/processing?). But this method is executed even if there is no visible output from a print statement

        # print("done")
        # print("on_epoch_end", self.returned_sequence_count, len(self.traintestval_split_indices_to_use))

        # Make sure every sample was used (does not work with use_multiprocessing)
        # assert self.returned_sequence_count == len(self.traintestval_split_indices_to_use)
        # assert len(self.indices2) == len(self.traintestval_split_indices_to_use), (len(self.indices2),
        # len(self.traintestval_split_indices_to_use))
        # assert (len(self.indices) * self.batch_size) >= len(self.traintestval_split_indices_to_use)
        # assert (len(self.indices) * (self.batch_size - 1)) < len(self.traintestval_split_indices_to_use)

        # Reset sequence:
        self.reset_sequence()


def predictions_to_label_numbers(predicted_labels_onehot, threshold_to_use, class_to_use=1):
    """
    Maps the model prediction values/scores/"probabilities" to a label number. In binary classification case, this
    will binarize the model prediction using the given threshold. In multi class classification this will use argmax.

    :param predicted_labels_onehot: model "onehot" output
    :param threshold_to_use:
    :return:
    """
    try:
        # Look for float threshold first:
        threshold_to_use = float(threshold_to_use)
        # Use the threshold to make predicted label numbers out of the the predicted "onehots"
        # (1 * x to convert boolean x to integer x)
        return 1 * (predicted_labels_onehot[:, class_to_use] >= threshold_to_use)
    except (ValueError, TypeError):
        # Not a float threshold. Check for argmax:
        if threshold_to_use == "argmax":
            # Use maximum value:
            return np.argmax(predicted_labels_onehot, axis=1)
        else:
            assert False, ("Invalid threshold", threshold_to_use, type(threshold_to_use))


def get_scores(model: tfkeras.Model, dataset_custodian: DatasetCustodian,
               split_name_or_indices: Union[str, List[int]] = None,
               sequences_and_labels_onehot: Tuple[List[np.ndarray], List[np.ndarray], List[str]] = None,
               threshold_to_use: Optional[Union[float, str]] = None, apply_softmax=True,
               no_print: bool = False, model_to_dataset_mapping: Dict[int, int] = None) \
        -> Tuple[float, str, Dict[str, Union[float, np.ndarray]], str]:
    """
    Evaluate the given model's prediction power by comparing the model's predictions for the given split/indices with
    the corresponding target outputs.

    :param apply_softmax: Whether softmax function should be applied to the model predictions.
    :param sequences_and_labels_onehot: As third element, a descriptive string for each sequence can be provided
    :param model:
    :param dataset_custodian:
    :param split_name_or_indices: Either a valid split name for the given dataset custodian (usually "train", "test", or
    "validation") or a list of indices into the given dataset custodian list of prepared sequences
    :param threshold_to_use: A float or "argmax". In binary classification case None is also allowed, to specify that
    the best threshold value shall be determined from data
    :param no_print:
    :return:
    """
    assert (split_name_or_indices is None) != (sequences_and_labels_onehot is None), \
        "split_name_or_indices XOR sequences_and_labels_onehot must be specified."
    is_binary_classification = len(dataset_custodian.get_label_ints()) == 2
    resulting_scores = {}
    summary_strings = []

    def sum_and_print(*args, **kwargs):
        summary_strings.extend([str(arg) + " " for arg in args])
        summary_strings.append("\n")
        if not no_print:
            print(*args, **kwargs)

    if not no_print:
        print("Computing scores for ", end="")
        if split_name_or_indices is not None:
            print("split", split_name_or_indices, "...")
        else:
            print(len(sequences_and_labels_onehot[0]), "sequence-label-pairs ...")

    # # start = time.time()
    # # Let the model predict every sequence of the given split name:
    # for sequence_index in dataset_custodian.traintestval_split_indices[split_name_or_indices]:
    #     sequence, label_int, label_onehot, corresponding_ast_nodes \
    #         = dataset_custodian.get_sequence_tuple_for_tensorflow_from_index(sequence_index)
    #
    #     # Only store the prediction result and the target label (for memory reasons):
    #     # predict expects a list/array of sequences:
    #     if not tensorflow.executing_eagerly():
    #         sequence = pad_sequences([sequence],
    #                                  maxlen=dataset_custodian.actual_max_sequence_length,
    #                                  dtype=np.float32
    #                                  )[0]
    #     predicted_labels_onehot.append(model.predict(to_numpy_array([sequence]))[0])
    #     target_labels_onehot.append(label_onehot)
    #
    # # Make arrays out of list of predictions and labels (to emulate result as if predict was called on all sequences)
    # # (to_numpy_array may create NaN if data was float64 before)
    # predicted_labels_onehot = to_numpy_array(predicted_labels_onehot)
    # target_labels_onehot = to_numpy_array(target_labels_onehot)

    # print("\n1", time.time() - start)
    # start = time.time()
    sequence_descriptions = None
    if split_name_or_indices is not None:
        # Let the model predict every sequence of the given split name:
        predicted_labels_onehot, target_labels_onehot = model.predict_generator(
            DatasetCustodianSequence(dataset_custodian,
                                     split_name_or_indices,
                                     64  # TODO: other value?
                                     ),
            verbose=0,
            max_queue_size=10 * multiprocessing.cpu_count(),
            use_multiprocessing=False,
            workers=multiprocessing.cpu_count())

        assert len(predicted_labels_onehot) == len(target_labels_onehot)
        assert len(predicted_labels_onehot) > 0, ("There must be at least one input and output. Used split: ",
                                                  split_name_or_indices)
    else:
        assert sequences_and_labels_onehot is not None  # to silence PyCharm warning below
        predicted_labels_onehot = []
        # Let the model predict every sequence of the given split name:
        for sequence in sequences_and_labels_onehot[0]:
            predicted_labels_onehot.append(model.predict(to_numpy_array([sequence]))[0])

        # Make arrays out of list of predictions and labels (to emulate result as if predict was called on all
        # sequences)
        predicted_labels_onehot = to_numpy_array(predicted_labels_onehot)
        target_labels_onehot = to_numpy_array(sequences_and_labels_onehot[1])
        if len(sequences_and_labels_onehot) > 2:
            sequence_descriptions = sequences_and_labels_onehot[2]

    # print("\n2", time.time() - start)
    # assert np.allclose(predicted_labels_onehot,predicted_labels_onehot2),
    # (predicted_labels_onehot,predicted_labels_onehot2)
    # assert np.array_equal(target_labels_onehot, target_labels_onehot2), (target_labels_onehot, target_labels_onehot2)

    # Check for NaN and infinity and remove it to avoid exceptions in score calculation
    check_and_make_finite(predicted_labels_onehot, print_prefix="get_scores: ")

    if apply_softmax:
        sum_and_print("Applying softmax to model predictions ...")
        predicted_labels_onehot = softmax(predicted_labels_onehot, axis=1)

    # target labels onehot to label numbers:
    target_label_numbers = np.argmax(target_labels_onehot, axis=1)
    # predictions "onehot" to label numbers:

    # Print PR curve values in csv format:
    # if not no_print:
    #     ps, rs, thresholds = precision_recall_curve(y_true=target_label_numbers,
    #                                                 probas_pred=predicted_labels_onehot[:, 1],
    #                                                 pos_label=1)
    #     for p, r, threshold in zip(ps, rs, thresholds):
    #         print(p, ",", r, ",", threshold)

    # Determine the threshold from data if no threshold-to-use is provided and if it is binary classification
    if is_binary_classification:
        if threshold_to_use is None:
            # Determine best threshold from data assuming 1 is the positive class:
            fprs, tprs, thresholds = roc_curve(y_true=target_label_numbers,
                                               y_score=predicted_labels_onehot[:, 1],
                                               pos_label=1)

            debug_plots = False
            if debug_plots:
                print(auc(fprs, tprs))
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(fprs, tprs)
                plt.show()
                plt.figure()

            best_distance = 1
            for fpr, tpr, threshold in zip(fprs, tprs, thresholds):
                # Distance to tpr=1, fpr=0 (hypot is euclidean norm)
                dist = math.hypot(1 - tpr, 0 - fpr)

                if debug_plots:
                    print(dist, threshold, auc([0, fpr, 1], [0, tpr, 1]),
                          confusion_matrix(y_true=target_label_numbers,
                                           y_pred=1 * (predicted_labels_onehot[:, 1] >= threshold)).tolist())

                if dist < best_distance:
                    best_distance = dist
                    threshold_to_use = threshold
                    if debug_plots:
                        plt.plot([0, fpr, 1], [0, tpr, 1])
                        plt.show()
    else:
        assert not isinstance(threshold_to_use, float), ("Float threshold can not be provided in non-binary case",
                                                         threshold_to_use)

    # From here threshold_to_use must be a non-none value:
    if threshold_to_use is None:
        threshold_to_use = "argmax"
    # Convert predicted "onehot" vectors for scores/probabilities to label numbers:
    predicted_label_numbers = predictions_to_label_numbers(predicted_labels_onehot, threshold_to_use)

    # Map model prediction integers to dataset target integer (necessary if model is assessed on dataset with different
    #  labels)
    if model_to_dataset_mapping is not None:
        assert len(predicted_label_numbers.shape) == 1, predicted_label_numbers
        for i in range(len(predicted_label_numbers)):
            predicted_label_numbers[i] = model_to_dataset_mapping[predicted_label_numbers[i]]

    sum_and_print("Using threshold", threshold_to_use, "for label selection.")
    # Remember the threshold used
    resulting_scores["threshold"] = threshold_to_use

    # Print descriptions of wrong predictions:
    if sequence_descriptions is not None:
        # Get indices of wrong predictions:
        # np.where with only condition given returns a N-dim tuple of arrays of indices where N is the dim of condition.
        # Here, N is one. Therefore, [0] selects the desired indices:
        wrong_indices = np.where(target_label_numbers != predicted_label_numbers)[0]
        for wrong_index in wrong_indices:
            sum_and_print("Wrong prediction",
                          dataset_custodian.int_to_label_mapping[predicted_label_numbers[wrong_index]],
                          "(" + str(predicted_labels_onehot[wrong_index]) + ";",
                          "actual:", dataset_custodian.int_to_label_mapping[target_label_numbers[wrong_index]] + ")",
                          "for", sequence_descriptions[wrong_index])

    accuracy = accuracy_score(y_true=target_label_numbers, y_pred=predicted_label_numbers)
    sum_and_print('Accuracy:', accuracy)
    resulting_scores["accuracy"] = accuracy

    balanced_accuracy = balanced_accuracy_score(y_true=target_label_numbers, y_pred=predicted_label_numbers)
    sum_and_print('Balanced Accuracy:', balanced_accuracy)
    resulting_scores["balanced_accuracy"] = balanced_accuracy

    classification_report_str = classification_report(y_true=target_label_numbers, y_pred=predicted_label_numbers,
                                                      labels=dataset_custodian.get_label_ints(),
                                                      target_names=dataset_custodian.get_labels(),
                                                      digits=4)
    sum_and_print("Classification Report",
                  "(For NO_FLAW class, precision and recall must be swapped (as positive is expected to be vulnerable)."
                  "Note that in binary classification, recall of the positive class is also known as \"sensitivity\";",
                  "recall of the negative class is \"specificity\".):\n", classification_report_str)
    resulting_scores["classification_report"] = classification_report_str

    f1_weighted_score = f1_score(y_true=target_label_numbers, y_pred=predicted_label_numbers, average="weighted",
                                 labels=dataset_custodian.get_label_ints())
    sum_and_print("F1-Score (weighted; see also classification report): " + str(f1_weighted_score))
    resulting_scores["f1_weighted_score"] = f1_weighted_score

    f1_macro_score = f1_score(y_true=target_label_numbers, y_pred=predicted_label_numbers, average="macro",
                              labels=dataset_custodian.get_label_ints())
    sum_and_print("F1-Score (macro; see also classification report): " + str(f1_macro_score))
    resulting_scores["f1_macro_score"] = f1_macro_score

    cohens_kappa_score = cohen_kappa_score(y1=target_label_numbers, y2=predicted_label_numbers,
                                           labels=dataset_custodian.get_label_ints())
    sum_and_print("Cohen's Kappa Score (CK): " + str(cohens_kappa_score))
    sum_and_print("  - Normalized to [0, 1]: " + str((1 + cohens_kappa_score) / 2))
    resulting_scores["cohens_kappa_score"] = cohens_kappa_score

    mcc = matthews_corrcoef(y_true=target_label_numbers, y_pred=predicted_label_numbers)
    sum_and_print("Matthews correlation coefficient (MCC): " + str(mcc))
    sum_and_print("  - Normalized to [0, 1]: " + str((1 + mcc) / 2))
    resulting_scores["mcc"] = mcc
    sum_and_print("  - Key for CK and MCC: +1 represents a perfect prediction,",
                  "0 an average random prediction and -1 and inverse prediction")

    if is_binary_classification:
        # Compute AROC (metric that is independent of class sizes):
        aroc_bin_score = roc_auc_score(y_true=target_label_numbers, y_score=predicted_label_numbers)
        sum_and_print("Area under the ROC curve (with custom threshold for class 1 (see above)):", str(aroc_bin_score))
        resulting_scores["aroc_bin_score"] = aroc_bin_score

        # pseudo probability predictions/scores for label 0
        samples_predicted_scores_c0 = predicted_labels_onehot[:, 0]
        # pseudo probability predictions/scores for label 1
        samples_predicted_scores_c1 = predicted_labels_onehot[:, 1]

        aroc_c1_score = roc_auc_score(y_true=target_label_numbers, y_score=samples_predicted_scores_c1)
        sum_and_print("Area under the ROC curve",
                      "(with automatic threshold from pseudo probability for class 1):", str(aroc_c1_score))
        resulting_scores["aroc_c1_score"] = aroc_c1_score

        # Compute Area under the precision-recall curve:
        auprc_c0_score = average_precision_score(y_true=target_label_numbers, y_score=samples_predicted_scores_c0,
                                                 pos_label=0)  # pos_label=0 is the same as y_true=1-target_label_number
        sum_and_print("Area under the precision-recall curve",
                      "(with automatic threshold from pseudo probability for class 0):", auprc_c0_score)
        resulting_scores["auprc_c0_score"] = auprc_c0_score

        auprc_c1_score = average_precision_score(y_true=target_label_numbers, y_score=samples_predicted_scores_c1,
                                                 pos_label=1)
        sum_and_print("Area under the precision-recall curve",
                      "(with automatic threshold from pseudo probability for class 1):", str(auprc_c1_score))
        resulting_scores["auprc_c1_score"] = auprc_c1_score

    conf_matrix = confusion_matrix(y_true=target_label_numbers, y_pred=predicted_label_numbers,
                                   labels=dataset_custodian.get_label_ints())
    sum_and_print("Confusion Matrix:\n", conf_matrix)
    sum_and_print("  Key: - True labels  [::]\n                   Predicted Labels")
    sum_and_print("       - [[True Positives  False Negatives]\n          [False Positives  True Negatives]]")
    sum_and_print("       - Positive/Negative: What the classifier outputs (vulnerable/non-vulnerable),",
                  "True/False: Whether the classifier output is correct.")
    resulting_scores["conf_matrix"] = conf_matrix

    main_score_key = "f1_macro_score"
    main_score_name = "F1"

    return resulting_scores[main_score_key], main_score_name, resulting_scores, "".join(summary_strings)


def model_scores_to_csv(json_file_path):
    """
    Computes "one-vs-rest" scores for each class based on the classification report and the confusion matrix in the
     given scores file. This is useful to retrieve separate scores for each CWE/class in case of a model trained on a
     merged dataset. In addition, sorted versions of training, validation, and test confusion matrices are generated.
     These are row- and column-wise sorted by the respective CWE ID. All results are stored on disk next to the given
     scores file.
    :param json_file_path: Absolute path to the file which contains a model's scores as json
    :return:
    """
    scores = from_json(read_file_content(json_file_path))
    class_wise_scores = None
    # For train, val, and test scores:
    for score_key, score_obj in scores.items():
        score_obj = score_obj["scores"]
        class_report: str = score_obj["classification_report"]

        # Create list of cwes from classification report:
        cwes = []
        for class_report_row in class_report.split("\n"):
            cwe_str_index = class_report_row.find("CWE")
            if cwe_str_index >= 0:
                cwes.append(int(class_report_row[cwe_str_index + 3: class_report_row.find("_")]))

        if class_wise_scores is None:
            class_wise_scores = [{"cwes": cwes[i]} for i in range(len(cwes))]
        else:
            assert [score["cwes"] for score in class_wise_scores] == cwes

        # Create sorted confusion matrix:
        conf_matrix = np.array(score_obj["conf_matrix"])
        assert len(conf_matrix.shape) == 2 and conf_matrix.shape[0] == conf_matrix.shape[1], conf_matrix.shape
        print(conf_matrix.shape)
        class_count = conf_matrix.shape[0]
        assert class_count == len(cwes)
        sample_sum = np.sum(conf_matrix)

        sorted_conf_matrix = conf_matrix.copy()
        cwes_to_sort = list(cwes)
        cwes_to_sort[-1] = 999
        new_order = np.argsort(np.array(cwes_to_sort)).tolist()
        for row in range(class_count):
            for col in range(class_count):
                sorted_conf_matrix[new_order.index(row), new_order.index(col)] = conf_matrix[row, col]
        # print(cwes, "\n", conf_matrix)
        # print(new_order, sorted(cwes_to_sort), "\n", sorted_conf_matrix)

        # Save sorted confusion matrix:
        np.savetxt(json_file_path + "." + score_key + "_cm.csv", sorted_conf_matrix, delimiter=",")

        # Compute separate "one-vs-rest" scores for each cwe:
        scores = []
        for class_index in range(class_count):
            tp = conf_matrix[class_index, class_index]
            fp = np.sum(conf_matrix[:, class_index]) - tp
            fn = np.sum(conf_matrix[class_index, :]) - tp

            scores = {}

            def add_score(name, value):
                scores[score_key + "_" + name] = value

            def get_score(name):
                return scores[score_key + "_" + name]

            add_score("support", tp + fn)
            add_score("precision", tp / (tp + fp) if get_score("support") > 0 else 0)
            add_score("recall", tp / (tp + fn) if get_score("support") > 0 else 0)
            add_score("F1", 2 * get_score("precision") * get_score("recall")
                            / (get_score("precision") + get_score("recall")) if get_score("support") > 0 else 0)
            class_wise_scores[class_index] = {**(class_wise_scores[class_index]), **scores}
    class_wise_scores.sort(key=lambda x: x["cwes"] if x["cwes"] != 0 else 999)
    means = {}
    for class_wise_score in class_wise_scores:
        for score, score_val in class_wise_score.items():
            means.setdefault(score, 0)
            means[score] += score_val
    for mean_key in means.keys():
        means[mean_key] /= len(class_wise_scores)
    print("Means:", means)

    csv_headings = class_wise_scores[0].keys()
    with open(json_file_path + ".csv", "w") as csv_file:
        print("Writing to csv", csv_file.name)
        writer = csv.writer(csv_file)
        writer.writerow([heading.replace("_", "-") for heading in csv_headings])
        for class_wise_score in class_wise_scores:
            print(class_wise_score)
            writer.writerow([class_wise_score[key] for key in csv_headings])


# @total_ordering  # adds other comparision operators
class ScoreData:
    """
    Class representing a model's score. In addition to the actual score, the context is captured in which the score was
    achieved.
    """

    min_reduction_percentage = 25  # %

    def __init__(self, model_name="", param_comb=None, repetition=0, epoch=0, main_score_name=None, main_score=0.0,
                 train_loss=Inf):
        self.main_score_name = main_score_name
        self.main_score = main_score
        self.train_loss = train_loss
        self.model_name = model_name
        self.epoch = epoch
        self.repetition = repetition
        self.param_comb = param_comb

    @staticmethod
    def assert_main_score_names_match(first: "ScoreData", second: "ScoreData"):
        if first.main_score_name is not None and second.main_score_name is not None:
            assert first.main_score_name == second.main_score_name, (first, second)

    def main_score_greater_than(self, other: "ScoreData"):
        self.assert_main_score_names_match(self, other)
        return self.main_score > other.main_score

    def main_score_equal(self, other: "ScoreData"):
        self.assert_main_score_names_match(self, other)
        return self.main_score == other.main_score

    def secondary_score_greater_than(self, other: "ScoreData", train_loss_reduction_percentage):
        # score "trainloss" A is greater than score "trainloss" B if A < B
        if self.train_loss < other.train_loss:
            # Make sure there is a significant reduction in train loss
            return (other.train_loss - self.train_loss) / other.train_loss * 100 >= train_loss_reduction_percentage
        else:
            return False

    # self > other
    def greater_than(self, other: "ScoreData", train_loss_reduction_percentage: float = 10.0):
        if self.main_score_greater_than(other):
            return True
        elif self.main_score_equal(other):
            return self.secondary_score_greater_than(other, train_loss_reduction_percentage)
        else:
            return False

    # self > other
    def __gt__(self, other: "ScoreData"):
        return self.greater_than(other)

    def compact_repr(self):
        """
        Returns a compact, human-readable, one-line representation of this score.
        :return:
        """
        return self.main_score_name + " " + SCORE_FORMAT_STRING.format(self.main_score) + " (tr. loss " \
               + SCORE_FORMAT_STRING.format(self.train_loss) + ") rep. " + str(self.repetition) + "; ep. " \
               + str(self.epoch) + " (" + self.model_name + ")"

    def __repr__(self):
        return str(self.__dict__)

    def to_json(self) -> str:
        return to_json(self)

    @classmethod
    def from_json(cls, json_string: str):  # -> "ScoreData":
        return from_json(json_string, ScoreData)


class GridSearch:
    """
    Class to perform training of a model for the Cartesian product of given parameter values.
    """
    dataset_custodian: DatasetCustodian
    parameter_grid: Dict[str, List]
    verbosity: int

    def __init__(self, dataset_custodian, parameter_grid):
        self.dataset_custodian = dataset_custodian
        verbosity = parameter_grid.pop("verbosity")
        assert len(verbosity) == 1, verbosity
        self.verbosity = verbosity[0]
        self.parameter_grid = parameter_grid

    @classmethod
    def print_overall_best_score_data(cls, overall_best_score: ScoreData, prefix="", suffix="\n"):
        print(prefix, "Best overall score ", overall_best_score.compact_repr(), " ", overall_best_score, suffix, sep="",
              end="")

    @staticmethod
    def filter_and_prefix_scores(scores: Dict[str, object], prefix, type_startswith):
        """
        Returns a copy of scores that contains only items whose value's type starts with type_startswith and whose keys
        are prefixed with prefix
        :param scores: Dictionary of scores. Keys are score names and values are score values.
        :param prefix:
        :param type_startswith:
        :return:
        """
        return {prefix + sn: sv for sn, sv in scores.items()
                if type(sv).__name__.startswith(type_startswith)}

    def search(self):
        """
        Performs the grid search. During this, best models, scores, and associated information are saved on disk as
        specified by the GridSearch's dataset custodian.
        :return: The ScoreData object of the best model
        """
        overall_start_time = time.time()
        # Get all possible parameter combinations as list
        parameter_combinations = list(dict(zip(self.parameter_grid, x))
                                      for x in itertools.product(*self.parameter_grid.values()))
        # Current grid search root path:
        gridsearch_root_path = os.path.join(self.dataset_custodian.prepared_data_dir_path, "GridSearch")
        if create_dir_if_necessary(gridsearch_root_path):
            # Write gridsearch parameters to file:
            write_file_content(os.path.join(gridsearch_root_path, "gridsearch_" + get_timestamp() + ".json"),
                               to_json({"train_parameter_values": self.parameter_grid,
                                        "param_combinations": parameter_combinations}))
        # Overall best file path:
        overall_best_score_file_path = os.path.join(gridsearch_root_path, OVERALL_BEST_SCORE_JSON_NAME)
        overall_best_score_data = ScoreData()
        if os.path.exists(overall_best_score_file_path):
            # Overall best score may be non-None from previous run!
            overall_best_score_data = ScoreData.from_json(read_file_content(overall_best_score_file_path))
        print("Searching best of", len(parameter_combinations), "parameter combinations ...")

        param_comb_count = 0
        # For pretty printing:
        param_combs_digits = int(math.log10(len(parameter_combinations))) + 1
        # Outer grid search loop over each parameter combination:
        for param_combination in parameter_combinations:
            param_comb_count += 1
            param_comb_start_time = time.time()
            param_comb_count_str = ("{:0" + str(param_combs_digits) + "d}/{}").format(param_comb_count,
                                                                                      len(parameter_combinations))
            if self.verbosity >= 1:
                print("\nParameter combination", param_comb_count_str + ":", param_combination)

            # Create model directory name from hash because string representation of params is too long (>255 chars)
            #  for file name
            curr_params_as_string = str(param_combination)
            # Directory name: MD5 plus start of all-parameter-string:
            param_comb_dir_name = as_safe_filename(get_md5_string(curr_params_as_string) + "_"
                                                   + ",".join((str(value) for value in param_combination.values())))
            param_comb_dir_name = param_comb_dir_name[0:255]
            param_comb_dir_path = os.path.join(gridsearch_root_path, param_comb_dir_name)
            # Create directory path if necessary:
            if create_dir_if_necessary(param_comb_dir_path, self.verbosity >= 2):
                # Write complete parameters as json:
                write_file_content(os.path.join(param_comb_dir_path, "params.json"), to_json(param_combination))

            # Create TensorBoard directory path if necessary:
            tensorboard_dir_root_path = os.path.join(param_comb_dir_path, "tensorboard")
            create_dir_if_necessary(tensorboard_dir_root_path, self.verbosity >= 2)

            param_comb_best_score_file_path = os.path.join(param_comb_dir_path, PARAM_COMB_BEST_SCORE_JSON_NAME)
            param_comb_best_score_data = ScoreData()
            if os.path.exists(param_comb_best_score_file_path):
                # Best score may be there from previous run!
                param_comb_best_score_data = ScoreData.from_json(read_file_content(param_comb_best_score_file_path))

            # Copy params dict because it is modified below
            param_combination_model_params = dict(param_combination)
            # Extract and remove non-model params:
            epochs = param_combination_model_params.pop("epochs")
            repetitions = param_combination_model_params.pop("repetitions")
            early_stopping = param_combination_model_params.pop("early_stopping")
            batch_size = param_combination_model_params.pop("batch_size")
            threshold_mode = param_combination_model_params.pop("threshold_mode")
            # Create the generator which yields the training sequences
            data_generator_sequence = DatasetCustodianSequence(self.dataset_custodian, "train", batch_size,
                                                               shuffle_on_end=True)

            reps_best_scores_data: List[ScoreData] = []
            rep_digits = int(math.log10(repetitions)) + 1
            # Multiple repetitions to average out variance
            for repetition in range(repetitions):
                # Initialize scores datas with "none" scores:
                reps_best_scores_data.append(ScoreData())
                early_stopping_best_main_scores_data = ScoreData()
                early_stopping_best_scores_data = ScoreData()
                # Create Tensorboard directory:
                tensorboard_dir_path = os.path.join(tensorboard_dir_root_path, str(repetition).zfill(rep_digits))
                create_dir_if_necessary(tensorboard_dir_path)

                # Create model:
                model = EagerModel.create(dataset_custodian=self.dataset_custodian,
                                          # input_shape=self.dataset_custodian.get_sample_shape(),
                                          output_neuron_count=len(self.dataset_custodian.get_labels()),
                                          **param_combination_model_params)
                if self.verbosity >= 2:
                    if model.built:
                        model.summary()
                    else:
                        print("Can not print model summary since model has not been built yet.")

                if tensorflow.executing_eagerly():
                    assert model.model_graph is not None

                # Create Tensorboard callback with model graph:
                tensorboard_callback = CustomTensorBoard(model_graph=model.model_graph, log_dir=tensorboard_dir_path,
                                                         write_graph=True)

                train_losses = []
                epochs_digits = int(math.log10(epochs)) + 1
                # Perform the training for given epochs
                for epoch in range(epochs):
                    epoch_start_time = time.time()
                    if self.verbosity >= 2:
                        # Print one line per epoch:
                        print("P.C.", param_comb_count_str + ", ", end="")
                        print(("Rep. {:0" + str(rep_digits) + "d}/{}, ").format(repetition + 1, repetitions),
                              end="")
                        print(("Ep. {:0" + str(epochs_digits) + "d}/{}: ").format(epoch + 1, epochs), end="")

                    # Silently ignore FutureWarning inside the context manager:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=FutureWarning)
                        # Generate a warning for NaN and/or (-)inf
                        with errstate(inf_or_nan=ExecutionCallback.WARN):
                            try:
                                # Train the model for one epoch. We use our own epoch loop instead of fit_generator's
                                # epochs > 1 to gain more control over pre- and post-epoch actions.
                                train_hist_obj = model.fit_generator(
                                    data_generator_sequence,
                                    epochs=1,
                                    # Verbosity: 0 ^= no ouput, 1 ^= progress bar, 2 ^= one line per epoch:
                                    verbose=1 if self.verbosity >= 4 else 0,
                                    callbacks=[tensorboard_callback],
                                    shuffle=False,  # Shuffling is done by DatasetCustodianSequence
                                    # Queue size: Number was heuristically chosen. Other numbers may work better in
                                    # other environments
                                    max_queue_size=10 * multiprocessing.cpu_count(),
                                    # Multiprocessing is deactivated since we rarely observed deadlocks
                                    use_multiprocessing=False,
                                    workers=multiprocessing.cpu_count())
                            except InvalidArgumentError as ia_ex:
                                # Error, which rarely occurs:
                                #  Input to reshape is a tensor with 65856 values, but the requested shape has 0
                                #  [[{{node training_52/Adam/gradients/value_repeated_52/Tile_grad/Reshape_1}}]]
                                print("InvalidArgumentError during model.fit_generator:", ia_ex)
                                # Skip this epoch
                                continue

                    train_loss = train_hist_obj.history["loss"][0]  # [0] to select the only epoch

                    # Display NaN gradients if any and if supported by optimizer (for now only own custom optimizer):
                    model_optimizer = model.optimizer.optimizer
                    if hasattr(model_optimizer, "print_and_reset_nan_infos"):
                        model_optimizer.print_and_reset_nan_infos(suffix=", ")

                    # Terminate on NaN:
                    if np.isnan(train_loss) or np.isinf(train_loss):
                        if self.verbosity >= 2:
                            print("Terminate On NaN: Invalid train loss", train_loss)
                        # Continue with next repetition:
                        break

                    # Employ the precision used for displaying the numbers
                    train_losses.append(float(SCORE_FORMAT_STRING.format(train_loss)))

                    # Determine train and validation (and test) scores:
                    train_and_val_scores = {}

                    # Whether softmax function should be applied to the models prediction before score calculation.
                    # Since restructure for eager execution, models do not have a soft max layer. Instead, the loss
                    # function contains the softmax function. Therefore, softmax should be applied to predictions
                    # for score calculation too.
                    # Models that were created before introduction of has_softmax_layer attribute, did not have a
                    # softmax layer, i.e., the default value is False.
                    apply_softmax = not getattr(model, "has_softmax_layer", False)

                    # Dont show warnings arising from the model only predicting one class (such that the other class
                    # has no support) like "UndefinedMetricWarning: Precision and F-score are ill-defined and being set
                    # to 0.0 in labels with no predicted samples."
                    with expect_warnings([UndefinedMetricWarning, RuntimeWarning]):
                        # Evaluate the model with training data
                        train_score, train_score_name, train_all_scores, train_summary \
                            = get_scores(model, self.dataset_custodian,
                                         split_name_or_indices="train",
                                         # Determine threshold depending on current parameter combination's value:
                                         threshold_to_use=threshold_mode,
                                         apply_softmax=apply_softmax,
                                         no_print=self.verbosity < 4)
                        # Add train loss and other train scores to collection of all scores:
                        train_all_scores["train_loss"] = train_loss
                        # Remember whether the threshold has been applied to softmax'ed outputs or to non-softmax'ed
                        #  outputs
                        train_all_scores["softmax_has_been_applied"] = apply_softmax
                        # Store train scores and summary text:
                        train_and_val_scores["train"] = {"scores": train_all_scores, "summary": train_summary}
                        if self.verbosity >= 2:
                            # Continue epoch output line:
                            print("Train loss", SCORE_FORMAT_STRING.format(train_loss) + ",",
                                  "Train score", train_score_name + ":", SCORE_FORMAT_STRING.format(train_score) + ", ",
                                  end="")

                        # Add train scores to tensorboard:
                        tensorboard_callback.add_scores(
                            GridSearch.filter_and_prefix_scores(train_all_scores, "train_", "float"))

                        if len(self.dataset_custodian.traintestval_split_indices["validation"]) > 0:
                            # Evaluate the model with validation data using the threshold determined using training data
                            val_score, val_score_name, val_all_scores, val_summary \
                                = get_scores(model, self.dataset_custodian,
                                             split_name_or_indices="validation",
                                             # Use the train threshold for validation
                                             threshold_to_use=train_all_scores["threshold"],
                                             apply_softmax=apply_softmax,
                                             no_print=self.verbosity < 4)
                            # Add val scores to collection of all scores:
                            train_and_val_scores["validation"] = {"scores": val_all_scores, "summary": val_summary}
                            if self.verbosity >= 2:
                                # Continue epoch output line:
                                print("Validation score", val_score_name + ":", SCORE_FORMAT_STRING.format(val_score) +
                                      ", " + "".join([sn + ": " + SCORE_FORMAT_STRING.format(sv) + ", "
                                                      for sn, sv in val_all_scores.items()
                                                      if type(sv).__name__.startswith("float")]), end="")
                                # Print non-float (i.e. string) threshold:
                                if isinstance(val_all_scores["threshold"], str):
                                    print("threshold:", val_all_scores["threshold"] + ", ", end="")
                                # Print confusion matrix on small class count
                                if len(self.dataset_custodian.get_label_ints()) < 5:
                                    print("conf_matrix:", str(val_all_scores["conf_matrix"].tolist()) + ", ", end="")

                            # Add scores to tensorboard:
                            tensorboard_callback.add_scores(
                                GridSearch.filter_and_prefix_scores(val_all_scores, "val_", "float"))
                        else:
                            if self.verbosity >= 2:
                                print("No validation data (using train scores), ", end="")
                            val_score_name = train_score_name
                            val_score = train_score

                    # Manage best scores:
                    model_file_name = SCORE_FORMAT_STRING.format(val_score) + "_" + val_score_name \
                                      + "_" + SCORE_FORMAT_STRING.format(train_loss) + "_train_loss" \
                                      + "_" + str(repetition) + "_" + str(epoch)

                    current_score_data = ScoreData(model_file_name, param_combination, repetition, epoch,
                                                   val_score_name, val_score, train_loss)

                    # Early stopping:
                    if early_stopping is not None:
                        es_patience = early_stopping["patience"]
                        es_train_loss_reduction_percentage = early_stopping["train_loss_reduction_percentage"]
                        es_train_loss_stop_threshold_patience = early_stopping["train_loss_stop_threshold_patience"]
                        es_train_loss_stop_threshold = early_stopping["train_loss_stop_threshold"]

                        # Update early stopping score using es_train_loss_reduction_percentage:
                        if current_score_data.greater_than(early_stopping_best_scores_data,
                                                           es_train_loss_reduction_percentage):
                            early_stopping_best_scores_data = current_score_data
                        # Update separate early stopping main score which only tracks main score improvements:
                        if current_score_data.main_score_greater_than(early_stopping_best_main_scores_data):
                            early_stopping_best_main_scores_data = current_score_data

                        # Check for main score progress during last es_train_loss_stop_threshold_patience epochs:
                        if epoch - early_stopping_best_main_scores_data.epoch > es_train_loss_stop_threshold_patience:
                            # No main score progress during last es_train_loss_stop_threshold_patience epochs.
                            # Check train loss threshold:
                            if current_score_data.train_loss < es_train_loss_stop_threshold:
                                if self.verbosity >= 2:
                                    print("Early Stopping: Main score did not improve for",
                                          es_train_loss_stop_threshold_patience,
                                          "epochs and train loss", current_score_data.train_loss,
                                          "is less than threshold", es_train_loss_stop_threshold)
                                # Continue with next repetition:
                                break
                        # Check for overall score progress improvement (including train loss improvements):
                        if epoch - early_stopping_best_scores_data.epoch > es_patience:
                            if self.verbosity >= 2:
                                print("Early Stopping: Score did not improve significantly for", es_patience, "epochs.")
                            # Continue with next repetition:
                            break

                    save_model = False
                    # Repetition wise best score
                    if current_score_data > reps_best_scores_data[repetition]:
                        # Update best score
                        reps_best_scores_data[repetition] = current_score_data
                        # Save model:
                        save_model = True

                    # Param-comb wise best score:
                    if current_score_data > param_comb_best_score_data:
                        # Update best score
                        param_comb_best_score_data = current_score_data
                        # Write best param comb score to file:
                        write_file_content(param_comb_best_score_file_path, param_comb_best_score_data.to_json())
                        # Save model:
                        save_model = True

                    if save_model:
                        # Save underlying model:
                        save_model_to_path(model, param_comb_dir_path, model_file_name, add_timestamp=False,
                                           verbose=False)

                        # Compute test scores if there is a test split:
                        if len(self.dataset_custodian.traintestval_split_indices["test"]) > 0:
                            with expect_warnings([UndefinedMetricWarning, RuntimeWarning]):
                                # Evaluate the model with test data using the best threshold determined using training
                                #  data
                                test_score, test_score_name, test_all_scores, test_summary \
                                    = get_scores(model, self.dataset_custodian,
                                                 split_name_or_indices="test",
                                                 # Use the train threshold for test
                                                 threshold_to_use=train_all_scores["threshold"],
                                                 apply_softmax=apply_softmax,
                                                 no_print=self.verbosity < 4)
                                # Add test scores to collection of all scores:
                                train_and_val_scores["test"] = {"scores": test_all_scores, "summary": test_summary}
                                # Add test scores to tensorboard:
                                tensorboard_callback.add_scores(
                                    GridSearch.filter_and_prefix_scores(test_all_scores, "test_", "float"))
                                # Print basic test score info:
                                if self.verbosity >= 2:
                                    print("Test score", test_score_name + ":",
                                          SCORE_FORMAT_STRING.format(test_score) + ", ",
                                          end="")
                        else:
                            if self.verbosity >= 2:
                                print("No test data, ", end="")

                        # Save scores:
                        write_file_content(os.path.join(param_comb_dir_path, model_file_name) + "_scores.json",
                                           to_json(train_and_val_scores))

                        print("Saved repetition-wise best model as", model_file_name + ", ", end="")
                    else:
                        if self.verbosity >= 2:
                            print("Best validation score so far:", param_comb_best_score_data.compact_repr() + ", ",
                                  end="")

                    # Overall best score:
                    if current_score_data > overall_best_score_data:
                        # Update overall best score:
                        overall_best_score_data = current_score_data

                        # ... and write it to file:
                        write_file_content(overall_best_score_file_path, overall_best_score_data.to_json())

                    # Epoch time:
                    if self.verbosity >= 2:
                        print(SCORE_FORMAT_STRING.format(time.time() - epoch_start_time), "s")
                # End of one repetition / one training. Tell CustomTensorBoard about this:
                tensorboard_callback.call_super_on_train_end()
            # Finished parameter combination: Print statistics and best model:
            if self.verbosity >= 1:
                print("Finished parameter combination in", time.time() - param_comb_start_time, "s with best score",
                      param_comb_best_score_data.compact_repr(),
                      "(parameter combination: " + str(param_combination) + ")")
                # Statistics for score over repetitions:
                repetition_best_main_scores = [rep_best_score.main_score for rep_best_score in reps_best_scores_data]
                repetition_score_summary = {"best_scores_data": reps_best_scores_data,
                                            "best_main_scores": repetition_best_main_scores,
                                            "mean": np.mean(repetition_best_main_scores),
                                            "median": np.median(repetition_best_main_scores),
                                            "std": np.std(repetition_best_main_scores),
                                            "var": np.var(repetition_best_main_scores)}
                print("Best validation scores by repetition:", repetition_score_summary)
                write_file_content(os.path.join(param_comb_dir_path, REPETITIONS_SUMMARY_JSON_NAME),
                                   to_json(repetition_score_summary))
                # Elapsed time
                print(time.time() - overall_start_time, "s elapsed so far.")

            # Print current best score:
            if self.verbosity >= 1:
                self.print_overall_best_score_data(overall_best_score_data, "Current ")

        # Finished all parameter combinations
        print("Completed grid search in", int(time.time() - overall_start_time), "s.")
        self.print_overall_best_score_data(overall_best_score_data)
        self.generate_gridsearch_summary_csv(gridsearch_root_path, is_path=True)
        return overall_best_score_data

    @staticmethod
    def get_model_scores_with_test_scores(model_dir_path):
        """
        Returns the scores of the model specified by the given path. If no test scores are present, they are calculated.
        This method exists, since for old models, test scores were not calculated upon save.
        In addition, in the past not every repetition-wise best model has been saved. Thus, model_dir_path may be
        non-existing and scores can not be returned.
        :param model_dir_path:
        :return:
        """
        model_scores_file_path = model_dir_path + "_scores.json"
        try:
            # Try to read the model's scores:
            model_scores = from_json(read_file_content(model_scores_file_path))
        except IOError:
            # In the past, only a model that is better than all other models of the same param comb was saved.
            # Repetition-wise best models may not have been better than the best model of a previous
            # repetition and thus have not been saved. Nothing that could be done:
            return None
        test_variant = "test"  # "test_argmax"
        # Check whether test scores were already calculated:
        if model_scores.get(test_variant, None) is None:
            try:
                # Load environment for calculation of test score:
                gridsearch_checkpoint = GridSearchCheckpointInfo(model_dir_path)
                # Evaluate test scores:
                test_score, test_score_name, test_all_scores, test_summary \
                    = get_scores(gridsearch_checkpoint.model, gridsearch_checkpoint.dataset_custodian, "test",
                                 threshold_to_use="argmax" if test_variant == "test_argmax" else
                                 gridsearch_checkpoint.model_scores["train"]["scores"]["threshold"],
                                 apply_softmax=False if test_variant == "test_argmax" else
                                 gridsearch_checkpoint.model_scores["train"]["scores"].get("softmax_has_been_applied",
                                                                                           False),
                                 no_print=True)
                # Add test scores to collection of all scores:
                model_scores[test_variant] = {"scores": test_all_scores, "summary": test_summary}
                # Save scores:
                write_file_content(model_scores_file_path, to_json(model_scores))
                print(" - Updated model", test_variant, "score file with test scores. Model score file:",
                      model_scores_file_path)
                # Read the scores back in by recursevily call this method (because otherwise there is a new line in
                # conf_matrix when the scores are written to csv)
                model_scores = GridSearch.get_model_scores_with_test_scores(model_dir_path)
                # Remove temporary model files:
                gridsearch_checkpoint.model.temp_model_dir_path_cleanup()
            except Exception as err:
                print(" - Unable to calculate", test_variant, "scores for model", model_dir_path, "because of error",
                      err)
                traceback.print_exc()
        return model_scores

    @staticmethod
    def generate_gridsearch_summary_csv(gridsearch_root_path_or_dataset_dir_name: str, is_path=False,
                                        write_csv=True):
        """
        Returns tabular information about a grid search's resulting models and scores. The information can be written to
         a csv file.
        :param gridsearch_root_path_or_dataset_dir_name: Both an absolute path to a grid search directory or the name of
         a dataset custodian are accepted. is_path must be set to true for the former and to false for the latter.
        :param is_path: See above.
        :param write_csv: Whether to write the information to a csv file in the grid search root directory.
        :return:
        """

        def add_model_scores(model_scores, merged_data, score_key, key_prefix="",
                             key_suffix=""):
            main_test_score = None
            test_variant = "test"  # "test_argmax"
            try:
                main_test_score \
                    = merged_data[key_prefix + test_variant + "_" + score_key + key_suffix] = \
                    model_scores[test_variant]["scores"][score_key]
                merged_data[key_prefix + test_variant + "_conf_matr" + key_suffix] = \
                    model_scores[test_variant]["scores"]["conf_matrix"]
            except KeyError as ke:
                if test_variant in str(ke):
                    # There are cases where no test score is available (e.g. in case of Memnet paper split)
                    print(" - No", test_variant, "score in score data")
                else:
                    raise

            merged_data[key_prefix + "val_" + score_key + key_suffix] = model_scores["validation"]["scores"][score_key]
            merged_data[key_prefix + "val_conf_matr" + key_suffix] = model_scores["validation"]["scores"]["conf_matrix"]

            merged_data[key_prefix + "train_" + score_key + key_suffix] = model_scores["train"]["scores"][score_key]
            merged_data[key_prefix + "train_conf_matr" + key_suffix] = model_scores["train"]["scores"]["conf_matrix"]
            return main_test_score

        gridsearch_root_path_or_dataset_dir_name = str(gridsearch_root_path_or_dataset_dir_name)
        if not is_path:
            gridsearch_path = os.path.join(PREPARED_DATASET_PATH_ROOT,
                                           gridsearch_root_path_or_dataset_dir_name,
                                           "GridSearch")
        else:
            gridsearch_path = gridsearch_root_path_or_dataset_dir_name
        is_incomplete = False
        param_combs_data = []
        for abs_path, file_or_dir_name in get_non_rec_dir_content_names(gridsearch_path):
            if os.path.isdir(abs_path):
                print("GridSearch Directory", file_or_dir_name, "...")
                try_counter = 5
                while True:
                    try:
                        params = from_json(read_file_content(os.path.join(abs_path, "params.json")))
                        rep_score_summary = None
                        try:
                            rep_score_summary = from_json(
                                read_file_content(os.path.join(abs_path, "repetition_scores_summary.json")))
                        except IOError:
                            print(" - Repetition score summary file not found. Either param comb has not been finished"
                                  + " (yet) or results are legacy. ", end="")
                            is_incomplete = True

                        # Copy data before modification below:
                        merged_data = dict(params)
                        merged_data["name"] = file_or_dir_name
                        if rep_score_summary is not None:
                            best_main_test_scores = []
                            best_main_val_scores = rep_score_summary["best_main_scores"]
                            main_score_name = None
                            for i, best_main_val_score in enumerate(best_main_val_scores):
                                if main_score_name is None:
                                    main_score_name = rep_score_summary["best_scores_data"][i]["main_score_name"]
                                else:
                                    assert main_score_name == rep_score_summary["best_scores_data"][i][
                                        "main_score_name"]
                                merged_data["val_" + main_score_name + "_rep" + str(i)] = best_main_val_score
                                model_name = rep_score_summary["best_scores_data"][i]["model_name"]
                                model_scores = GridSearch.get_model_scores_with_test_scores(os.path.join(abs_path,
                                                                                                         model_name))
                                if model_scores is not None:
                                    # Add best model's scores:
                                    score_keys = ["f1_macro_score", "f1_weighted_score", "aroc_bin_score", "accuracy",
                                                  "balanced_accuracy", "train_loss", "mcc"]
                                    # Score to consider for best_test_... result:
                                    main_score_key = "f1_macro_score" if main_score_name == "F1" else "aroc_bin_score"
                                    assert main_score_key in score_keys, (score_keys, main_score_key)
                                    for score_key in score_keys:
                                        try:
                                            main_test_score = add_model_scores(model_scores, merged_data, score_key,
                                                                               key_suffix="_rep" + str(i))
                                            if score_key == main_score_key:
                                                best_main_test_scores.append(main_test_score)
                                        except KeyError as ke:
                                            print("KeyError for", score_key, ke)
                                            continue
                                else:
                                    print(" - Repetition", i + 1, "model", model_name,
                                          "had not been saved. Skipping it.")
                                    # This can not happen in the first repetition:
                                    # assert i > 0, i
                                merged_data["train_loss_rep" + str(i)] = rep_score_summary["best_scores_data"][i][
                                    "train_loss"]
                                merged_data["epoch_rep" + str(i)] = rep_score_summary["best_scores_data"][i]["epoch"]

                            best_val_rep_nr, best_main_val_score = argmax(best_main_val_scores, True)
                            merged_data["best_val_" + main_score_name] = best_main_val_score
                            merged_data["best_val_" + main_score_name + "model_path"] = os.path.join(abs_path,
                                                                                                     rep_score_summary[
                                                                                                         "best_scores_data"][
                                                                                                         best_val_rep_nr][
                                                                                                         "model_name"])

                            if len(best_main_test_scores) == len(best_main_val_scores):
                                # Find the best validation model's corresponding test score:
                                corresponding_test_score = best_main_test_scores[best_val_rep_nr]
                                if corresponding_test_score is not None:
                                    merged_data["best_test_" + main_score_name] = corresponding_test_score

                            if len(best_main_test_scores) == len(best_main_val_scores):
                                # Compute test statistics:
                                merged_data = {**merged_data,
                                               "test_mean_reps": np.mean(best_main_test_scores),
                                               "test_median_reps": np.median(best_main_test_scores),
                                               "test_std_reps": np.std(best_main_test_scores),
                                               "test_var_reps": np.var(best_main_test_scores)
                                               }

                            # Add validation statistics:
                            for key, value in rep_score_summary.items():
                                if isinstance(value, (float, np.float)):
                                    merged_data["val_" + key + "_reps"] = value
                        else:
                            # Legacy results:
                            try:
                                best_score_data = from_json(
                                    read_file_content(os.path.join(abs_path, "best_score.json")))
                            except IOError:
                                print("Best score data not available. Skipping directory.")
                                break

                            if best_score_data.get("main_score_name", None) is not None:
                                print("Using simple best score data ...")
                                merged_data["best_val_" + best_score_data["main_score_name"]] \
                                    = best_score_data["main_score"]
                                merged_data["best_train_loss"] = best_score_data["train_loss"]
                                # Try to get other scores from best model data
                                model_path = os.path.join(abs_path, best_score_data["model_name"])
                                best_model_score_data = GridSearch.get_model_scores_with_test_scores(model_path)
                                if best_model_score_data is not None:
                                    add_model_scores(best_model_score_data, merged_data, "f1_weighted_score",
                                                     key_prefix="best_")
                                else:
                                    print("Best model score data not available")
                            elif best_score_data.get("score_name", None) is not None:
                                print("Using legacy results ...")
                                merged_data["best_val_" + best_score_data["score_name"]] \
                                    = best_score_data["score"]
                            else:
                                assert False, ("Unknown best score data", best_score_data)
                        param_combs_data.append(merged_data)
                        break
                    except IOError as io_err:
                        print("Unexpected IO-Error:", io_err)
                        if try_counter == 0:
                            print(" - Aborting.")
                        else:
                            print(" - Trying again shortly ...")
                            sleep(1)
                    finally:
                        try_counter -= 1
                print(" - Done")
        print("Got", len(param_combs_data), "param combs.")
        csv_file_path = None
        if len(param_combs_data) > 0 and write_csv:
            column_header = []
            for param_comb_data in param_combs_data:
                for key in param_comb_data.keys():
                    if key not in column_header:
                        column_header.append(key)
            csv_file_path = os.path.join(gridsearch_path, "summary" + ("_incomplete" if is_incomplete else "") + ".csv")
            with open(csv_file_path, "w") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(column_header)
                for param_comb_data in param_combs_data:
                    writer.writerow([param_comb_data.get(key, "<NA>") for key in column_header])
            print("Wrote summary of", csv_file_path)
        return param_combs_data, csv_file_path
