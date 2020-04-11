"""
Creation of TensorFlow eager models
"""

from functools import partial

from tensorflow import Dimension
from tensorflow.python.keras.layers import Embedding, Dense, concatenate, Lambda, TimeDistributed, Masking, \
    Bidirectional
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.ops.losses.losses_impl import Reduction

from DataPreparation import *
from NaluLayer import NALU


def create_slice_subgraph(model, feat_vec_index_start, feat_vec_index_end, name: str, force_slice_range=False):
    # assert feat_vec_index_start < input_shape[1], "index " + str(
    #     feat_vec_index_start) + " must be smaller than feat_vec length"
    if feat_vec_index_start == feat_vec_index_end - 1 and not force_slice_range:
        # Return tensor for that index (==> dimensionality is reduced by one)
        slicer_lambda = lambda x: x[:, :, feat_vec_index_start]
    else:
        # Return range of tensor for that indices (==> dimensionality stays the same )
        slicer_lambda = lambda x: x[:, :, feat_vec_index_start:feat_vec_index_end]
    lambda_layer = Lambda(slicer_lambda, name="slice_" + name)
    setattr(model, lambda_layer.name, lambda_layer)
    # print("Created slicer for", feat_vec_index_start, feat_vec_index_end, name, force_slice_range)


class ArchitectureKind(Enum):
    BASIC = 1
    EMBED_NALU_NON_REK = 2
    EMBED_NALU_REK_ALL_IN_MAIN_LSTM = 3


class EagerModel(tfkeras.Model):
    def __init__(self,
                 dataset_custodian: DatasetCustodian,
                 output_neuron_count,
                 net_type: str,
                 embedding_vector_sizes,
                 main_lstm_neuron_count,
                 nalu_neuron_count,
                 nalu_nac_only,
                 dense_neuron_count,
                 activation):
        super().__init__()
        # Check neuron counts:
        assert abs(main_lstm_neuron_count) > 0, "main_lstm_neuron_count must be positive (main lstm neuron count) or" \
                                                + "negative (for non-lstm mode where it is the negative fixed sequence" \
                                                + "length)" + str(main_lstm_neuron_count)
        self.main_lstm_neuron_count = main_lstm_neuron_count
        assert dense_neuron_count >= 0, dense_neuron_count
        self.dense_neuron_count = dense_neuron_count
        assert output_neuron_count >= 1, output_neuron_count
        # nalu_neuron_count can be zero to specify that NALU layers should not be added to the network
        assert nalu_neuron_count >= 0, nalu_neuron_count
        self.nalu_neuron_count = nalu_neuron_count

        # Feat vec infos:
        self.kind_numbers_feat_vec_component_info \
            = dataset_custodian.frontend.get_feat_vec_component_info("kind_numbers")
        self.feat_numbers_feat_vec_component_info \
            = dataset_custodian.frontend.get_feat_vec_component_info("feat_numbers")
        basic_feat_vec_list = dataset_custodian.frontend.get_feat_vec_component_info("basic_feat_vec")
        assert len(basic_feat_vec_list) == 1
        self.basic_feat_vec_component_info = basic_feat_vec_list[0]

        # Architecture to create:
        if net_type.endswith("_bidirect"):
            old_CuDNNLSTM = globals().get("CuDNNLSTM")

            def bidirectional_lstm(**kwargs_lstm):
                layer_name = kwargs_lstm.pop("name")
                return Bidirectional(old_CuDNNLSTM(**kwargs_lstm), name=layer_name, merge_mode="concat")

            CuDNNLSTM = bidirectional_lstm
            net_type = net_type.replace("_bidirect", "")
        else:
            CuDNNLSTM = globals().get("CuDNNLSTM")

        if net_type == "basic":
            self.architecture_kind = ArchitectureKind.BASIC
        elif net_type == "embedding_alt":
            self.architecture_kind = ArchitectureKind.EMBED_NALU_NON_REK
        elif net_type == "embedding":
            self.architecture_kind = ArchitectureKind.EMBED_NALU_REK_ALL_IN_MAIN_LSTM
        else:
            assert False, net_type

        # Attribute that will hold the model's graph
        self.model_graph = None

        # Init of layer weights should be done after creating the model. Weight-init-functions can be added during
        # creating and they will be called after model has been built.
        self.init_funcs = []

        # Attribute that will hold the model's embeddings layer's results after invoking its call method. This should
        # only be used for visualization and not during training because it will result in memory leakage of the tensors
        # stores in embedding_results
        self.embedding_results = None

        # ### Actually start creating the model ###
        self.masking_layer = Masking(mask_value=0.0)
        if self.architecture_kind == ArchitectureKind.BASIC:
            # Add single slice layer:
            create_slice_subgraph(self, self.basic_feat_vec_component_info.index_start,
                                  self.basic_feat_vec_component_info.index_end,
                                  self.basic_feat_vec_component_info.name, force_slice_range=True)
        else:
            # Add slice layers:
            for input_info in self.kind_numbers_feat_vec_component_info + self.feat_numbers_feat_vec_component_info:
                # if input_matching_info.name[-2] != "_":
                create_slice_subgraph(self, input_info.index_start, input_info.index_end,
                                      input_info.name,
                                      force_slice_range=input_info in self.feat_numbers_feat_vec_component_info)

            # Add embeddings:
            for input_kind_info in self.kind_numbers_feat_vec_component_info:
                vocab_size = input_kind_info.kind_count
                embedding_output_dim = embedding_vector_sizes[input_kind_info.name]
                # Create embedding layer:
                # Embedding returns null-vector vor integers >= vocabulary size!
                embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_output_dim,
                                            name="embedding_" + input_kind_info.name)
                # Check whether there are initialization weights available:
                list_of_embedding_init_weights = \
                    dataset_custodian.frontend.get_embedding_init_weights(input_kind_info, embedding_output_dim)
                if list_of_embedding_init_weights is not None:
                    assert len(list_of_embedding_init_weights) == 1, list_of_embedding_init_weights
                    # Use the output_dim from the initialization:
                    assert embedding_output_dim == list_of_embedding_init_weights[0].shape[1], \
                        (embedding_output_dim, list_of_embedding_init_weights)
                    # print("Set", embedding_name, "embedding's output to", embedding_vector_size, "from init weights")
                    # Register init function (list_of_embedding_init_weights=list_of_embedding_init_weights to capture
                    # current value of list_of_embedding_init_weights):
                    self.init_funcs.append(lambda embedding_layer=embedding_layer,
                                                  list_of_embedding_init_weights=list_of_embedding_init_weights:
                                           (embedding_layer.set_weights(list_of_embedding_init_weights),
                                            "Initialized weights of " + embedding_layer.name + " with provided weights.")
                                           )
                setattr(self, embedding_layer.name, embedding_layer)

            # Add Sub-LSTM, -NALU
            for sub_layer_name_suffix in [info.name for info in self.kind_numbers_feat_vec_component_info] \
                                         + ["member_num", "concatenate_values_num", "type_qual_name_num",
                                            "concatenate_ness_num"]:
                if "kind" not in sub_layer_name_suffix:
                    if self.nalu_neuron_count > 0:
                        # LSTM should learn to extract numbers and NALU to do calculations. As there are more combinations to do
                        # arithmetic with given numbers than given number count, NALU layer should have more neurons.
                        sub_lstm_layer = CuDNNLSTM(units=int(self.nalu_neuron_count / 2) + 1,
                                                   name="lstm_sub_" + sub_layer_name_suffix,
                                                   return_sequences=self.architecture_kind == ArchitectureKind.EMBED_NALU_REK_ALL_IN_MAIN_LSTM)
                        setattr(self, sub_lstm_layer.name, sub_lstm_layer)

                        if self.architecture_kind == ArchitectureKind.EMBED_NALU_NON_REK:
                            sub_nalu_layer = NALU(units=self.nalu_neuron_count, nac_only=nalu_nac_only,
                                                  name="nalu_sub_" + sub_layer_name_suffix)
                        elif self.architecture_kind == ArchitectureKind.EMBED_NALU_REK_ALL_IN_MAIN_LSTM:
                            sub_nalu_layer = TimeDistributed(NALU(units=self.nalu_neuron_count,
                                                                  nac_only=nalu_nac_only),
                                                             name="nalu_sub_" + sub_layer_name_suffix)
                        else:
                            assert False
                        setattr(self, sub_nalu_layer.name, sub_nalu_layer)
        # Main LSTM:
        if self.main_lstm_neuron_count > 0:
            self.lstm_main_layer = CuDNNLSTM(units=self.main_lstm_neuron_count, name="lstm_main")
        # Add main NALU for basic type if neuron count > 0
        if self.architecture_kind == ArchitectureKind.BASIC and self.nalu_neuron_count > 0:
            self.nalu_main_layer = NALU(units=self.nalu_neuron_count, name="nalu_main", nac_only=nalu_nac_only)
        # Fully connected layer:
        if self.dense_neuron_count > 0:
            self.dense_fc_layer = Dense(units=dense_neuron_count, activation=activation, name="dense")
        # Output layer (no activation)
        self.dense_output_layer = Dense(units=output_neuron_count, activation=None, name="dense_output")

        # Set whether this model has a softmax layer:
        self.has_softmax_layer = False

    def build(self, input_shape):
        super().build(input_shape)

        # Call init functions:
        for init_func in self.init_funcs:
            print(init_func())

    # @tensorflow.contrib.eager.defun
    def call(self, inputs, training=None, mask=None):
        # print("sub_layer_name_suffix", tensorflow.executing_eagerly(), "mask", mask)
        # print(type(inputs))
        if isinstance(inputs, list):
            assert len(inputs) == 1
            input = inputs[0]
        else:
            input = inputs
        # input_shape = inputs.shape
        # print("input", input.shape)

        # Masking:
        input = self.masking_layer(input)

        sub_results_for_dense = []
        # Determine the input for main lstm layer:
        if self.architecture_kind == ArchitectureKind.BASIC:
            # Just use the basic feat vec's slice:
            for_lstm_main = self.slice_basic_feat_vec(input)
        else:
            # Connect sub lstm and sub nalu and concatenate them:
            value_slice_results = []
            ness_slice_results = []
            other_slice_results = {}
            for input_matching_info in self.kind_numbers_feat_vec_component_info + self.feat_numbers_feat_vec_component_info:
                slice_name = "slice_" + input_matching_info.name
                sliced_result = getattr(self, slice_name)(input)
                if "arrsize_num" in input_matching_info.name or "value_num" in input_matching_info.name:
                    value_slice_results.append(sliced_result)
                elif "ness_num" in input_matching_info.name:
                    ness_slice_results.append(sliced_result)
                else:
                    other_slice_results[slice_name] = sliced_result

            sub_results_for_main_lstm = []
            for sub_layer_name_suffix in [info.name for info in self.kind_numbers_feat_vec_component_info] \
                                         + ["member_num", "concatenate_values_num", "type_qual_name_num",
                                            "concatenate_ness_num"]:
                if sub_layer_name_suffix == "concatenate_values_num":
                    sliced_or_embed_result = concatenate(value_slice_results, name="concatenate_values_num")
                elif sub_layer_name_suffix == "concatenate_ness_num":
                    sliced_or_embed_result = concatenate(ness_slice_results, name="concatenate_ness_num")
                else:
                    sliced_or_embed_result = other_slice_results["slice_" + sub_layer_name_suffix]

                if "kind" in sub_layer_name_suffix:
                    sliced_or_embed_result = getattr(self, "embedding_" + sub_layer_name_suffix)(sliced_or_embed_result)
                    # The following leads to memory leaks:
                    if self.embedding_results is not None:
                        self.embedding_results.append(sliced_or_embed_result)
                # print(sliced_or_embed_result.name, sliced_or_embed_result.shape)
                sliced_or_embed_or_nalu_result = sliced_or_embed_result
                # lstm_or_nalu_result = getattr(self, "sub_lstm_" + sub_layer_name_suffix)(sliced_or_embed_result)
                # print(lstm_or_nalu_result.name, lstm_or_nalu_result.shape)
                if "kind" not in sub_layer_name_suffix:
                    sub_lstm_layer = getattr(self, "lstm_sub_" + sub_layer_name_suffix, None)
                    if sub_lstm_layer is not None:
                        # Connect sub lstm:
                        sliced_or_embed_or_nalu_result = sub_lstm_layer(sliced_or_embed_or_nalu_result)

                        # Connect sub nalu:
                        sub_nalu_layer = getattr(self, "nalu_sub_" + sub_layer_name_suffix)
                        sliced_or_embed_or_nalu_result = sub_nalu_layer(sliced_or_embed_or_nalu_result)
                    else:
                        # sub lstm and nalu are only allowed to be absent if it has been requested through
                        # nalu neuron count value of zero:
                        assert self.nalu_neuron_count == 0, self.nalu_neuron_count

                    # Concatenate depending on architecture. With nalu_neuron_count == 0, both are the same:
                    if self.architecture_kind == ArchitectureKind.EMBED_NALU_REK_ALL_IN_MAIN_LSTM \
                            or self.nalu_neuron_count == 0:
                        # With "rekurrent" nalu concat nalu output and input it to main lstm:
                        sub_results_for_main_lstm.append(sliced_or_embed_or_nalu_result)
                    elif self.architecture_kind == ArchitectureKind.EMBED_NALU_NON_REK:
                        # With non-rekurrent nalu concat nalu output and input it to dense
                        sub_results_for_dense.append(sliced_or_embed_or_nalu_result)
                    else:
                        assert False

                    # print(lstm_or_nalu_result.name, lstm_or_nalu_result.shape)
                else:
                    sub_results_for_main_lstm.append(sliced_or_embed_or_nalu_result)

            for_lstm_main = concatenate(sub_results_for_main_lstm, name="concatenate_for_lstm_main")

        # Connect main lstm layer:
        lstm_main_layer = getattr(self, "lstm_main_layer", None)
        if lstm_main_layer is not None:
            lstm_main_result = lstm_main_layer(for_lstm_main)
        else:
            assert self.main_lstm_neuron_count <= 0, self.main_lstm_neuron_count
            # Reshape data to non-sequence format. Create new shape as tuple of ints according to for_lstm_main's shape:
            new_shape = [for_lstm_main.shape[0], for_lstm_main.shape[1] * for_lstm_main.shape[2]]
            # The last dimension must be fixed! Therefore, this only works for fixed sequences length. In this
            # case the sequence length must be provided as negative lstm neuron count.
            if str(new_shape[0]) == "?":
                new_shape = [Dimension(None), -self.main_lstm_neuron_count]
            # Replace Dimensions with ints and replace placeholder in shape (i.e. "?" / Dimension(None)) with -1:
            for shape_part_index in range(len(new_shape)):
                if str(new_shape[shape_part_index]) == "?":
                    new_shape[shape_part_index] = -1
                else:
                    new_shape[shape_part_index] = int(new_shape[shape_part_index])
            lstm_main_result = tensorflow.reshape(for_lstm_main, new_shape)
        # print("Reshaped from", for_lstm_main.shape, "to", lstm_main_result.shape)

        # Determine the input for first dense layer:
        if self.architecture_kind == ArchitectureKind.BASIC:
            # Main nalu
            nalu_main_layer = getattr(self, "nalu_main_layer", None)
            if nalu_main_layer is not None:
                for_dense = nalu_main_layer(lstm_main_result)
            else:
                assert self.nalu_neuron_count == 0, self.nalu_neuron_count
                for_dense = lstm_main_result
        elif self.architecture_kind == ArchitectureKind.EMBED_NALU_REK_ALL_IN_MAIN_LSTM or self.nalu_neuron_count == 0:
            assert len(sub_results_for_dense) == 0
            for_dense = lstm_main_result
        elif self.architecture_kind == ArchitectureKind.EMBED_NALU_NON_REK:
            assert len(sub_results_for_dense) > 0, sub_results_for_dense
            for_dense = concatenate([lstm_main_result] + sub_results_for_dense, name="concatenate_for_dense")
        else:
            assert False

        # Main Dense
        dense_fc_layer = getattr(self, "dense_fc_layer", None)
        if dense_fc_layer is not None:
            dense_result = dense_fc_layer(for_dense)
        else:
            assert self.dense_neuron_count == 0, self.dense_neuron_count
            dense_result = for_dense

        dense_output_result = self.dense_output_layer(dense_result)

        if not tensorflow.executing_eagerly() and self.model_graph is None:
            # We are currently in non eager mode and have a graph which can be saved:
            self.model_graph = dense_output_result.graph
        return dense_output_result


def create(**kwargs):
    initial_kwargs = dict(kwargs)
    dataset_custodian: DatasetCustodian = kwargs.get("dataset_custodian")
    # Check whether a pretrained model should be loaded:
    pretrained_model_path = kwargs.pop("pretrained_model_path", None)
    if pretrained_model_path is not None:
        model = load_model_from_path(pretrained_model_path)
        # Make sure the loss function returns single losses of each class such that they can be weighted:
        if not isinstance(model.loss, partial):
            model.loss = partial(model.loss, reduction=Reduction.NONE)
        return model
    else:
        # Remove non-model-related arguments from kwargs before passing them to the model constructor:
        optimizer = kwargs.pop("optimizer")
        learn_rate = kwargs.pop("learn_rate", None)
        # Loss (https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow/47034889)
        # (for multilabel: sigmoid in last layer and binary_crossentropy):
        loss = kwargs.pop("loss", "softmax_cross_entropy")

        # Actually create the model object:
        model = EagerModel(**kwargs)
        # Asoociate model creation function with the model to allow saving and loading it:
        model.create_model_func_kwargs = initial_kwargs
        create_model_module = pickle.source.getmodule(create)
        if create_model_module is not None:
            model.create_model_func_source = pickle.source.getsource(create_model_module)
        else:
            model.create_model_func_source = None
            print("Warning: Unable to get the create function's module. Has the model been loaded from disk?")

        # Create optimizer (dynamically create object):
        # optimizer_class = getattr(tfkeras.optimizers, optimizer)
        optimizer_class = getattr(tensorflow.train, optimizer)

        # Custom Optimizer that inherits from chosen optimizer and that checks for NaN gradients
        class NanAwareOptimizer(optimizer_class):

            nan_infos = None

            def __init__(self, **optimizer_init_kwargs):
                self.reset_nan_infos()
                super().__init__(**optimizer_init_kwargs)

            def reset_nan_infos(self):
                self.nan_infos = []

            def print_and_reset_nan_infos(self, prefix="", suffix="\n"):
                # Do nothing when there where no NaNs:
                if len(self.nan_infos) == 0:
                    self.reset_nan_infos()
                    return

                var_names = set()
                nan_count_sum = 0
                overall_count_sum = 0
                for var_name, nan_count, overall_count in self.nan_infos:
                    var_names.add(var_name)
                    nan_count_sum += nan_count
                    overall_count_sum += overall_count
                nan_count_avg = nan_count_sum / len(self.nan_infos)
                overall_count_avg = overall_count_sum / len(self.nan_infos)
                print(prefix,
                      "NaN gradients: avg. {}/{} ({:.2f}%) for {{{}}}".
                      format(nan_count_avg,
                             overall_count_avg,
                             nan_count_avg / overall_count_avg * 100,
                             ", ".join(list(var_names))),
                      suffix,
                      end="", sep="")
                self.reset_nan_infos()

            def apply_gradients(self, grads_and_vars, global_step=None, name=None):
                # (Shallow) copy grads_and_vars iterator such that it can be iterated here and passed to
                # super.apply_gradients (otherwise the iteration here would consume it, such that it becomes the empty
                # tuple). Not necessary anymore as a new iterable is created and passed to super.apply_gradients
                # grads_and_vars, grads_and_vars_copy = itertools.tee(grads_and_vars)

                # Create new grads and vars iterable where all NaNs in grad are replaced:
                new_grads_and_vars = []
                for grad, var in grads_and_vars:
                    nan_count = tensorflow.count_nonzero(tensorflow.is_nan(grad))
                    if nan_count > 0:
                        overall_count = tensorflow.size(grad, out_type=tensorflow.int64)
                        self.nan_infos.append((var.name, nan_count, overall_count))
                        # print("\nWarning:", var.name, "gradient values that are NaN: {}/{} ({:.2f}%)".
                        #       format(nan_count, overall_count, nan_count / overall_count * 100), global_step, name)
                        # Replace NaNs with zeros:
                        grad = tensorflow.where(tensorflow.is_nan(grad), tensorflow.ones_like(grad) * 0, grad)
                    new_grads_and_vars.append((grad, var))
                # Perform actual gradient application with non-NaN gradients:
                super().apply_gradients(new_grads_and_vars, global_step, name)

        optimizer_class = NanAwareOptimizer
        if learn_rate is not None:
            optimizer_object = optimizer_class(learning_rate=learn_rate)
        else:
            optimizer_object = optimizer_class()
        # Create loss:
        loss_func = getattr(tensorflow.losses, loss)
        # Make sure the loss function returns single losses of each class such that they can be weighted:
        loss_func = partial(loss_func, reduction=Reduction.NONE)
        if ENVIRONMENT.gpu_count > 1:
            # Make model using multiple gpus:
            model = multi_gpu_model(model, gpus=ENVIRONMENT.gpu_count)
        # Compile
        model.compile(loss=loss_func, optimizer=optimizer_object, run_eagerly=True)
        # Build the model (this will invoke model.call in non-eager mode). If model.build is not called explicitly here,
        # it will be called by model.fit_generator implicitly when the first batch is about to be feed to the network.
        model.build((None, None, dataset_custodian.get_sample_shape()[1]))
        return model
