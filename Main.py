# -*- coding: utf-8 -*-
import pprint
from enum import auto, IntFlag

import CweTaxonomy
import SummarizeJulietExperiments
from DataPreparation import *
from Training import GridSearch


class DatasetMode(IntFlag):
    # Main dataset modes:
    JULIET = auto()
    MEMNET = auto()
    ARRDECLACCESS = auto()
    LOAD = auto()

    # Sub dataset modes:
    LEVEL1 = auto()
    ALL = auto()
    SELECTED = auto()
    SEPARATE = auto()
    MERGE = auto()

    # Sub sub dataset modes:
    LEVEL2 = auto()
    EXISTING = auto()
    NEW = auto()


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # --- Begin of configuration ---------------------------------------------------------------------------------------

    # There will be one training for each prepared dataset and for each combination of training-related parameter
    #  values.

    # First, specify the dataset mode by uncommenting and adjusting the desired assignment. The dataset mode
    #  specifies which dataset(s) are prepared and trained on. The training-related parameters can be specified
    #  afterwards and are identical for each prepared dataset.

    # --- Juliet dataset ---
    #  If the last element in the mode array is a dictionary, its items overwrite the items in the Juliet base config
    #  (see implementation below).

    # For each Juliet CWE, prepare and train on one separate dataset
    # mode = [DatasetMode.JULIET, DatasetMode.SEPARATE]

    # For each SELECTED Juliet CWE, prepare and train on one separate dataset
    mode = [DatasetMode.JULIET, DatasetMode.SELECTED, ["CWE122_Heap_Based_Buffer_Overflow"]]
    # mode = [DatasetMode.JULIET, DatasetMode.SELECTED, ["CWE427_Uncontrolled_Search_Path_Element", "CWE367_TOC_TOU",
    #                                                    "CWE242_Use_of_Inherently_Dangerous_Function"]]

    # Prepare and train on one multi-class dataset that contains ALL Juliet CWEs
    # mode = [DatasetMode.JULIET, DatasetMode.ALL]

    # For each CWE taxonomy group, prepare and train on one multi-class dataset that contains all CWEs from
    #  the respective group. Each of these taxonomy-group datasets is MERGEd from either already EXISTING (i.e., already
    #  prepared and saved) separate-CWE datasets or from NEWly prepared datasets. Using existing datasets reduces the
    #  preparation time.
    # mode = [DatasetMode.JULIET, DatasetMode.MERGE, DatasetMode.EXISTING,
    #         ["/home/schroeder/MasterThesisData/PreprocessedDatasets_JulietOverview"]]
    # mode = [DatasetMode.JULIET, DatasetMode.MERGE, DatasetMode.NEW]
    # To not use the taxonomy and to specify a single custom group of CWEs, a fourth element containing the desired
    # CWEs as list of strings can additionally be provided:
    # mode = [DatasetMode.JULIET, DatasetMode.MERGE, DatasetMode.EXISTING,
    #         [os.path.join(ENVIRONMENT.data_root_path, "PreprocessedDatasets_JulietOverview")],
    #         SummarizeJulietExperiments.JULIETPAPERCWES]

    # --- MemNet dataset ---
    # mode = [DatasetMode.MEMNET]

    # --- ArrDeclAccess dataset ---
    # mode = [DatasetMode.ARRDECLACCESS]

    # --- Loading ---
    # LOAD an already existing dataset. load_from's value is expected to be a dataset custodian's directory name in the
    #  current environment. To specify an absolute path to a dataset custodian's directory, is_path must be set to True.
    #  forget_sequences specifies whether loaded feature sequences should be ignored and recalculated from the loaded
    #  ASTs.
    # mode = [DatasetMode.LOAD, {"load_from": "ff64dfea0506f843be314ac5fe61bffd___ArrDeclAccess____C++_____ClangTooling______faulty____correct______test_fraction___0.15___validation_fraction___0.15_____home_simon_MasterThesisData_ArrDeclAccess___None__None__None__False____random____around-decision-bor",
    #                            "is_path": False, "forget_sequences": False}]

    # Second, define the training-related parameter values. These are used for a grid search, i.e., there will be one
    #  training for each combination of the parameter values (and for each dataset as specified above).

    # Specifies how much output should be produced during training.
    # 0 corresponds to almost no output, 10 to most output. 2 is default.
    verbosity = [2]

    # Specifies how often each trained should be repeated
    repetitions = [3]

    # Specify the number of epochs each training will last.
    #  epochs specifies the maximum number of epochs.
    #  early_stopping specifies under what circumstances the training will be stopped before reaching the "epochs"th
    #  epoch.
    #   The training will be stopped if there is no progress for "patience" epochs. There is progress if the validation
    #   score increases. If the validation score stays the same (e.g. because it reached its maximum), there is progress
    #   only if the training loss does improve by "train_loss_reduction_percentage" percent. This avoids too many
    #   checkpoints for only infinitesimal small training loss improvements.
    #   Additionally, the training will be stopped if the training loss is below "train_loss_stop_threshold" and the
    #   validation score did not improve for "train_loss_stop_threshold_patience" epochs. This prevents overfitting and
    #   unnecessary time consumption.
    epochs = [1000]
    early_stopping = [{"patience": 50, "train_loss_reduction_percentage": 25,
                       "train_loss_stop_threshold_patience": 20, "train_loss_stop_threshold": 0.05}]

    # Specifies the ANN type and thus which feature representation will be used.
    net_type = ["embedding", "basic"]
    embedding_vector_sizes = [{"node_kind_num": 13, "type_kind_num": 8, "operator_kind_num": 4, "member_kind_num": 4,
                               "value_kind_num": 4}]

    # Specifies the activation function of the FC layers
    activation = ["sigmoid"]

    # Specifies the training algorithm aka optimizer to use. The name of the desired optimizer's class as defined in
    # module tensorflow.train must be specified.
    optimizer = ["AdamOptimizer"]

    # Specifies the learning rate that should be used by the training algorithm. Must be None if the training algorithm
    #  does not accept a learning rate constructor argument.
    learn_rate = [0.001]  # 0.001 is default for RMSProp and Adam

    # Specifies the batch size
    batch_size = [128]

    # Specifies the number of neurons in the main LSTM layer. See "nalu_neuron_count" for sub-LSTM layers' neuron
    #  counts.
    main_lstm_neuron_count = [24]

    # Specifies the number of neurons in either the sub-NALU layer or the main NALU layer (depending on "net_type").
    #  The sub-LSTM layers' neuron counts (in case of embeddings net type) are automatically derived from this value
    #  (see implementation).
    nalu_neuron_count = [8]

    # Specifies whether a full NALU or only a NAC should be used
    nalu_nac_only = [False]

    # Specifies the number of neurons in the first FC layer (the last FC layer always has as many neurons as classes are
    #  present in the dataset)
    dense_neuron_count = [4]

    # Specifies which threshold should be used for computation of training, validation, and test scores.
    #  In non-binary classification, this setting is ignored and argmax is always used.
    #  In binary classification, "None" specifies that the best threshold should automatically be determined from the
    #   training set. "argmax" specifies that argmax should be used.
    threshold_mode = ["argmax"]

    # Specifies whether a fresh ANN should be initialized before each training ("None") or whether a given ANN model
    #  should be loaded instead. In case of the latter, an absolute path to the ANN model directory must be specified.
    #  This allows the retraining of an ANN model (on another dataset).
    pretrained_model_path = [None]
    # pretrained_model_path = ["/home/simon/MasterThesisData/PreparedDatasets/ff64dfea0506f843be314ac5fe61bffd___ArrDeclAccess____C++_____ClangTooling______faulty____correct______test_fraction___0.15___validation_fraction___0.15_____home_simon_MasterThesisData_ArrDeclAccess___None__None__None__False____random____around-decision-bor/GridSearch/21f58def11441f999a3ecec667d9b61e_128_24_8_False_4_embedding_sigmoid_AdamOptimizer_0.001_argmax_None_1000_3___patience___50___train_loss_reduction_percentage___25___train_loss_stop_threshold_patience___20___train_loss_stop_threshold___0.05____node_kind_num/0.352941_F1_0.935221_train_loss_0_0"]

    # --- End of configuration -----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Assert correct mode values:
    assert mode[0] < DatasetMode.LEVEL1, mode
    if len(mode) > 1 and isinstance(mode[1], DatasetMode):
        assert DatasetMode.LEVEL1 < mode[1] < DatasetMode.LEVEL2, mode
    if len(mode) > 2 and isinstance(mode[2], DatasetMode):
        assert DatasetMode.LEVEL2 < mode[2], mode

    print("Determining data preparation configs for mode", mode, "...")
    # Branch on mode values:
    if mode[0] == DatasetMode.LOAD:
        data_preparation_configs = [mode[1]]
    elif mode[0] == DatasetMode.JULIET:
        # Default Juliet config:
        base_config = dict(
            type=Juliet,
            # CWEs from which a multi-class dataset will be created
            label_whitelist=None,  # None corresponds to one multi-class dataset with all CWEs.
            # Difficulty class according to Juliet userguide. None corresponds to all classes.
            difficulty_classes=None,  # [Juliet.DifficultyClass.SIMPLE],
            max_sequence_length=None,
            max_samples_to_prepare_for_training=None,
            min_sample_count_per_class=10,
            skip_samples_with_helper_methods=False,
            language="C++",  # "Java",
            split_info=dict(test_fraction=0.15, validation_fraction=0.15)
        )

        # Check for dict in last element which may contain adjustments to the base config:
        if isinstance(mode[-1], dict):
            for adjusted_base_conf_key, adjusted_base_conf_val in mode[-1].items():
                assert adjusted_base_conf_key in base_config.keys(), \
                    ("Adjustment dict contain unknown key", adjusted_base_conf_key, mode, base_config)
                base_config[adjusted_base_conf_key] = adjusted_base_conf_val
            # Remove adjustment dict
            mode = mode[:-1]

        if mode[1] == DatasetMode.ALL:
            data_preparation_configs = [base_config]
        elif mode[1] == DatasetMode.SELECTED:  # TODO: Merge wih SEPARATE
            # Each selected cwe separately:
            data_preparation_configs = [{**base_config, "label_whitelist": [cwe_name]} for cwe_name in mode[2]]
        elif mode[1] == DatasetMode.SEPARATE:
            # Each cwe separately:
            data_preparation_configs = [{**base_config, "label_whitelist": [cwe_name]} for cwe_name in ALL_CWE_CLASSES]
        elif mode[1] == DatasetMode.MERGE:
            data_preparation_configs = []
            juliet_cwe_dict = CweTaxonomy.get_juliet_cwes()
            train_result_data_list = None
            if mode[2] == DatasetMode.EXISTING:
                # Ensure list if only one path is provided:
                if isinstance(mode[3], str):
                    mode[3] = [mode[3]]
                # Read all already prepared datasets from the given paths:
                train_result_data_list = SummarizeJulietExperiments.get_train_result_summaries(mode[3])

            if len(mode) > 4 and isinstance(mode[4], list):
                # Use specified cwes as single group:
                juliet_cwe_groups = {"000": mode[4]}
            else:
                # Use cwe taxonomy groups:
                juliet_cwe_groups = CweTaxonomy.get_cwe_groups(juliet_cwe_dict)

            for group_top_level_parent_cwe, group_member_cwe_list in juliet_cwe_groups.items():
                if len(group_member_cwe_list) == 1:
                    print("Skipping group", group_top_level_parent_cwe, "because it only contains one element.")
                    continue
                if mode[2] == DatasetMode.EXISTING:
                    # Determine the absolute paths to the already existing datasets depending on the requested CWEs:
                    merge_from_paths = []
                    for group_member_cwe in group_member_cwe_list:
                        found = False
                        for train_result_data in train_result_data_list:
                            # Check whether the current dataset is a single-CWE dataset with a matching CWE:
                            if train_result_data["cwes"] == group_member_cwe:
                                merge_from_paths.append(train_result_data["abs_path"])
                                found = True
                                break
                        if not found:
                            raise ValueError("No already existing single-CWE dataset found for cwe", group_member_cwe,
                                             "in specified paths", mode[3])
                    # The use of the merge_from key results in the creation of a dataset that is merged from already
                    #  existing datasets
                    data_preparation_configs.append({"merge_from": merge_from_paths,
                                                     "is_path": True,
                                                     "split_info": base_config["split_info"],
                                                     **base_config})
                elif mode[2] == DatasetMode.NEW:
                    # Create "normal" multi-class dataset:
                    data_preparation_configs.append({**base_config,
                                                     "label_whitelist": [juliet_cwe_dict[group_member_cwe]
                                                                         for group_member_cwe in
                                                                         group_member_cwe_list]})
                else:
                    assert False, (1, mode)
        else:
            assert False, (2, mode)
    elif mode[0] == DatasetMode.MEMNET:
        data_preparation_configs = [
            dict(
                type=MemNet,
                file_names_to_use=None,  # None ^= all
                split_info="paper",  # dict(test_fraction=0.15, validation_fraction=0.15)
            )
        ]
    elif mode[0] == DatasetMode.ARRDECLACCESS:
        data_preparation_configs = [
            dict(
                type=ArrDeclAccess,
                language="C++",
                kinds=["random", "around-decision-border", "around-decision-border", "around-decision-border"],
                sample_count=10,
                lower_limit=0,
                upper_limit=10,
                # Extrapolation testing is done manually later. TODO: Implement extrapolation split
                split_info=dict(test_fraction=0.15, validation_fraction=0.15)
            )
        ]
    else:
        assert False, (3, mode)

    print("Determined", len(data_preparation_configs), "data preparation config(s)",
          "\n" + pprint.pformat(data_preparation_configs) if len(data_preparation_configs) > 1 else "")

    # Create parameter dict from config above:
    # Ordered such that keys which have multiple values and which are important are at the beginning. This way, during
    # printing those come first
    train_parameter_values = dict(
        batch_size=batch_size,
        main_lstm_neuron_count=main_lstm_neuron_count,
        nalu_neuron_count=nalu_neuron_count,
        nalu_nac_only=nalu_nac_only,
        dense_neuron_count=dense_neuron_count,
        net_type=net_type,
        activation=activation,
        optimizer=optimizer,
        learn_rate=learn_rate,
        threshold_mode=threshold_mode,
        # dropout_rate=dropout_rate,
        # init_mode=init_mode,
        # weight_constraint=weight_constraint,
        pretrained_model_path=pretrained_model_path,
        epochs=epochs,
        repetitions=repetitions,
        early_stopping=early_stopping,
        embedding_vector_sizes=embedding_vector_sizes,
        verbosity=verbosity
    )

    # For each data preparation config, utilize the training and testing pipeline:
    for data_preparation_index, data_preparation_config in enumerate(data_preparation_configs):

        # ### Data preparation ###
        print("Data preparation config ", data_preparation_index + 1, "/", len(data_preparation_configs), ":\n",
              pprint.pformat(data_preparation_config), "\n ...", sep="")
        # Create dataset custodian based on config:
        dataset_custodian = DatasetCustodian.create(data_preparation_config)
        # Produce ast nodes, generate sequences, and split them:
        dataset_custodian.prepare()
        # Print dataset statistics and first few samples:
        dataset_custodian.print_dataset_statistics()
        dataset_custodian.print_first_samples(2, omit_padding=False)
        # Do not perform training in case of no sequences:
        if dataset_custodian.get_sample_count() == 0 or (dataset_custodian.get_labels()) == 0:
            print("Skipping grid search because there are either no samples or no classes.")
            continue

        # ### Training and Testing ###
        # Initialize and perform grid search:
        grid_search = GridSearch(dataset_custodian, train_parameter_values)
        best_param_and_score = grid_search.search()
        print("Best model information and scores:", best_param_and_score)
