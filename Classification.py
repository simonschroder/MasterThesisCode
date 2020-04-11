"""
Classification (& Attribution) component and extrapolation testing of ArrDeclAccess dataset
"""

# To avoid errors like "_tkinter.TclError: couldn't connect to display "localhost:10.0"" when closing ssh connection:
import matplotlib

matplotlib.use('Agg')
from matplotlib.pyplot import ioff

ioff()

from DataPreparation import MemNet, ArrDeclAccess, Juliet

from Attribution import visualize_grads_of_output_wrt_to_input, GridSearchCheckpointInfo
from Training import get_scores
from Helpers import *


def get_arrdeclaccess_sequence(sequence, decl_upper_bound, access_upper_bound):
    """
    Adjusts the decl and access upper bound in the given ArrDeclAccess feature sequence. This allows to create
     a custom extrapolation test set for "simple" or legacy models.
    :param sequence:
    :param decl_upper_bound:
    :param access_upper_bound:
    :return:
    """
    sequence = sequence.copy()
    if sequence.shape[1] == 2:
        # Simple feature representation
        sequence[0, 0] = sequence[0, 1] = decl_upper_bound
        sequence[2, 0] = sequence[2, 1] = access_upper_bound
    else:
        decl_found = 0
        access_found = 0
        for row_index in range(sequence.shape[0]):
            # Look for the correct node kind features
            if sequence[row_index, 0] == 56:
                decl_found += 1
                if decl_found == 1:
                    decl_ub1 = sequence[row_index, 9]
                    decl_ub2 = sequence[row_index, 46]
                    assert decl_ub1 == decl_ub2
                    sequence[row_index, 9] = sequence[row_index, 46] = decl_upper_bound
            elif sequence[row_index, 0] == 155:
                access_found += 1
                if access_found == 2:
                    access_ub1 = sequence[row_index, 27]
                    access_ub2 = sequence[row_index, 38]
                    assert access_ub1 == access_ub2
                    sequence[row_index, 27] = sequence[row_index, 38] = access_upper_bound
    return sequence


if __name__ == "__main__":

    # ------------------------------------------------------------------------------------------------------------------
    # --- Begin of configuration ---------------------------------------------------------------------------------------

    # First, specify what testing/classification/attribution tasks you want to perform. Afterwards, you can specify
    #  which existing model and dataset to load and use for this.

    # Compute (test) scores.
    #  For Juliet multi-class datasets/models, this includes separate accuracy scores.
    #  For MemNet datasets/models, this includes separate test set level scores.
    #  For ArrDeclAccess datasets/models, this includes extrapolation tests. See implementation for more details and
    #   adjustments
    compute_scores = True

    # Evaluate the model on the given source code snippet, i.e., classify the source code snippet, and generate
    #  attribution overlays for it. The result is stored in the loaded model's directory.
    eval_unknown_source = True
    unknown_source_content_tuple = ("""
int main() {
  int arr[""" + str(30) + """];   
  for(int i = 0; i < """ + str(30) + """; i += 1) {   
     arr[i] = i;
  }  
  return 0;
}""", ".cpp")

    # Classify each feature sequence from the test set and generate attribution overlays for it. The results are stored
    #  in the loaded model's directory.
    visualize_test_data = True

    # Second, specify the model and dataset to load and use for the above-specified tasks. Assign a tuple to
    #  saved_model_and_dc_paths for this.
    #
    #  The tuple's first component specifies the absolute path to the directory of the model that will be loaded. If no
    #  other information is provided, the model's associated dataset is used. If a different dataset should be used, its
    #  dataset custodian directory name can be specified as second component. To provide a full path to the dataset
    #  custodian directory instead, provide True as third component.
    #
    #  It is also possible to perform the above tasks on a batch of models or datasets. See the implementation below for
    #  information about how to set the components in these cases.

    saved_model_and_dc_paths = (
        "/home/simon/MasterThesisData/PreparedDatasets/ff64dfea0506f843be314ac5fe61bffd___ArrDeclAccess____C++_____ClangTooling______faulty____correct______test_fraction___0.15___validation_fraction___0.15_____home_simon_MasterThesisData_ArrDeclAccess___None__None__None__False____random____around-decision-bor/GridSearch/21f58def11441f999a3ecec667d9b61e_128_24_8_False_4_embedding_sigmoid_AdamOptimizer_0.001_argmax_None_1000_3___patience___50___train_loss_reduction_percentage___25___train_loss_stop_threshold_patience___20___train_loss_stop_threshold___0.05____node_kind_num/0.352941_F1_0.935221_train_loss_0_0/",
    )

    # --- End of configuration -----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Check whether the tasks should be performed on multiple models. First component is None in this case and the
    #  second component should contain a path to a CSV file. The model paths given in column best_val_F1model_path are
    #  loaded and used.
    if saved_model_and_dc_paths[0] is None and isinstance(saved_model_and_dc_paths[1], str):
        # Determine models from csv:
        import pandas as pd

        df = pd.read_csv(saved_model_and_dc_paths[1])
        saved_column = df.best_val_F1model_path
        list_of_saved_model_and_dc_paths = []
        for model_path in saved_column:
            list_of_saved_model_and_dc_paths.append((model_path,))
    # Check whether the tasks should be performed on multiple datasets. First two components are None, third component
    #  should contain model path, and fourth component a list of paths to dataset custodian directories.
    elif saved_model_and_dc_paths[0] is None and saved_model_and_dc_paths[1] is None:
        # Test model on different datasets:
        list_of_saved_model_and_dc_paths = []
        for dc_path in saved_model_and_dc_paths[3]:
            list_of_saved_model_and_dc_paths.append((saved_model_and_dc_paths[2], dc_path, True))
    # Normal case as described in configuration above
    else:
        list_of_saved_model_and_dc_paths = [saved_model_and_dc_paths]

    for saved_model_and_dc_paths in list_of_saved_model_and_dc_paths:
        gridsearch_checkpoint = GridSearchCheckpointInfo(*saved_model_and_dc_paths)

        print("### Classification ###", saved_model_and_dc_paths)

        # Select feat vec components to show depending on model architecture:
        if gridsearch_checkpoint.model_params["net_type"].startswith("embedding"):
            feat_vec_components_to_show = ["kind_numbers", "feat_numbers"]
        elif gridsearch_checkpoint.model_params["net_type"].startswith("basic"):
            feat_vec_components_to_show = ["basic_feat_vec"]
        else:
            # Show everything possible:
            feat_vec_components_to_show = None

        loaded_train_threshold = gridsearch_checkpoint.model_scores["train"]["scores"]["threshold"]
        loaded_apply_softmax = gridsearch_checkpoint.model_scores["train"]["scores"].get("softmax_has_been_applied",
                                                                                         False)
        print("Loaded train threshold:", loaded_train_threshold)
        print("Loaded softmax application:", loaded_apply_softmax)

        # # loaded_train_threshold = "argmax"
        # loaded_train_threshold = 0.25
        # loaded_apply_softmax = True
        # print("RESETTING loaded train threshold and softmax application", loaded_train_threshold,
        #       loaded_apply_softmax)

        memnet_paper_split = None
        if isinstance(gridsearch_checkpoint.dataset_custodian, MemNet):
            memnet_paper_split = "test" if gridsearch_checkpoint.dataset_custodian.split_info[
                                               "test_fraction"] > 0 else "validation"

        if compute_scores:
            print()
            # Evaluate train scores:
            train_score, train_score_name, train_all_scores, train_summary \
                = get_scores(gridsearch_checkpoint.model, gridsearch_checkpoint.dataset_custodian, "train",
                             threshold_to_use=loaded_train_threshold, apply_softmax=loaded_apply_softmax,
                             no_print=False, model_to_dataset_mapping=gridsearch_checkpoint.model_to_dataset_mapping)
            print()
            print("Actual Train Scores:", train_all_scores)
            print("Loaded Train Scores:", gridsearch_checkpoint.model_scores["train"])
            print()

            # Memnet Paper split:
            if memnet_paper_split is not None:
                # "Memnet Paper" split
                memnet_paper_split_indices \
                    = gridsearch_checkpoint.dataset_custodian.traintestval_split_indices[memnet_paper_split]
                for test_split_part in range(4):
                    part_lower_index = test_split_part * 1000
                    part_upper_index = (test_split_part + 1) * 1000
                    test_part_indices = memnet_paper_split_indices[part_lower_index:part_upper_index]
                    print("\n\nPart", test_split_part + 1, "of MemNet paper test split",
                          test_part_indices[0], "-", test_part_indices[-1], "...")
                    test_score, test_score_name, test_all_scores, test_summary \
                        = get_scores(gridsearch_checkpoint.model, gridsearch_checkpoint.dataset_custodian,
                                     test_part_indices,
                                     threshold_to_use=loaded_train_threshold, apply_softmax=loaded_apply_softmax,
                                     no_print=True,
                                     model_to_dataset_mapping=gridsearch_checkpoint.model_to_dataset_mapping)
                    print(test_score_name, test_score)
                    print("f1_macro_score", test_all_scores["f1_macro_score"])
                    print("f1_weighted_score", test_all_scores["f1_weighted_score"])
                    print("balanced_accuracy", test_all_scores["balanced_accuracy"])
                    print(test_all_scores["conf_matrix"])
                    print(test_all_scores["classification_report"])
                    print("(auroc (multiple thresholds derived from data): ", test_all_scores["aroc_c1_score"], ")",
                          sep="")
            # ArrDeclAccess Extrapolation:
            if isinstance(gridsearch_checkpoint.dataset_custodian, ArrDeclAccess):
                dc: ArrDeclAccess = gridsearch_checkpoint.dataset_custodian

                print("\nTesting extrapolation of model trained with samples in [", dc.lower_limit, ", ",
                      dc.upper_limit, "] ", sep="", end="")
                test_lower_limit = dc.lower_limit
                test_upper_limit = dc.upper_limit * 10
                test_every_ith = 5
                print("with new unknown samples from interval [", test_lower_limit, ", ", test_upper_limit, "] ...",
                      sep="")

                sequences, labels_onehot, descriptions = [], [], []
                example_sequence = to_numpy_array(
                    gridsearch_checkpoint.dataset_custodian.get_sequence_tuple_for_tensorflow_from_index(
                        dc.traintestval_split_indices["train"][0])
                    [0]
                )
                # print(array_to_string(example_sequence))
                for i in range(test_lower_limit, test_upper_limit, test_every_ith):
                    for j in range(2):
                        sequence = get_arrdeclaccess_sequence(example_sequence, i, i + j)
                        sequences.append(sequence)
                        labels_onehot.append(int_to_onehot(1 - j, 2))
                        descriptions.append(dc.generate_sample(i, i + j, dc.language).replace("\n", " "))

                # sources = list(dc.gen_sample_sources("around-decision-border"))
                # s
                # for source, label in sources:
                #     print(source)
                #     sequence = next(dc.gen_sequence_tuples_for_tensorflow_of_unknown_source_content(source,
                #                                                                                     ".cpp",
                #                                                                                     verbosity=1),
                #                     None)[0]
                #     sequences.append(sequence)
                #     print(array_to_string(sequence))
                #     labels_onehot.append(int_to_onehot(dc.label_to_int_mapping[label], len(dc.get_labels())))
                #     descriptions.append(source.replace("\n", " "))

                # Evaluate extrapolation scores:
                extrapolation_score, extrapolation_score_name, extrapolation_all_scores, extrapolation_summary \
                    = get_scores(gridsearch_checkpoint.model, gridsearch_checkpoint.dataset_custodian,
                                 sequences_and_labels_onehot=(sequences, labels_onehot, descriptions),
                                 threshold_to_use=loaded_train_threshold,
                                 apply_softmax=loaded_apply_softmax,
                                 no_print=False,
                                 model_to_dataset_mapping=gridsearch_checkpoint.model_to_dataset_mapping)
                print(saved_model_and_dc_paths, "Extrapolation Scores:", extrapolation_all_scores)
            else:
                # Evaluate test scores:
                test_score, test_score_name, test_all_scores, test_summary \
                    = get_scores(gridsearch_checkpoint.model, gridsearch_checkpoint.dataset_custodian, "test",
                                 threshold_to_use=loaded_train_threshold, apply_softmax=loaded_apply_softmax,
                                 no_print=False,
                                 model_to_dataset_mapping=gridsearch_checkpoint.model_to_dataset_mapping)
                print("Test Scores:", test_all_scores)

                # Compute separate Accuracy scores:
                if isinstance(gridsearch_checkpoint.dataset_custodian, Juliet) \
                        and len(gridsearch_checkpoint.dataset_custodian.get_labels()) > 2:
                    test_indices = gridsearch_checkpoint.dataset_custodian.traintestval_split_indices["test"]
                    cwe_wise_indices = {}
                    for test_index in test_indices:
                        x_data = gridsearch_checkpoint.dataset_custodian \
                            .get_sequence_tuple_for_tensorflow_from_index(test_index)
                        # Retrieve CWE from AST node's file path:
                        file_path: str = x_data[3][0].tu.id.spelling
                        cwe_str_index = file_path.find("CWE")
                        assert cwe_str_index >= 0, file_path
                        # Sequence's CWE
                        actual_sequence_cwe_label = file_path[cwe_str_index: file_path.find("/", cwe_str_index)]
                        # Sequence's good/bad specific label:
                        merged_sequence_cwe_label = gridsearch_checkpoint.dataset_custodian.int_to_label_mapping[
                            x_data[1]]
                        # Create object, which contains indices for each cwes good and bad cases:
                        cwe_index_obj = cwe_wise_indices.setdefault(actual_sequence_cwe_label, {"bad": [], "good": []})
                        # If the sequence is a good one, it has No-bug label in merged dataset. Otherwise its a bad one
                        if merged_sequence_cwe_label == gridsearch_checkpoint.dataset_custodian.get_no_bug_label():
                            cwe_index_obj["good"].append(test_index)
                        else:
                            assert actual_sequence_cwe_label == merged_sequence_cwe_label, \
                                (actual_sequence_cwe_label, merged_sequence_cwe_label)
                            cwe_index_obj["bad"].append(test_index)

                    print("CWE-wise test scores:")
                    for cwe, cwe_index_obj in cwe_wise_indices.items():
                        # Evaluate test scores:
                        test_score, test_score_name, test_all_scores, test_summary \
                            = get_scores(gridsearch_checkpoint.model, gridsearch_checkpoint.dataset_custodian,
                                         split_name_or_indices=cwe_index_obj["good"] + cwe_index_obj["bad"],
                                         threshold_to_use=loaded_train_threshold, apply_softmax=loaded_apply_softmax,
                                         no_print=False,
                                         model_to_dataset_mapping=gridsearch_checkpoint.model_to_dataset_mapping)
                        print("cwe", cwe, "\n", test_all_scores)

        if eval_unknown_source:
            # Evaluate (unknown) source code snippet:
            x_datas = list(
                gridsearch_checkpoint.dataset_custodian.gen_sequence_tuples_for_tensorflow_of_unknown_source_content(
                    *unknown_source_content_tuple))
            # Source content may contain multiple source code snippets:
            for index, x_data in enumerate(x_datas):
                visualize_grads_of_output_wrt_to_input(gridsearch_checkpoint, x_data[0], x_data[3],
                                                       save_plot_file_name_prefix="test" + str(index) + get_timestamp(),
                                                       feat_vec_components_to_show=feat_vec_components_to_show,
                                                       multiplicate_with_input=True,
                                                       integrated_gradients_baseline=0,
                                                       integrated_gradients_steps=100,
                                                       add_embeddings=True,
                                                       threshold_to_use=loaded_train_threshold,
                                                       apply_softmax=loaded_apply_softmax)

        if visualize_test_data:
            # Generate visualizations for a specific Juliet testcase and for different attribution technique variants:
            # if isinstance(gridsearch_checkpoint.dataset_custodian, Juliet):
            #     index = 0
            #     while True:
            #         x_data = gridsearch_checkpoint.dataset_custodian.get_sequence_tuple_for_tensorflow_from_index(index)
            #         file_path = x_data[3][0].tu.id.spelling
            #         if "CWE124_Buffer_Underwrite__char_declare_cpy_17" in file_path:
            #             test_sample_as_array = to_numpy_array([x_data[0]])
            #             for (baseline, steps) in ((None, 100), (0, 20), (0, 100), (0, 300)):
            #                 visualize_grads_of_output_wrt_to_input(gridsearch_checkpoint, test_sample_as_array[0],
            #                                                        x_data[3], x_data[1],
            #                                                        save_plot_file_name_prefix=str(
            #                                                            index) + "_act_" + str(
            #                                                            gridsearch_checkpoint.dataset_custodian.int_to_label_mapping[
            #                                                                x_data[1]]),
            #                                                        visualization_dir_path=None,  # "/home/simon/Desktop",
            #                                                        feat_vec_components_to_show=feat_vec_components_to_show,
            #                                                        multiplicate_with_input=True,
            #                                                        integrated_gradients_baseline=baseline,  # 0,
            #                                                        integrated_gradients_steps=steps,
            #                                                        threshold_to_use=loaded_train_threshold,
            #                                                        apply_softmax=loaded_apply_softmax,
            #                                                        add_embeddings=True,
            #                                                        skip_gradient_calculation=False)
            #
            #         index += 1

            # Determine test indices depending on dataset (split):
            if memnet_paper_split is not None:
                test_indices = gridsearch_checkpoint.dataset_custodian.traintestval_split_indices[memnet_paper_split]
                memnet_test_indices = []
                CONTINUOUS_COUNT = 10
                curr_index_to_indices_list = 0
                curr_per_sub_split_count = 0
                stop = False
                while not stop:
                    for sub_split_index in range(4):
                        curr_index_to_indices_list = sub_split_index * 1000 + curr_per_sub_split_count
                        for _ in range(CONTINUOUS_COUNT):
                            if curr_index_to_indices_list < len(test_indices):
                                memnet_test_indices.append(test_indices[curr_index_to_indices_list])
                                curr_index_to_indices_list += 1
                            else:
                                stop = True
                                break
                    curr_per_sub_split_count += CONTINUOUS_COUNT
                test_indices = memnet_test_indices
            else:
                test_indices = gridsearch_checkpoint.dataset_custodian.traintestval_split_indices["test"]
            # For each test index, generate attribution (overlays):
            for test_index in test_indices:
                # Get corresponding source code snippet as feature sequence:
                x_data = gridsearch_checkpoint.dataset_custodian.get_sequence_tuple_for_tensorflow_from_index(
                    test_index)

                test_sample_as_array = to_numpy_array([x_data[0]])

                # Multiple attribution technique variants can be specified:
                for (baseline, steps) in [(0, 100)]:  # ((None, 100), (0, 20), (0, 100), (0, 300)):
                    visualize_grads_of_output_wrt_to_input(gridsearch_checkpoint, test_sample_as_array[0], x_data[3],
                                                           x_data[1],
                                                           save_plot_file_name_prefix=str(test_index) + "_act_" + str(
                                                               gridsearch_checkpoint.dataset_custodian.int_to_label_mapping[
                                                                   x_data[1]]),
                                                           visualization_dir_path=None,  # "/home/simon/Desktop",
                                                           feat_vec_components_to_show=feat_vec_components_to_show,
                                                           multiplicate_with_input=True,
                                                           integrated_gradients_baseline=baseline,
                                                           integrated_gradients_steps=steps,
                                                           threshold_to_use=loaded_train_threshold,
                                                           apply_softmax=loaded_apply_softmax,
                                                           add_embeddings=True,
                                                           skip_gradient_calculation=False)
