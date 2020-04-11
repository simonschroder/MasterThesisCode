import ast
import csv
import os

# Use "regex" instead of default "re" as "regex" supports atomic groups (?>...|...)
import regex as re

import DataPreparation
from CweTaxonomy import get_cwe_name
from Environment import ENVIRONMENT
from Helpers import get_non_rec_dir_content_names, from_json, read_file_content, from_legacy_dc_info_json_string, \
    serialize_data, as_safe_filename, deserialize_data
from Training import GridSearch

# 22 CWEs used in paper "On the capability of static code analysis to detect security vulnerabilities":
JULIETPAPERCWES = ["78",
                   "122",
                   "134",
                   "197",
                   "242",
                   "367",
                   "391",
                   "401",
                   "415",
                   "416",
                   "457",
                   "467",
                   "468",
                   "476",
                   "478",
                   "480",
                   "482",
                   "561",
                   "562",
                   "563",
                   "590",
                   "835"]


def get_train_result_summaries(train_result_root_dir_paths, limit_juliet_dirs_count=None):
    train_result_data_list = []
    output_info_matcher = re.compile(r"^"
                                     # Optional (if it does not match, the group will be None). Do not use
                                     # (?:Sample length statistics: (\{.+?\})\n)?  because nested greedy operators
                                     # (+? inside (...)?) may yield catastrophic backtracking. Instead, as soon as
                                     # "Determined max" was read, stop looking for "Sample length statistics:" (if
                                     # latter is there, it will be before the former)
                                     # See: https://www.regular-expressions.info/catastrophic.html
                                     + r".*?Collected (\d+?) test cases \((\d+?) of these test cases need windows environment\)\n"
                                     + r".*?Skipped (\d+?) of (\d+?) AST collections because of parsing error\(s\)\n"
                                     + r".*?(?>Preparing AST|Skipped (\d+?) of (\d+?) AST collections because of parsing error\(s\)\n)"
                                     + r".*?(\d+?) samples found(?: |)\n"
                                     + r".*?(?>Determined max|Sample length statistics: \{'min': (\d*\.?\d+), 'max': (\d*\.?\d+), 'mean': (\d*\.?\d+), 'median': (\d*\.?\d+), 'std': (\d*\.?\d+), 'var': (\d*\.?\d+)\}\n)"
                                     + r".*?(?>No too small classes|Ignoring.*?samples: (\[.*?\])\n)"
                                     + r".*?Prepared and serialized sequences in (\d+?) s"
                                     + r".*?(?:Split:|into) (\{.+?\})\n"  # (?:a|b) is non capturing OR
                                     + r".*?Label count: (\d+?)\n"
                                     + r".*?Counts per label: (\[.*?\])\n"
                                     + r".*?Sequence count: (\d+?)\n"
                                     + r"Sequence shape: \((\d+?), (\d+?)\)\n"
                                     + r".*?$",
                                     re.DOTALL)

    dir_paths = []

    print("Collecting training result directory paths in ...")
    for train_result_root_dir_path in train_result_root_dir_paths:
        print(train_result_root_dir_path, "...")
        dir_paths.extend([abs_path
                          for abs_path, file_or_dir_name
                          in get_non_rec_dir_content_names(train_result_root_dir_path)
                          if os.path.isdir(abs_path)])

    print("Collected", len(dir_paths), "paths of training result directories.")
    # Sort by date
    dir_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    for index, dir_path in enumerate(dir_paths):
        file_or_dir_name = os.path.basename(dir_path)
        print("Directory ", index + 1, "/", len(dir_paths), ": ", file_or_dir_name,
              " (", os.path.dirname(dir_path), ") ...", sep="")
        train_result_data = {"dir_name": file_or_dir_name, "abs_path": dir_path}
        try:
            info_data = from_json(read_file_content(os.path.join(dir_path, "info.json")))
            if isinstance(info_data, str):
                info_data = from_legacy_dc_info_json_string(info_data)

            output_string = read_file_content(os.path.join(dir_path, "output.txt"))
        except IOError as io_err:
            print(" - Skipping because of error", io_err)
            continue

        # Make sure info_data is an array of strings and not a string
        assert len(info_data[0]) > 1
        if info_data[0] != "Juliet":
            print(" - Skipping non-Juliet training", info_data)
            continue

        if info_data[1] != "C++":
            print(" - Skipping non-C++ training", info_data)
            continue

        if info_data[2] != "'ClangTooling'" and info_data[2] != ["ClangTooling"]:
            print(" - Skipping non-ClangTooling training", info_data)
            continue

        cwe_juliet_names = info_data[3]
        if cwe_juliet_names[-1] == "CWE00_NO_FLAW_FOUND":
            cwe_juliet_names = cwe_juliet_names[:-1]
        cwes = [cwe_name.split("_")[0][3:] for cwe_name in cwe_juliet_names]
        cwe_names = [get_cwe_name(cwe) for cwe in cwes]
        cwe_names = [cwe_name if cwe_name is not None else "<NA>" for cwe_name in cwe_names]
        train_result_data["cwe_juliet_names"] = ",".join(cwe_juliet_names)
        train_result_data["cwe_names"] = ",".join(cwe_names)
        train_result_data["cwes"] = ",".join(cwes)

        matches = output_info_matcher.search(output_string)
        if matches is None:
            print(" - Skipping because regexp does not match.")
            # print(output_string)
            # assert False
            continue
        else:
            for group_index, key in enumerate(["testcase_count_overall", "testcase_count_win",
                                               "parse_0_testcase_skip", "parse_0_testcase_overall",
                                               "parse_1_testcase_skip", "parse_1_testcase_overall",
                                               "sequence_count_pre_filter",
                                               "sequence_len_min", "sequence_len_max", "sequence_len_mean",
                                               "sequence_len_median", "sequence_len_std", "sequence_len_var",
                                               "ignored_counts_per_label",
                                               "preparation_time_in_sec", "split", "label_count",
                                               "counts_per_label", "sequence_count", "sequence_len_max",
                                               "sequence_feat_vec_length"]):
                # First group is whole match (i.e. whole output)
                train_result_data[key] = matches.group(group_index + 1)

        print(train_result_data)

        # Actual train results:
        gridsearch_dir_path = os.path.join(dir_path, "GridSearch")
        if os.path.exists(gridsearch_dir_path):
            param_combs_data, _ = GridSearch.generate_gridsearch_summary_csv(gridsearch_dir_path,
                                                                             is_path=True,
                                                                             write_csv=False)
            for param_comb_index, param_comb_data in enumerate(param_combs_data):
                for param_comb_data_key, param_comb_data_val in param_comb_data.items():
                    train_result_data["pc" + str(param_comb_index) + param_comb_data_key] = param_comb_data_val
        print(" -", train_result_data)
        train_result_data_list.append(train_result_data)
        if limit_juliet_dirs_count is not None and len(train_result_data_list) >= limit_juliet_dirs_count:
            print("Stopping after reaching limit", limit_juliet_dirs_count, " of train result data.")
            break
    print("Got", len(train_result_data_list), "train results.")
    return train_result_data_list


if __name__ == "__main__":
    summary_dir_path = ENVIRONMENT.data_root_path

    train_result_root_dir_paths = [DataPreparation.PREPARED_DATASET_PATH_ROOT]
    # train_result_root_dir_paths = [
    #     "/run/user/1000/gvfs/sftp:host=server1,user=schroeder/home/schroeder/MasterThesisData/PreprocessedDatasets",
    #     "/run/user/1000/gvfs/sftp:host=server2,user=schroeder/home/schroeder/MasterThesisData/PreprocessedDatasets"]

    # train_result_root_dir_paths = [
    #     "/home/schroeder/MasterThesisData/PreprocessedDatasets_JulietOverview"]

    limit_juliet_dirs_count = None

    result_data_pickle_file_path = os.path.join(summary_dir_path,
                                                as_safe_filename(str(train_result_root_dir_paths) + ".pkl"))

    if not os.path.exists(result_data_pickle_file_path):
        serialize_data(get_train_result_summaries(train_result_root_dir_paths, limit_juliet_dirs_count),
                       os.path.join(summary_dir_path, result_data_pickle_file_path))

    train_result_data_list = deserialize_data(result_data_pickle_file_path)

    net_types = []
    for train_result_data in train_result_data_list:
        if len(train_result_data["cwes"].split(",")) == 1:
            # Unify testcase parse skip data:
            testcase_count_overall = int(train_result_data["testcase_count_overall"])
            testcase_count_win = int(train_result_data["testcase_count_win"])
            testcase_overall_skipped = 0
            testcase_win_skipped = 0
            if testcase_count_overall > 0:
                if testcase_count_win == 0:
                    # Only one non-windows parse
                    assert train_result_data["parse_1_testcase_skip"] is None, train_result_data
                    assert int(
                        train_result_data["parse_0_testcase_overall"]) == testcase_count_overall, train_result_data
                    testcase_overall_skipped = int(train_result_data["parse_0_testcase_skip"])
                elif testcase_count_win == testcase_count_overall:
                    # Only one windows parse
                    assert train_result_data["parse_1_testcase_skip"] is None, train_result_data
                    assert int(train_result_data["parse_0_testcase_overall"]) == testcase_count_win, train_result_data
                    testcase_overall_skipped = testcase_win_skipped = int(train_result_data["parse_0_testcase_skip"])
                else:
                    # Two parses: one windows and one non-windows (in that order):
                    assert train_result_data["parse_1_testcase_skip"] is not None, train_result_data
                    assert int(train_result_data["parse_0_testcase_overall"]) == testcase_count_win, train_result_data
                    testcase_win_skipped = int(train_result_data["parse_0_testcase_skip"])
                    assert int(train_result_data["parse_1_testcase_overall"]) == (
                            testcase_count_overall - testcase_count_win), train_result_data
                    testcase_overall_skipped = testcase_win_skipped + int(train_result_data["parse_1_testcase_skip"])
            train_result_data["testcase_overall_skipped"] = str(testcase_overall_skipped)
            train_result_data["testcase_win_skipped"] = str(testcase_win_skipped)

            # Pretty version of CWE Names without CWE number and without underscores
            train_result_data["cwe_juliet_names_pretty"] = " ".join(
                train_result_data["cwe_juliet_names"].split("_")[1:])

            # Make sure that order of gridsearch combs is the same for all train results. This is checked based on
            # net_type:
            for index in range(2):
                index = str(index)
                if train_result_data.get("pc" + index + "net_type", None) is not None:
                    pc_index_net_type = train_result_data["pc" + index + "net_type"]
                    if pc_index_net_type not in net_types:
                        # Create the order based on which comes first
                        net_types.append(pc_index_net_type)
                    else:
                        # Check whether order in this train result matches the overall order:
                        if net_types.index(pc_index_net_type) != int(index):
                            # Order does not match. Switch all train results:
                            for tr_key, tr_val in list(train_result_data.items()):
                                if tr_key.startswith("pc" + index):
                                    other_key = tr_key.replace("pc" + index, "pc" + str((int(index) + 1) % 2))
                                    train_result_data[tr_key] = train_result_data.get(other_key, None)
                                    train_result_data[other_key] = tr_val
            if train_result_data["sequence_len_mean"] is not None:
                train_result_data["seq_count_mult_seq_len_mean"] = str(
                    int(float(train_result_data["sequence_count"]) * float(train_result_data["sequence_len_mean"])))

            if "ignored_counts_per_label" in train_result_data:
                # Aggregate good and bad sequence counts:
                good_seq_count = 0
                bad_seq_count = 0
                counts_per_label = ast.literal_eval(train_result_data["counts_per_label"])
                assert sum([label_count_pair[1] for label_count_pair in counts_per_label]) == int(
                    train_result_data["sequence_count"])
                if train_result_data["ignored_counts_per_label"] is not None:
                    counts_per_label += ast.literal_eval(train_result_data["ignored_counts_per_label"])
                for label_count_pair in counts_per_label:
                    # no bug label:
                    if "NO_FLAW_FOUND" in label_count_pair[0] or label_count_pair[0] == "correct":
                        # This should only happen once:
                        assert good_seq_count == 0, (good_seq_count, counts_per_label)

                        good_seq_count += label_count_pair[1]
                    else:
                        bad_seq_count += label_count_pair[1]
                assert good_seq_count + bad_seq_count == int(train_result_data["sequence_count_pre_filter"])
                train_result_data["sequence_good_count"] = good_seq_count
                train_result_data["sequence_bad_count"] = bad_seq_count


    def latex_csv_replace(value):
        return str(value).replace("_", "-").replace(",", ";") if value is not None else None


    # CWE-wise preparation time with sequence count and max sequence length
    csv_file_path = os.path.join(summary_dir_path, "summarypreptimeseqcountseqlength.csv")
    csv_headings = ["cwes", "cwe_juliet_names", "cwe_juliet_names_pretty", "cwe_names", "abs_path",
                    "testcase_count_overall", "testcase_count_win",
                    "testcase_overall_skipped", "testcase_win_skipped",
                    "parse_0_testcase_skip", "parse_0_testcase_overall",
                    "parse_1_testcase_skip", "parse_1_testcase_overall",
                    "sequence_count_pre_filter", "sequence_good_count", "sequence_bad_count", "sequence_count",
                    "preparation_time_in_sec",
                    "sequence_len_min", "sequence_len_max",
                    "sequence_len_mean", "sequence_len_median", "sequence_len_std", "sequence_len_var",
                    "seq_count_mult_seq_len_mean"]
    found_cwes = []
    filter = lambda x: True  # lambda row_data: int(row_data["label_count"]) == 2
    with open(csv_file_path, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([latex_csv_replace(csv_heading) for csv_heading in csv_headings])
        for train_result_data in train_result_data_list:
            if train_result_data["cwes"] in found_cwes:
                print(" - Skipping already found cwe", train_result_data["cwes"], train_result_data)
                continue
            else:
                found_cwes.append(train_result_data["cwes"])
            if filter(train_result_data):
                writer.writerow([latex_csv_replace(train_result_data.get(key, None)) for key in csv_headings])
    print("Wrote", csv_file_path)

    # CWE-wise training results
    csv_file_path = os.path.join(summary_dir_path, "summarytrainresults.csv")
    csv_headings = ["cwes", "cwe_juliet_names", "cwe_juliet_names_pretty", "cwe_names", "abs_path",
                    "testcase_count_overall", "testcase_count_win",
                    "testcase_overall_skipped", "testcase_win_skipped",
                    "parse_0_testcase_skip", "parse_0_testcase_overall",
                    "parse_1_testcase_skip", "parse_1_testcase_overall",
                    "sequence_count_pre_filter", "preparation_time_in_sec", "sequence_count",
                    "sequence_len_min", "sequence_len_max",
                    "sequence_len_mean", "sequence_len_median", "sequence_len_std", "sequence_len_var",
                    "seq_count_mult_seq_len_mean",
                    "pc0net_type", "pc1net_type", "pc0nalu_neuron_count", "pc0nalu_nac_only",
                    "pc0main_lstm_neuron_count", "pc0dense_neuron_count",
                    "pc0best_val_F1",
                    "pc0best_val_F1_weighted_score", "pc0best_test_F1_weighted_score",
                    "pc0best_val_F1_macro_score", "pc0best_test_F1_macro_score",
                    "pc0val_F1_weighted_score_rep0", "pc0val_F1_weighted_score_rep1",
                    "pc0val_F1_weighted_score_rep2",
                    "pc0test_F1_weighted_score_rep0", "pc0test_F1_weighted_score_rep1",
                    "pc0test_F1_weighted_score_rep2",
                    "pc0test_argmax_F1_weighted_score_rep0", "pc0test_argmax_F1_weighted_score_rep1",
                    "pc0test_argmax_F1_weighted_score_rep2",
                    "pc0test_F1_macro_score_rep0", "pc0test_F1_macro_score_rep1",
                    "pc0test_F1_macro_score_rep2",
                    'pc0test_mean_reps', 'pc0test_median_reps', 'pc0test_std_reps', 'pc0test_var_reps',
                    'pc0val_mean_reps', 'pc0val_median_reps', 'pc0val_std_reps', 'pc0val_var_reps',
                    "pc0mean_reps", "pc0std_reps",
                    "pc0val_accuracy_rep0", "pc0val_accuracy_rep1", "pc0val_accuracy_rep2",
                    "pc0test_accuracy_rep0", "pc0test_accuracy_rep1", "pc0test_accuracy_rep2",
                    "pc0epoch_rep0", "pc0epoch_rep1", "pc0epoch_rep2",

                    "pc1best_val_F1",
                    "pc1best_val_F1_weighted_score", "pc1best_test_F1_weighted_score",
                    "pc1best_val_F1_macro_score", "pc1best_test_F1_macro_score",
                    "pc1val_F1_weighted_score_rep0", "pc1val_F1_weighted_score_rep1",
                    "pc1val_F1_weighted_score_rep2",
                    "pc1test_F1_weighted_score_rep0", "pc1test_F1_weighted_score_rep1",
                    "pc1test_F1_weighted_score_rep2",
                    "pc1test_argmax_F1_weighted_score_rep0", "pc1test_argmax_F1_weighted_score_rep1",
                    "pc1test_argmax_F1_weighted_score_rep2",
                    'pc1test_mean_reps', 'pc1test_median_reps', 'pc1test_std_reps', 'pc1test_var_reps',
                    'pc1val_mean_reps', 'pc1val_median_reps', 'pc1val_std_reps', 'pc1val_var_reps',
                    "pc1mean_reps", "pc1std_reps",
                    "pc1val_accuracy_rep0", "pc1val_accuracy_rep1", "pc1val_accuracy_rep2",
                    "pc1test_accuracy_rep0", "pc1test_accuracy_rep1", "pc1test_accuracy_rep2",
                    "pc1epoch_rep0", "pc1epoch_rep1", "pc1epoch_rep2",
                    ]
    extended_csv_headings = []
    for key in csv_headings:
        extended_csv_headings.append(key)
        if key.lower() != key:
            extended_csv_headings.append(key.lower())
    csv_headings = extended_csv_headings
    with open(csv_file_path, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([latex_csv_replace(csv_heading) for csv_heading in csv_headings])
        for train_result_data in train_result_data_list:
            print(train_result_data.keys())
            writer.writerow([latex_csv_replace(train_result_data.get(key, None)) for key in csv_headings])
    print("Wrote", csv_file_path)
