"""
Data Preparation component
"""
from enum import Enum

from Frontends import *

PREPARED_DATASET_PATH_ROOT = os.path.join(ENVIRONMENT.data_root_path, "PreparedDatasets")
DATASET_CUSTODIAN_PICKLE_FILE_NAME = "DatasetCustodian.pkl"
PICKLED_SEQUENCES_FILE_NAME_DIGIT_COUNT = 7


class DatasetCustodian(ABC):
    """
    Loading, holding, preparing, ... of dataset data and associated information like the frontend instance used for
    parsing
    """
    frontend: ParserFrontend = None
    project_path = None
    int_to_label_mapping: Dict[int, str] = None
    label_to_int_mapping: Dict[str, int] = None
    per_label_counts = None
    language: str = None

    # Paths to prepared data:
    pickled_samples_as_ast_nodes_and_labels_file_paths: Optional[List[str]] = None
    pickled_samples_as_sequences_and_labels_file_paths: Optional[List[str]] = None

    # Variable holding the length which all sequences will be padded to in non-eager mode (also for unknown inputs)
    actual_max_sequence_length = None

    # Config:
    max_sequence_length: int
    max_samples_to_prepare_for_training: int
    min_sample_count_per_class: int
    skip_samples_with_helper_methods: bool = None

    # Train validation test data:
    traintestval_split_indices: Optional[Dict[str, List[int]]] = None

    train_test_sample_shape: Tuple[int, int] = None
    prepared_data_dir_path: str = None
    split_info: Dict = None

    def __init__(self, language: str,
                 frontend: ParserFrontend,
                 project_path: str,
                 labels: List[str],
                 split_info: Dict,
                 max_sequence_length: int = None,
                 max_samples_to_prepare_for_training: int = None,
                 min_sample_count_per_class: int = None,
                 skip_samples_with_helper_methods: bool = False
                 ):
        self.language = language
        self.frontend = frontend
        self.project_path = project_path
        assert split_info is not None and isinstance(split_info, dict), split_info
        self.split_info = split_info
        self.create_and_set_label_mappings(labels)
        self.max_sequence_length = max_sequence_length
        self.max_samples_to_prepare_for_training = max_samples_to_prepare_for_training
        self.min_sample_count_per_class = min_sample_count_per_class
        self.skip_samples_with_helper_methods = skip_samples_with_helper_methods
        # Create string representation for this dataset instance:
        dataset_representation = repr(self) + "_" + get_timestamp()
        # dataset_representation is usually too long. Use hash at the beginning for uniqueness and add as much
        # information as possible
        dir_name = as_safe_filename(get_md5_string(dataset_representation) + "_" + dataset_representation)
        dir_name = dir_name[0:255]
        # Create directory path if necessary:
        self.prepared_data_dir_path = os.path.join(PREPARED_DATASET_PATH_ROOT, dir_name)
        if create_dir_if_necessary(self.prepared_data_dir_path):
            # Write complete representation as json:
            write_file_content(os.path.join(self.prepared_data_dir_path, "info.json"),
                               to_json(self.repr_list()))

        # Save current python file (if not run from console)
        # TODO: Copy all imported files too!
        if __file__ is not None:
            curr_file_abs_path = os.path.abspath(__file__)
            copy_file(os.path.join(os.path.dirname(curr_file_abs_path), "Main.py"), self.prepared_data_dir_path)
            copy_file(curr_file_abs_path, self.prepared_data_dir_path)

        # Also write output to file
        self.enable_tee()

        # Print/Write current time:
        print("Current date and time", get_timestamp())

    def repr_list(self):
        return [type(self).__name__, self.language, repr(self.frontend), self.get_labels(), self.split_info,
                self.project_path, self.max_sequence_length, self.max_samples_to_prepare_for_training,
                self.min_sample_count_per_class, self.skip_samples_with_helper_methods]

    def __repr__(self):
        return repr(self.repr_list())

    def enable_tee(self):
        Tee.enable(os.path.join(self.prepared_data_dir_path, "output.txt"))

    def disable_tee(self):
        Tee.disable()

    def save(self) -> None:
        """
        Saves the current state to the directory associated with this instance
        :return:
        """
        serialize_data(self, os.path.join(self.prepared_data_dir_path, DATASET_CUSTODIAN_PICKLE_FILE_NAME))

    @classmethod
    def create(cls, config: Dict) -> "DatasetCustodian":
        """
        Creates new dataset custodian from given config:
        :param config:
        :return:
        """
        if config.get("load_from", None):
            loaded_dataset_custodian = cls.load(config.pop("load_from"), config.pop("is_path", False))
            if config.pop("forget_sequences", False):
                loaded_dataset_custodian.forget_sequences()
            assert len(config) == 0, "Too many or unknown arguments:" + str(config)
            return loaded_dataset_custodian
        elif config.get("merge_from", None):
            merged_dataset_custodian = cls.create_merged(config.pop("merge_from"), **config)
            return merged_dataset_custodian
        else:
            # Get and remove type value from config:
            dataset_custodian_type = config.pop("type", None)
            assert dataset_custodian_type is not None, config
            # Instantiate custodian of type dataset_custodian_type:
            return dataset_custodian_type(**config)

    @classmethod
    def load(cls, dir_name_or_path: str, is_path: bool = False, reenable_tee: bool = True) -> "DatasetCustodian":
        """
        Loads the already existing dataset custodian specified by the given directory name
        :param dir_name_or_path:
        :param is_path: Whether dir_name_or_path is just the directory name or an absolute path
        :return:
        """
        # Support for PosixPath objects by converting it to string first:
        dir_name_or_path = str(dir_name_or_path)
        # Add default data set root path is only a directory name is provided:
        if not is_path:
            load_dir_path = os.path.join(PREPARED_DATASET_PATH_ROOT, dir_name_or_path)
        else:
            load_dir_path = dir_name_or_path
        # Load dataset custodian from disk:
        dataset_custodian: DatasetCustodian = deserialize_data(
            os.path.join(load_dir_path, DATASET_CUSTODIAN_PICKLE_FILE_NAME))
        # Check whether dataset custodian's internal path matches the path it was loaded from:
        # TODO: Do not store absolute paths inside the dataset custodian at all!
        if dataset_custodian.prepared_data_dir_path != load_dir_path:
            print("Dataset custodian has been moved/copied to other path. Changing custodian's paths temporary...")
            print("DC path:  ", dataset_custodian.prepared_data_dir_path)
            print("Load path:", load_dir_path)
            new_prepared_data_dir_path = load_dir_path
            file_path_lists_to_adjust = []
            if dataset_custodian.pickled_samples_as_ast_nodes_and_labels_file_paths is not None:
                file_path_lists_to_adjust.append(dataset_custodian.pickled_samples_as_ast_nodes_and_labels_file_paths)
            if dataset_custodian.pickled_samples_as_sequences_and_labels_file_paths is not None:
                file_path_lists_to_adjust.append(dataset_custodian.pickled_samples_as_sequences_and_labels_file_paths)
            for file_path_list_to_adjust in file_path_lists_to_adjust:
                for i in range(len(file_path_list_to_adjust)):
                    file_path_list_to_adjust[i] = file_path_list_to_adjust[i] \
                        .replace(dataset_custodian.prepared_data_dir_path, new_prepared_data_dir_path)
            dataset_custodian.prepared_data_dir_path = new_prepared_data_dir_path
            if not os.path.isdir(dataset_custodian.project_path):
                print("Project path", dataset_custodian.project_path, "does not exist!")
        if reenable_tee:
            # Re-enable Tee (new stdout and stderr because of new script execution)
            dataset_custodian.enable_tee()
        print("\n" + "-" * 100, "\nLoaded previous dataset custodian from", dir_name_or_path)
        return dataset_custodian

    @classmethod
    def create_merged(cls, dir_names_or_paths: List[str], is_path: bool = False, **kwargs):
        print("Creating merged dataset custodian ...")
        assert len(dir_names_or_paths) > 0, dir_names_or_paths
        dataset_custodian_types = []
        merged_labels = []
        merged_pickled_samples_as_ast_nodes_and_labels_file_paths = []
        languages = []
        project_paths = []
        for dir_name_or_path in dir_names_or_paths:
            dataset_custodian = cls.load(dir_name_or_path, is_path, reenable_tee=False)
            assert dataset_custodian.pickled_samples_as_sequences_and_labels_file_paths is not None
            # Sequences can not be used directly because their corresponding frontend's lookups do not match. Instead,
            # add ast_nodes such that the sequence generation is done by the one frontend which will be created later.
            # Of course, due to this, sequence generation has to be done again which may take a lot of time.

            # Do not consider data sets without samples:
            if len(dataset_custodian.pickled_samples_as_ast_nodes_and_labels_file_paths) > 0:
                merged_pickled_samples_as_ast_nodes_and_labels_file_paths \
                    .extend(dataset_custodian.pickled_samples_as_ast_nodes_and_labels_file_paths)
                if len(dataset_custodian.get_labels()) == 0:
                    # Labels were eliminated during sequence generation but they are needed here. Determine them
                    # from info json:
                    assert len(dataset_custodian.pickled_samples_as_sequences_and_labels_file_paths) == 0
                    print(os.path.join(dataset_custodian.prepared_data_dir_path, "info.json"))
                    info_data = from_json(
                        read_file_content(os.path.join(dataset_custodian.prepared_data_dir_path, "info.json")))
                    merged_labels.extend(info_data[3])
                else:
                    merged_labels.extend(dataset_custodian.get_labels())
                dataset_custodian_types.append(type(dataset_custodian))
                languages.append(dataset_custodian.language)
                project_paths.append(dataset_custodian.project_path)
                assert isinstance(dataset_custodian.frontend, ClangTooling), type(dataset_custodian.frontend)

        merged_count = len(languages)

        # Remove duplicates:
        merged_labels = list(set(merged_labels))

        # There should be no duplicate pickled paths:
        # assert len(merged_pickled_samples_as_sequences_and_labels_file_paths) \
        #        == len(list(set(merged_pickled_samples_as_sequences_and_labels_file_paths))), \
        #     merged_pickled_samples_as_sequences_and_labels_file_paths
        assert len(merged_pickled_samples_as_ast_nodes_and_labels_file_paths) \
               == len(list(set(merged_pickled_samples_as_ast_nodes_and_labels_file_paths))), \
            merged_pickled_samples_as_ast_nodes_and_labels_file_paths

        # Make sure languages and projects paths are the same
        languages = list(set(languages))
        assert len(languages) == 1, languages
        project_paths = list(set(project_paths))
        assert len(project_paths) == 1, project_paths
        dataset_custodian_types = list(set(dataset_custodian_types))
        assert len(dataset_custodian_types) == 1 and dataset_custodian_types[0] == Juliet, dataset_custodian_types

        # Create new dataset custodian:
        assert kwargs.pop("type") == Juliet, kwargs
        kwargs["language"] = languages[0]
        kwargs["label_whitelist"] = merged_labels
        merged_dataset_custodian = Juliet(**kwargs)
        # Add all sample paths:
        merged_dataset_custodian.pickled_samples_as_ast_nodes_and_labels_file_paths \
            = merged_pickled_samples_as_ast_nodes_and_labels_file_paths

        # Save dataset custodian
        merged_dataset_custodian.save()

        print("Created merged dataset from", merged_count, "other datasets.")

        return merged_dataset_custodian

    def forget_sequences(self):
        # TODO: Remove files?
        self.pickled_samples_as_sequences_and_labels_file_paths = None
        self.actual_max_sequence_length = None
        self.traintestval_split_indices = None
        print("Forgot sequences.")

    def get_pickled_sequences_dir_name(self):
        return os.path.join(self.prepared_data_dir_path, "FeatureSequences")

    def get_sample_count(self) -> int:
        return sum(self.per_label_counts.values())

    def get_sample_shape(self) -> Tuple:
        return self.train_test_sample_shape

    @classmethod
    def create_label_mappings(cls, labels):
        """
        Creates a two-way-label-mapping between the given labels and its indices
        :param labels:
        :return:
        """
        # Not sure if OrderedDict is really necessary, because keys() and values() are used together
        # but it is nice to have keys() and values() in ascending order anyways
        label_to_int_mapping, int_to_label_mapping = OrderedDict(), OrderedDict()
        for label in labels:
            int_repr = len(label_to_int_mapping)
            label_to_int_mapping[label] = int_repr
            int_to_label_mapping[int_repr] = label
        return label_to_int_mapping, int_to_label_mapping

    def create_and_set_label_mappings(self, labels):
        """
        Creates and sets the two-way-label-mapping between the given labels and its indices of the current instance
        :param labels:
        :return:
        """
        self.label_to_int_mapping, self.int_to_label_mapping = self.create_label_mappings(labels)

    def get_labels(self) -> List[str]:
        """
        Returns all possible labels
        :return:
        """
        return list(self.label_to_int_mapping.keys())

    def get_label_ints(self) -> List[int]:
        """
        Returns label numbers of all possible labels
        :return:
        """
        label_ints = list(self.label_to_int_mapping.values())
        assert list(range(
            len(label_ints))) == label_ints, "Expected label ints to be an increasing-by-one sequence."
        return label_ints

    def get_no_bug_label(self) -> str:
        return self.get_labels()[-1]

    def labels_to_label_ints_as_array(self, labels: List[str]):  # -> ndarray
        """
        Converts list of labels to array of label ints using the label mapping
        :param labels:
        :return:
        """
        labels_numbers = [self.label_to_int_mapping[label] for label in labels]
        return to_numpy_array(labels_numbers)

    @abc.abstractmethod
    def gen_sample_nodes_label_pairs(self) -> Tuple[List[ParserFrontend.AstNodeType], str]:
        """
        Yields pairs of ast node list and label, where ast node list is a list of method root nodes
        :return:
        """
        pass

    def gen_ast_nodes_of_relevant_methods_from_testcases_source_file_paths(self,
                                                                           source_file_path_collections_with_context:
                                                                           Dict[str, typing.Union[List[str], Dict]],
                                                                           relevant_method_name_to_label_mapping:
                                                                           Dict[str, List[str]] = None,
                                                                           verbosity=2) \
            -> Tuple[List[ParserFrontend.AstNodeType], str]:
        """
        Wrapper around gen_ast_nodes_of_relevant_methods_from_testcases_asts
        :param source_file_path_collections_with_context: For each test case a collection of associated source files
        :param relevant_method_name_to_label_mapping: 
        :param verbosity: 
        :return: 
        """
        # Parse files. The result is for each test case a list of asts of the corresponding test case files
        testcases_asts = self.frontend.get_asts_from_source_path_collections(source_file_path_collections_with_context)
        yield from self.gen_ast_nodes_of_relevant_methods_from_testcases_asts(testcases_asts,
                                                                              relevant_method_name_to_label_mapping,
                                                                              verbosity)

    def gen_ast_nodes_of_relevant_methods_from_testcases_asts(self,
                                                              testcases_asts: List[List[ParserFrontend.AstNodeType]],
                                                              relevant_method_name_to_label_mapping: Dict[
                                                                  str, List[str]] = None,
                                                              verbosity=2) \
            -> Tuple[List[ParserFrontend.AstNodeType], str]:
        """
        Extracts the relevant ast nodes from each test case's asts. For each method which is considered relevant based
        on relevant_method_name_to_label_mapping, one pair is yielded. The first element of such a pair is a list
        of relevant body ast nodes of that method (and its callees). The second element is the corresponding label which
        that method was associated with based on relevant_method_name_to_label_mapping.
        There may be multiple relevant methods per test case (e.g. for Juliet usually one bad and two good methods are
        present and considered relevant). For MemNet there is only one ast per test case and exactly one method is
        considered relevant per test case.
        :param verbosity:
        :param testcases_asts: For each test case a list of asts. A test case may have multiple associated asts because
        there may be multiple files associated to that test case
        :param relevant_method_name_to_label_mapping:
        :return:
        """
        if relevant_method_name_to_label_mapping is None:
            # Default is all methods with "no bug" label
            relevant_method_name_to_label_mapping = {self.get_no_bug_label(): [r".*"]}
            if verbosity >= 2:
                print("No relevant-method-name-to-label mapping: All methods are considered as no-bug:",
                      relevant_method_name_to_label_mapping)
        skipped = 0
        # For each testcase there is a list of asts (because there may be multiple files belonging to one test case)
        for asts in testcases_asts:
            if None in asts:
                # print("Skipped content because of parsing error")
                skipped += 1
                continue

            entry_method_def_nodes_with_label = []

            # Collecting all methods that are in the ASTs of the current test case:
            all_methods_in_content = []
            for ast in asts:
                for node, depth, parent in self.frontend.depth_search_tree(ast):
                    # Look for method definitions only:
                    if self.frontend.is_method_definition_node(node):
                        # Get semantic name (see add_called_methods_to_list for information why only semantic part of
                        # name must be used):
                        method_semantic_name = self.frontend.get_semantic_name(self.frontend.get_method_name(node))
                        assert method_semantic_name not in all_methods_in_content, (method_semantic_name,
                                                                                    all_methods_in_content)
                        all_methods_in_content.append((node, method_semantic_name))
                    else:
                        # TODO: other node types?
                        pass

            # For each weakness check for each of its regexp whether one of the collected methods match:
            # Each regexp describes a method which is an source code snippets entry point. E.g. good or bad
            for corresponding_label, relevant_method_name_regexps in relevant_method_name_to_label_mapping.items():
                entry_point_found = False
                for relevant_method_regexp in relevant_method_name_regexps:
                    for method_node, method_semantic_name in all_methods_in_content:
                        if re.search(relevant_method_regexp, method_semantic_name) is not None:
                            # method_node matches
                            entry_method_def_nodes_with_label.append((method_node, corresponding_label))
                            # print(method_name)
                            entry_point_found = True
                    if entry_point_found:
                        # Do not try to add methods with other regexps of this label
                        break
                if not entry_point_found:
                    assert "Did not found entry point method for label", (corresponding_label, all_methods_in_content)

            # For each entry point method determine its callee methods and order them. Then, for each method, extract
            # the relevant method body ast nodes:
            for entry_method_def_node, label in entry_method_def_nodes_with_label:
                called_methods = []
                # Add called methods recursively (without entry_method_def_node itself):
                self.frontend.add_called_methods_to_list(entry_method_def_node, called_methods, all_methods_in_content)
                if self.skip_samples_with_helper_methods and len(called_methods) > 0:
                    if verbosity >= 1:
                        print("Skipped method def with label " + label + " because it (indirectly) calls " + str(
                            len(called_methods)) + " other in-content methods",
                              self.frontend.get_node_as_string(entry_method_def_node),
                              [self.frontend.get_node_as_string(called_method) for called_method in called_methods])
                    continue
                # Methods, which are called last, should be seen first, such that a called method has always been
                # seen already
                entry_plus_called_method_def_nodes = list(reversed(called_methods)) + [entry_method_def_node]

                # add up all sub nodes of entry_method_def_node and of called method def nodes
                methods_body_nodes: List[ParserFrontend.AstNodeType] = []
                for method_def_node in entry_plus_called_method_def_nodes:
                    # Skip methods that are not defined in the data set (i.e. included from e.g. standard library)
                    if self.frontend.is_included_or_imported_node(method_def_node, self.project_path):
                        # print("Skipped node with location", method_def_node.location, method_def_node)
                        continue
                    # Add the method node's body nodes only (and not its parameters and the method node itself)
                    methods_body_nodes.extend(
                        self.frontend.get_relevant_nodes_from_method_def_node(method_def_node))
                if len(methods_body_nodes) == 0 and verbosity >= 1:
                    print("No body nodes for entry point", entry_method_def_node, called_methods)
                yield methods_body_nodes, label
        if verbosity >= 1:
            print("Skipped", skipped, "of", len(testcases_asts), "AST collections because of parsing error(s)")

    def gen_sequences_from_prepared_samples_ast_nodes(self) \
            -> Tuple[np.ndarray, str, List[ParserFrontend.AstNodeType]]:  # 2D ndarray
        for pickle_file_path in self.pickled_samples_as_ast_nodes_and_labels_file_paths:
            list_of_sample_related_nodes_lists, labels = deserialize_data(pickle_file_path)
            yield from self.gen_sequences_from_samples_ast_nodes(list_of_sample_related_nodes_lists, labels)

    def gen_sequences_from_samples_ast_nodes(self,
                                             list_of_sample_related_nodes_lists: List[
                                                 List[ParserFrontend.AstNodeType]],
                                             labels: List[str]) \
            -> Tuple[np.ndarray, str, List[ParserFrontend.AstNodeType]]:  # 2D ndarray
        """
        Computes the sequence of feature vectors for each sample's list of ast nodes
        :param list_of_sample_related_nodes_lists:
        :param labels:
        :return:
        """
        # print("Generating " + str(
        #    len(list_of_sample_related_nodes_lists)) + " sequences of feature vectors out of lists of AST nodes ...")
        # sample_related_nodes_list is a list of top level ast nodes
        # corresponding_ast_nodes ist a list of all ast nodes (even nested ones)
        for sample_related_nodes_list, label in zip(list_of_sample_related_nodes_lists, labels):
            sequence_of_feat_vec_as_array, corresponding_ast_nodes = \
                self.frontend.get_sequence_as_array_from_a_sampleis_node_list(sample_related_nodes_list)
            if sequence_of_feat_vec_as_array is not None:  # Skip cases where no node has a feature vector
                yield sequence_of_feat_vec_as_array, label, corresponding_ast_nodes

    def prepare(self) -> None:
        """
        Performs the data preparation according to the data set's info. This includes reading or generating pairs of
        source code and label, parsing of the source code, extraction of relevant nodes from the ast, and splitting
        into train, validation, and test sets.
        The results are managed by the dataset custodian. gen_sequence_tuples_for_tensorflow or
        get_sequence_tuple_for_tensorflow_from_index can be called afterwards to retrieve the prepared sequences.
        :return:
        """
        # Every x samples
        status_every = 500
        save_every = 5000

        if self.pickled_samples_as_ast_nodes_and_labels_file_paths is None:
            print("Getting samples as lists of AST nodes ...")
            samples_as_ast_nodes_and_labels_file_path = os.path.join(self.prepared_data_dir_path,
                                                                     "Samples_as_AST_nodes_with_labels_{:03d}.pkl")
            start_time = time.time()
            last_status_time = start_time
            overall_count = 0
            # List of all pickle files with list of sample related ast nodes (which is a list if body-nodes for each
            # sample)
            self.pickled_samples_as_ast_nodes_and_labels_file_paths: List[str] = []
            # list of sample related ast nodes (which is a list if body-nodes for each sample)
            list_of_sample_related_nodes_lists: List[List[ParserFrontend.AstNodeType]] = []
            labels: List[str] = []
            generator_obj = self.gen_sample_nodes_label_pairs()
            stop = False
            while not stop:
                sample_related_nodes_list, label = next(generator_obj, (None, None))
                if sample_related_nodes_list is None and label is None:
                    stop = True
                else:
                    list_of_sample_related_nodes_lists.append(sample_related_nodes_list)
                    labels.append(label)
                    overall_count += 1

                # Statistics:
                curr_count = len(list_of_sample_related_nodes_lists)
                # Print Progress:
                if curr_count % status_every == 0:
                    curr_time = time.time()
                    print(
                        "{} samples since last save; {} in total; {:.2f} samples/s over last {}; {:.2f} samples/s "
                        "over all ...".format(
                            curr_count,
                            overall_count,
                            status_every / (curr_time - last_status_time),
                            status_every,
                            overall_count / (curr_time - start_time),
                        ))
                    last_status_time = curr_time

                # Save progress if requested
                if stop or curr_count % save_every == 0:
                    # Add current count to file name (at {} position):
                    file_path = samples_as_ast_nodes_and_labels_file_path.format(
                        len(self.pickled_samples_as_ast_nodes_and_labels_file_paths))
                    # Create pickable nodes:
                    # print("Making AST nodes pickable ...")
                    # pickable_list_of_sample_related_nodes_lists = [[self.frontend.get_node_as_pickable(node)
                    #                                                 for node in sample_related_nodes_list]
                    #                                                for sample_related_nodes_list
                    #                                                in list_of_sample_related_nodes_lists]
                    serialize_data((list_of_sample_related_nodes_lists, labels), file_path)
                    self.pickled_samples_as_ast_nodes_and_labels_file_paths.append(file_path)
                    # # Delete saved data from memory:
                    # del pickable_list_of_sample_related_nodes_lists
                    # Reset lists:
                    list_of_sample_related_nodes_lists = []
                    labels = []
            print("Got and serialized", overall_count, "samples as lists of AST nodes in " + str(
                int(time.time() - start_time)) + "s to\n",
                  "\n".join(self.pickled_samples_as_ast_nodes_and_labels_file_paths))
            self.save()

        status_every = 2500
        # This must be one, i.e. one sequence-label-ast-nodes tuple per file, because the downstream pipeline parts rely
        # on this.
        save_every = 1
        assert save_every == 1

        # Make sequences of feature vectors out of the AST nodes if necessary:
        if self.pickled_samples_as_sequences_and_labels_file_paths is None:
            # List of all picke files with list of sample sequences
            self.pickled_samples_as_sequences_and_labels_file_paths: List[str] = []

            assert self.actual_max_sequence_length is None, \
                "Samples have already been prepared for training. This can not be done twice for a dataset custodian instance"

            pickled_sequences_dir_name = self.get_pickled_sequences_dir_name()
            create_dir_if_necessary(pickled_sequences_dir_name)

            samples_as_sequences_and_labels_file_path = os.path.join(pickled_sequences_dir_name,
                                                                     "{:0"
                                                                     + str(PICKLED_SEQUENCES_FILE_NAME_DIGIT_COUNT)
                                                                     + "d}")

            sequences, labels, list_of_corresponding_ast_nodes = [], [], []
            self.per_label_counts = {label: 0 for label in self.get_labels()}

            print("Preparing AST node lists for training ...")
            start_time = time.time()
            last_status_time = start_time
            # Histogram of sample length (which correspond to sample's ast node count)
            hist_of_lengths = {}
            overall_count = 0
            skip_count = 0
            feat_vec_length = None
            generator_obj = self.gen_sequences_from_prepared_samples_ast_nodes()
            stop = False
            while not stop:
                sequence, label, corresponding_ast_nodes = next(generator_obj, (None, None, None))
                if sequence is None and label is None:
                    stop = True
                else:
                    # Info: sequence.shape = (ast node count, feature vector length)
                    sequence_length = sequence.shape[0]  # corresponds to sequence's ast node count
                    # Remember feat_vec_length to set the sequence shape later
                    if feat_vec_length is not None:
                        # All sequences' feature vector length must match:
                        assert feat_vec_length == sequence.shape[1], \
                            "Inconsistent feature vector length across sequences " \
                            + str(feat_vec_length) + str(sequence.shape)
                    else:
                        feat_vec_length = sequence.shape[1]
                    # Update histogram of lengths: Set count to 0 if there was no count before and increment it:
                    hist_of_lengths[sequence_length] = hist_of_lengths.setdefault(sequence_length, 0) + 1
                    # Has this sample an allowed length?
                    if self.max_sequence_length is None or sequence_length <= self.max_sequence_length:
                        sequences.append(sequence)
                        labels.append(label)
                        list_of_corresponding_ast_nodes.append(corresponding_ast_nodes)
                        self.per_label_counts[label] += 1
                        overall_count += 1
                    else:
                        skip_count += 1
                        # print("Skipped sample with sequence of length " + str(sample.shape[0]) + " (max "
                        #       + str(MAX_SEQUENCE_LENGTH) + " allowed)")
                        continue

                if self.max_samples_to_prepare_for_training is not None \
                        and overall_count >= self.max_samples_to_prepare_for_training:
                    print("Max sequence count reached.")
                    stop = True

                # Print Progress:
                curr_count = len(sequences)
                if overall_count % status_every == 0:
                    curr_time = time.time()
                    print(
                        "{} sequences since last save; {} in total; {:.2f} sequences/s over last {}; {:.2f} "
                        "sequences/s over all ...".format(
                            curr_count,
                            overall_count,
                            status_every / (curr_time - last_status_time),
                            status_every,
                            overall_count / (curr_time - start_time),
                        ))
                    last_status_time = curr_time

                # Save progress if requested and if there is at least one new sample since last save:
                if (stop or curr_count % save_every == 0) and len(sequences) > 0:
                    # Add current count to file name (at {} position):
                    file_path = samples_as_sequences_and_labels_file_path.format(
                        len(self.pickled_samples_as_sequences_and_labels_file_paths))
                    assert len(sequences) == 1, sequences
                    # Make sure that each sequence's datatype is float32 and that there are no NaNs and/or infs
                    for sequence, corresponding_ast_nodes in zip(sequences, list_of_corresponding_ast_nodes):
                        assert sequence.dtype == np.float32, sequence.dtype
                        if check_and_make_finite(sequence):
                            # Display the location where the infinite number occured:
                            print(" ", ParserFrontend.get_ast_nodes_location_summary_string(corresponding_ast_nodes,
                                                                                            separator="\n  "))
                    # Write data:
                    serialize_data((sequences, labels, list_of_corresponding_ast_nodes), file_path,
                                   verbose=save_every > 100)
                    self.pickled_samples_as_sequences_and_labels_file_paths.append(file_path)
                    # Delete saved data from memory: Reset lists:
                    sequences, labels, list_of_corresponding_ast_nodes = [], [], []

            print(overall_count, "samples found", "" if self.max_sequence_length is None else
            "with length <= max_sequence_length.")

            # Histogram of sample length:
            print_histogram(hist_of_lengths, 30, 70, "sample lengths")

            # More sample length statistics:
            all_lengths = [length for length, count in hist_of_lengths.items() for _ in range(count)]
            sample_length_stats = {"min": np.min(all_lengths),
                                   "max": np.max(all_lengths),
                                   "mean": np.mean(all_lengths),
                                   "median": np.median(all_lengths),
                                   "std": np.std(all_lengths),
                                   "var": np.var(all_lengths)
                                   }
            print("Sample length statistics:", sample_length_stats)

            if self.max_sequence_length is not None:
                print(skip_count, "samples skipped because of sequence length greater than", self.max_sequence_length)
                # Greatest length that is smaller or equal than max_sequence_length:
                self.actual_max_sequence_length = max([key for key in hist_of_lengths.keys()
                                                       if key <= self.max_sequence_length])
            else:
                self.actual_max_sequence_length = max(hist_of_lengths.keys())
            print("Determined max sequence length " + str(self.actual_max_sequence_length) + " from samples")
            # Store the samples shape:
            assert feat_vec_length is not None
            self.train_test_sample_shape = (self.actual_max_sequence_length, feat_vec_length)

            if self.min_sample_count_per_class is not None:
                # Prepare to remove classes with less than min_sample_count_per_class samples:
                labels_to_use = []
                labels_not_to_use = []
                for label in self.get_labels():
                    if self.per_label_counts[label] >= self.min_sample_count_per_class:
                        labels_to_use.append(label)
                    else:
                        # Remember removed classes and corresponding per label count for printing/statistic:
                        labels_not_to_use.append((label, self.per_label_counts[label]))
                        # Remove too small class:
                        self.per_label_counts.pop(label)
                # Create and set new updated label mappings based on the reduced labels
                self.create_and_set_label_mappings(labels_to_use)

                # check whether all lengths match
                assert self.get_labels() == labels_to_use, "Internal error while updating label mapping"
                assert list(self.per_label_counts.keys()) == self.get_labels(), \
                    "per_label_count's keys should match label mapping's keys"

                to_filter_count = overall_count - self.get_sample_count()
                if to_filter_count > 0:
                    print("Ignoring", to_filter_count, "sequences of the following", len(labels_not_to_use),
                          "classes which have less than", self.min_sample_count_per_class, "samples:",
                          labels_not_to_use)
                    filtered_pickled_samples_as_sequences_and_labels_file_paths = []
                    checked_count = 0
                    removed_count = 0
                    for pickle_file_path in self.pickled_samples_as_sequences_and_labels_file_paths:
                        _, labels, _ = deserialize_data(pickle_file_path, verbose=False)
                        assert len(labels) == 1, (labels, pickle_file_path)
                        if labels[0] not in labels_to_use:
                            # Remove sample:
                            remove_file(pickle_file_path, verbose=False)
                            removed_count += 1
                        else:
                            filtered_pickled_samples_as_sequences_and_labels_file_paths.append(pickle_file_path)
                            if checked_count % 100 == 0:
                                print("Checked {}/{} sequences; removed {}/{} sequences ..."
                                      .format(checked_count,
                                              len(self.pickled_samples_as_sequences_and_labels_file_paths),
                                              removed_count,
                                              to_filter_count
                                              )
                                      )
                        checked_count += 1
                    assert removed_count == to_filter_count
                    assert removed_count == (len(self.pickled_samples_as_sequences_and_labels_file_paths)
                                             - len(filtered_pickled_samples_as_sequences_and_labels_file_paths))
                    assert len(filtered_pickled_samples_as_sequences_and_labels_file_paths) == self.get_sample_count()
                    print("Ignored", removed_count, "sequences of too small classes.")

                    self.pickled_samples_as_sequences_and_labels_file_paths \
                        = filtered_pickled_samples_as_sequences_and_labels_file_paths
                else:
                    print("No too small classes.")
            else:
                # Use all classes:
                labels_to_use = self.get_labels()

            print("Using", self.get_sample_count(), "sequences of the following", len(labels_to_use),
                  "classes:", self.per_label_counts)

            # Save sequences paths as part of the custodian:
            self.save()

            print("Prepared and serialized sequences in", int(time.time() - start_time), "s",
                  ("to " + str(self.pickled_samples_as_sequences_and_labels_file_paths))
                  if len(self.pickled_samples_as_sequences_and_labels_file_paths) < 20
                  else "")

        # Split set of sequences in train, validation, test sets:
        self.split_into_train_test_val(self.split_info["test_fraction"],
                                       self.split_info.get("validation_fraction", 0.0),
                                       self.split_info.get("shuffle", True))

    def get_sequence_tuple_for_tensorflow_from_index(self, index) \
            -> Tuple[np.ndarray, int, np.ndarray, List[ParserFrontend.AstNodeType]]:
        """
        Returns the sequence tuple for the use in tensorflow for the prepared sequence described by the given index
        :param index: Index into the data set's list of prepared sequences
        :return:
        """
        assert 0 <= index < len(self.pickled_samples_as_sequences_and_labels_file_paths), \
            (index, self.pickled_samples_as_sequences_and_labels_file_paths)
        # sequence_file_path = os.path.join(self.get_pickled_sequences_dir_name(),
        #                                   str(index).zfill(PICKLED_SEQUENCES_FILE_NAME_DIGIT_COUNT))
        sequence_file_path = self.pickled_samples_as_sequences_and_labels_file_paths[index]
        sequences, labels, list_of_corresponding_ast_nodes = deserialize_data(sequence_file_path, verbose=False)
        assert len(sequences) == 1, (sequences, index)
        return self.get_sequence_tuple_for_tensorflow(sequences[0], labels[0], list_of_corresponding_ast_nodes[0])

    def get_sequence_tuple_for_tensorflow(self, sequence: np.ndarray, label: str,
                                          corresponding_ast_nodes: List[ParserFrontend.AstNodeType]) \
            -> Tuple[np.ndarray, int, np.ndarray, List[ParserFrontend.AstNodeType]]:
        """
        Normalizes and checks the given sequence and label such that is can be feed to the ANN using tensorflow.
        E.g., make a onehot encoded int vector out of the label string. The corresponding ast nodes are just passed
        through.
        :param sequence:
        :param label:
        :param corresponding_ast_nodes:
        :return:
        """
        assert sequence.shape != (0,), (sequence, sequence.shape)
        assert label in self.get_labels(), (label, self.get_labels())

        # # Pad sequence:
        # sequence = pad_sequences([sequence],
        #                          maxlen=self.actual_max_sequence_length,
        #                          dtype=np.float32
        #                          )[0]
        # Label to int:
        label_int = self.label_to_int_mapping[label]
        # Label onehot:
        label_onehot = int_to_onehot(label_int, len(self.get_labels()))

        return sequence, label_int, label_onehot, corresponding_ast_nodes

    def gen_sequence_tuples_for_tensorflow(self) \
            -> Tuple[np.ndarray, int, np.ndarray, List[ParserFrontend.AstNodeType]]:
        """
        Yields all sequences of this data set for the use in tensorflow
        :return:
        """
        assert self.pickled_samples_as_sequences_and_labels_file_paths is not None, "Please produce samples first"
        # Pickled data is available:
        yielded_sample_count = 0
        # empty_skip_count = 0
        # labels_to_use = self.get_labels()
        assert self.get_sample_count() == sum(self.per_label_counts.values())
        # Do not print "Deserialized..." messages when there are a lot of files:
        verbose = len(self.pickled_samples_as_sequences_and_labels_file_paths) < 100
        for pickle_file_path in self.pickled_samples_as_sequences_and_labels_file_paths:
            samples, labels, list_of_corresponding_ast_nodes = deserialize_data(pickle_file_path, verbose=verbose)

            for sample, label, corresponding_ast_nodes in zip(samples, labels, list_of_corresponding_ast_nodes):
                prepared_sample_tuple = self.get_sequence_tuple_for_tensorflow(sample, label, corresponding_ast_nodes)
                if prepared_sample_tuple is None:
                    continue
                yielded_sample_count += 1
                yield prepared_sample_tuple

        assert self.get_sample_count() == yielded_sample_count, "per_label_counts should sum up to samples count " + \
                                                                str(self.get_sample_count()) + ", " + \
                                                                str(yielded_sample_count)
        print("Yielded", yielded_sample_count, "samples.")

    def gen_sequence_tuples_for_tensorflow_of_unknown_source_content(self,
                                                                     source_code: str,
                                                                     extension: str,
                                                                     relevant_method_name_to_label_mapping:
                                                                     Dict[str, List[str]] = None,
                                                                     **kwargs):
        assert extension[0] == ".", extension
        # Generate file from input:
        with tempfile.NamedTemporaryFile("w", dir=self.project_path, suffix=extension) as temp_input_file:
            temp_input_file.write(source_code)
            temp_input_file.flush()
            for sequence_tuple in self.gen_sequence_tuples_for_tensorflow_of_unknown_source_file(
                    temp_input_file.name, relevant_method_name_to_label_mapping, **kwargs):
                for ast_node in sequence_tuple[3]:
                    self.frontend.add_source_code_to_ast(ast_node, source_code, extension, temp_input_file.name)
                yield sequence_tuple

    def gen_sequence_tuples_for_tensorflow_of_unknown_source_file(self,
                                                                  file_path,
                                                                  relevant_method_name_to_label_mapping:
                                                                  Dict[str, List[str]] = None,
                                                                  needs_windows=False,
                                                                  verbosity=2):
        if not file_path.startswith(self.project_path):
            print("Warning: File is not in project.")

        source_file_path_collections_with_context = {"source_path_collections": [[file_path]],
                                                     "context": {"needs_windows": needs_windows}}
        # content_as_nodes_list is a list of top level ast nodes
        # corresponding_ast_nodes is a list of all ast nodes (even nested ones)
        for content_as_nodes_list, label in \
                self.gen_ast_nodes_of_relevant_methods_from_testcases_source_file_paths(
                    source_file_path_collections_with_context,
                    relevant_method_name_to_label_mapping=
                    relevant_method_name_to_label_mapping,
                    verbosity=verbosity):
            sequence_of_feat_vec_as_array, corresponding_ast_nodes = \
                self.frontend.get_sequence_as_array_from_a_sampleis_node_list(content_as_nodes_list)
            # Check if there is a feature vector for at least one node
            if sequence_of_feat_vec_as_array is not None:
                yield self.get_sequence_tuple_for_tensorflow(sequence_of_feat_vec_as_array, label,
                                                             corresponding_ast_nodes)

    def split_into_train_test_val(self, test_fraction, validation_fraction, shuffle, force_resplit=False):
        if self.traintestval_split_indices is None or force_resplit:
            assert 0 <= test_fraction <= 1 and 0 <= validation_fraction <= 1 \
                   and test_fraction + validation_fraction <= 1, (test_fraction, validation_fraction)
            sequence_count = len(self.pickled_samples_as_sequences_and_labels_file_paths)
            indices = list(range(sequence_count))
            if shuffle:
                random.shuffle(indices)
            # first test, than, train, than val (WARNING: When changes are done here, Memnet's split_info=="paper"
            # must be changed accordingly!)
            splits = np.split(np.array(indices),
                              [int(test_fraction * sequence_count),
                               int((1 - validation_fraction) * sequence_count)])
            self.traintestval_split_indices = {
                "train": splits[1],
                "test": splits[0],
                "validation": splits[2]
            }

            # for split_name in ("train", "validation", "test"):
            #     for index in (0, -1):
            #         sequence_file_path = self.pickled_samples_as_sequences_and_labels_file_paths[self.traintestval_split_indices[split_name][index]]
            #         sequences, labels, list_of_corresponding_ast_nodes = deserialize_data(sequence_file_path, verbose=False)
            #         print(split_name, index, labels, list_of_corresponding_ast_nodes[0][0].tu.id.spelling[1])
            # print(self.traintestval_split_indices)

            # Save split as part of the custodian:
            self.save()

            print("Split", sequence_count, "sequences.", end="")
        else:
            print("Warning: Sequences have already been split! Use force_resplit to force splitting.", end="")
        print(" Split:", {s_name: len(s_indices) for s_name, s_indices in self.traintestval_split_indices.items()})

    def get_class_weights(self):
        """
        Returns list of class weights >= 1.0 in the order specified by label mapping
        :return:
        """
        if len(self.get_labels()) == 0:
            return []
        else:
            max_class_size = max(self.per_label_counts.values())
            assert list(self.int_to_label_mapping.keys()) == list(range(len(self.get_labels())))
            class_weights = [max_class_size / self.per_label_counts[label] for label in
                             self.int_to_label_mapping.values()]
            no_bug_index = self.label_to_int_mapping[self.get_no_bug_label()]
            assert no_bug_index == len(class_weights) - 1, (no_bug_index, class_weights)
            # Ensure high weights for the no bug class to prevent false positives
            class_weights[no_bug_index] = max(class_weights)
            if len(set(class_weights)) == 1:
                # If all weights are the same (e,g, in binary classificatio), still ensure higher weight for
                #  non-vulnerable class:
                class_weights[no_bug_index] *= 2
            return class_weights

    def print_dataset_statistics(self):
        """
        Prints some data set statistics:
        :return:
        """
        print("Label count:", len(self.get_labels()))
        print("Counts per label:", sorted(self.per_label_counts.items(), key=lambda kv: kv[1], reverse=True))
        print("Sequence count:", self.get_sample_count())
        print("Sequence shape:", self.get_sample_shape())
        print("Class weights:", {label: weight for weight, label in
                                 zip(self.get_class_weights(), self.int_to_label_mapping.values())})

        feat_vec_infos = self.frontend.get_feat_vec_component_info()
        # Make sure that the feature vector size and feat vec info matches:
        assert feat_vec_infos[-1].index_end == self.get_sample_shape()[1], \
            ("Dataset feat vec size does not match feat vec info!", feat_vec_infos, self.get_sample_shape())
        # Print feat vec component info:
        for feat_vec_info in feat_vec_infos:
            print(feat_vec_info)

    def print_first_samples(self, to_print_count: int = 2, omit_padding=False):
        """
        Prints first few samples (e.g. for debugging)
        :param to_print_count:
        :param omit_padding:
        :return:
        """
        for sequence, label_int, label_onehot, corresponding_ast_nodes in self.gen_sequence_tuples_for_tensorflow():
            print("Label number:", label_int, "Label onehot:", label_onehot,
                  "Sequence shape:", sequence.shape,
                  "Sequence with corresponding AST nodes" + ("without padding" if omit_padding else ""),
                  "Path:", ParserFrontend.get_ast_nodes_location_summary_string(corresponding_ast_nodes,
                                                                                separator="; ") + ":")
            # Number of sequence items that are padding
            padded_row_count = sequence.shape[0] - len(corresponding_ast_nodes)
            for index in range(padded_row_count if omit_padding else 0, sequence.shape[0]):
                print(array_to_string(sequence[index]),
                      self.frontend.get_node_as_string(corresponding_ast_nodes[index - padded_row_count])
                      if index >= padded_row_count
                      else "<padded>")
            to_print_count -= 1
            if to_print_count == 0:
                break


class ArrDeclAccess(DatasetCustodian):
    # Which kinds should be generated
    kinds: List[str]
    sample_count: int
    lower_limit: int
    upper_limit: int
    every_ith: int

    # # Cache for test data sets:
    # test_set_cache = None
    #
    # # Static/constant attributes:
    # SAMPLES_DIR_NAME = "samples"
    # LABELS_DIR_NAME = "labels"

    def __init__(self, language, kinds, sample_count, lower_limit, upper_limit, every_ith=1, **kwargs):
        # Currently only ClangTooling (i.e. C++ or C) is supported
        assert language in ["C++", "C"], language
        self.test_set_cache = {}
        project_path = os.path.join(ENVIRONMENT.data_root_path, "ArrDeclAccess")  # parent path for generated samples
        frontend = ClangTooling(project_path=project_path)

        assert len(kinds) > 0, kinds
        self.kinds = kinds
        self.sample_count = sample_count
        assert lower_limit >= 0, lower_limit
        self.lower_limit = lower_limit
        assert upper_limit > lower_limit, str(lower_limit) + ", " + str(upper_limit)
        self.upper_limit = upper_limit
        assert 1 <= every_ith < (self.upper_limit - self.lower_limit), (every_ith, self)
        self.every_ith = every_ith

        # Must come last (because use of __repr__)
        super().__init__(language,
                         frontend,
                         project_path,  # parent path for generated samples
                         ["faulty", "correct"],
                         **kwargs
                         )

    def repr_list(self):
        return super().repr_list() + [self.kinds, self.sample_count, self.lower_limit, self.upper_limit, self.every_ith]

    def gen_sample_source_label_pairs(self) -> Tuple[str, str]:
        for kind in self.kinds:
            if kind == "all-combinations":
                print("Generating all-combinations-test-data for every", self.every_ith, "'th value from",
                      self.lower_limit, "to", self.upper_limit, "...")
                for decl_nr in range(self.lower_limit, self.upper_limit, self.every_ith):
                    for access_nr in range(self.lower_limit, self.upper_limit, self.every_ith):
                        label = "faulty" if access_nr > decl_nr else "correct"
                        assert label in self.get_labels(), label
                        sample_source = self.generate_sample(decl_nr, access_nr, language=self.language)
                        sample_desc = "decl: " + str(decl_nr) + ", access: " + str(access_nr) + ", actual: " + label
                        # TODO: yield description!?
                        yield sample_source, label
            elif kind in ["random", "around-decision-border"]:
                # pos_count = int(self.sample_count / 2)
                yield from self.gen_sample_sources(kind=kind)

    def gen_sample_nodes_label_pairs(self) -> Tuple[List[ParserFrontend.AstNodeType], str]:
        for sample_source, label in self.gen_sample_source_label_pairs():
            # Parse the sample code:
            sample_ast = self.frontend.get_ast_from_source(sample_source, ".cpp")
            # There are no cases in which parsing could fail:
            assert sample_ast is not None
            # Check ast format:
            tu_decl = sample_ast.children[0]
            function_decl = tu_decl.children[0]
            assert function_decl.kind[0:2] == ["Decl", "Function"], "Expected TU's child to be a function decl"
            assert function_decl.id.spelling.endswith(":main"), "Expected function decl's name to be \"main\""
            # yield from self.gen_ast_nodes_of_relevant_methods_from_testcases_asts([[function_decl]])
            # Extract relevant body nodes of the samples function:
            fun_body_sub_asts = self.frontend.get_relevant_nodes_from_method_def_node(function_decl)
            # ... and yield it together with its label
            yield fun_body_sub_asts, label

    # def generate_all_combinations_test_set(self, lower_limit, upper_limit, language="Java"):
    #     print("Generating all-combinations-test-data-set for values from " + str(lower_limit) + " to " + str(
    #         upper_limit) + " ...")
    #     samples = []
    #     list_of_corresponding_ast_nodes = []
    #     labels_onehot = []
    #     sample_descriptions = []
    #     gen_count = 0
    #
    #     for def_nr in range(lower_limit, upper_limit):
    #         for use_nr in range(lower_limit, upper_limit):
    #             labels_onehot.append([0, 1] if use_nr > def_nr else [1, 0])
    #             source_as_sequence_as_array, corresponding_ast_nodes = self.preprocess_source(
    #                 self.generate_sample(def_nr, use_nr, language=language))
    #             samples.append(source_as_sequence_as_array)
    #             list_of_corresponding_ast_nodes.append(corresponding_ast_nodes)
    #             sample_descriptions.append(
    #                 "def: " + str(def_nr) + ", use: " + str(use_nr) + ", actual: " + str(labels_onehot[-1]))
    #             gen_count += 1
    #             if gen_count % 1000 == 0:
    #                 print(str(gen_count) + "/" + str((upper_limit - lower_limit) ** 2))
    #     samples_as_array = to_numpy_array(samples)
    #     labels_onehot_as_array = to_numpy_array(labels_onehot)
    #     return (samples_as_array, labels_onehot_as_array, sample_descriptions, list_of_corresponding_ast_nodes)
    #
    # # Returns samples as array with onehot lables for all combinations in the given ranges. In addition for each
    # # sample a description and a list of corresponding ast nodes is returned.
    # def get_all_combinations_test_set(self, lower_limit, upper_limit, language="Java"):
    #     key = str(lower_limit) + "_" + str(upper_limit) + "_" + language
    #     # Generate first, if dataset is not in cache:
    #     if (key not in self.test_set_cache):
    #         self.test_set_cache[key] = self.generate_all_combinations_test_set(lower_limit, upper_limit, language)
    #     # Return genereated or cached version in the same way:
    #     return self.test_set_cache[key]

    @classmethod
    def generate_sample(cls, array_decl_length, array_access_length, language):
        # print("gen", array_decl_length, array_access_length)
        if language == "C++":
            return """\
int main() {
  int arr[""" + str(array_decl_length) + """];
  for(int i = 0; i < """ + str(array_access_length) + """; i += 1) {
    arr[i] = i;
  }
  return 0;
}
"""
        elif language == "Java":
            return """\
public class T {
  public static void main(String[] args) {
    int[] arr = new int[""" + str(array_decl_length) + """];
    for(int i = 0; i < """ + str(array_access_length) + """; i += 1) {
      arr[i] = i;
      if(i > 0) {
        System.out.println(arr[i-1]);
      }
    }
  }
}
"""
        elif language == "StupidJava":
            return """\
public class T {
  public static void main(String[] args) {
    int[] arr = new int[""" + str(array_decl_length) + """];
    int[] arr = new int[""" + str(array_decl_length) + """];
    int[] arr = new int[""" + str(array_decl_length) + """];
    int[] arr = new int[""" + str(array_decl_length) + """];
    int[] arr = new int[""" + str(array_decl_length) + """];
    int[] arr = new int[""" + str(array_decl_length) + """];
    int[] arr = new int[""" + str(array_decl_length) + """];
    int[] arr = new int[""" + str(array_decl_length) + """];
    int[] arr = new int[""" + str(array_decl_length) + """];
    int[] arr = new int[""" + str(array_decl_length) + """];
    for(int i = 0; i < """ + str(array_access_length) \
                   + """ && i < """ + str(array_access_length) \
                   + """ && i < """ + str(array_access_length) \
                   + """ && i < """ + str(array_access_length) \
                   + """ && i < """ + str(array_access_length) \
                   + """ && i < """ + str(array_access_length) \
                   + """ && i < """ + str(array_access_length) \
                   + """ && i < """ + str(array_access_length) \
                   + """ && i < """ + str(array_access_length) \
                   + """ && i < """ + str(array_access_length) \
                   + """; i++) {
      arr[i] = i;
      if(i > 0) {
        System.out.println(arr[i-1]);
      }
    }
  }
}
"""
        elif language == "Python":
            return """\
arr = tuple(i for i in range(""" + str(array_decl_length) + """))
for i in range(""" + str(array_access_length) + """):
  print(arr[i])"""
        else:
            raise Exception("Unknown language " + str(language))

    def generate_positive_sample(self):
        # Shuffle until it is a positive combination:
        while True:
            arr_decl_len = random.randint(self.lower_limit, self.upper_limit)
            arr_access_len = random.randint(self.lower_limit, self.upper_limit)
            if arr_decl_len >= arr_access_len:
                break
        return self.generate_sample(arr_decl_len, arr_access_len, self.language)

    def generate_negative_sample(self):
        # Shuffle until it is a negative combination:
        while True:
            arr_decl_len = random.randint(self.lower_limit, self.upper_limit)
            arr_access_len = random.randint(self.lower_limit, self.upper_limit)
            if arr_decl_len < arr_access_len:
                break
        return self.generate_sample(arr_decl_len, arr_access_len, self.language)

    # def create_samples(self, pos_count, neg_count=None, kind="random"):
    #     start = time.time()
    #
    #     print("Creating/Emptying directory structure ...")
    #     local_sample_path_root = os.path.join(self.project_path, kind, self.language)
    #     recreate_dir(local_sample_path_root)
    #
    #     samples_dir_path = os.path.join(local_sample_path_root, self.SAMPLES_DIR_NAME)
    #     recreate_dir(samples_dir_path)
    #
    #     labels_dir_path = os.path.join(local_sample_path_root, self.LABELS_DIR_NAME)
    #     recreate_dir(labels_dir_path)
    #
    #     count = 0
    #     for sample_src, label in self.gen_sample_sources(pos_count, neg_count, kind):
    #         file_name = str(count).zfill(10)  # add leading zeros
    #         # Write sample source:
    #         write_file_content(os.path.join(samples_dir_path, file_name), content)
    #         # Write label:
    #         write_file_content(os.path.join(labels_dir_path, file_name), label)
    #         count += 1
    #
    #     end = time.time()
    #     print("Created " + str(pos_count + neg_count) + " samples in " + str(end - start) + "s")

    def gen_sample_sources(self, kind="random"):
        print("Creating ", end="")
        print_suffix = "samples for language " + self.language + " of kind " + kind + " ..."
        if kind == "random":
            print(self.sample_count, print_suffix)
            for i in range(int(self.sample_count / 2)):
                yield self.generate_positive_sample(), "correct"
                yield self.generate_negative_sample(), "faulty"
        elif kind == "around-decision-border":
            print(int((self.upper_limit - self.lower_limit) * 2 / self.every_ith), print_suffix)
            for i in range(self.lower_limit, self.upper_limit, self.every_ith):
                yield self.generate_sample(i, i, self.language), "correct"
                yield self.generate_sample(i, i + 1, self.language), "faulty"
        else:
            assert False, "Unknown kind " + kind

    # def get_samples_list(self, language="Java", kind="random"):
    #     """
    #     Retrieves all samples for given language
    #     :param language:
    #     :param kind:
    #     :return:
    #     """
    #     samples_list = []
    #     labels_list = []
    #     for sample_file_path, label in self.generate_sample_paths_list(language, kind):
    #         samples_list.append(sample_file_path)
    #         labels_list.append(label)
    #         # print(sample_file_path + ":" + str(label))
    #     print("Found " + str(len(samples_list)) + " samples.")
    #     return samples_list, labels_list
    #
    # def generate_sample_paths_list(self, language="Java", kind="random"):
    #     """
    #     Generator for retrieval of all samples for given language
    #     :param language:
    #     :param kind:
    #     :return:
    #     """
    #     local_sample_path_root = os.path.join(self.project_path, kind, language)
    #
    #     samples_dir_path = os.path.join(local_sample_path_root, self.SAMPLES_DIR_NAME)
    #     labels_dir_path = os.path.join(local_sample_path_root, self.LABELS_DIR_NAME)
    #     for f in listdir(samples_dir_path):
    #         sample_file_path = os.path.join(samples_dir_path, f)
    #         if isfile(sample_file_path):
    #             # get corresponding label
    #             label_file_path = os.path.join(labels_dir_path, f)
    #             label = int(read_file_content(label_file_path))
    #
    #             yield sample_file_path, label
    #
    #             # print(sample_file_path + ":" + str(label))
    #         else:
    #             raise Exception('Found unexpected directory')
    #
    # def generate_samples_list(self, language="Java", kind="random"):
    #     for sample_file_path, label in self.generate_sample_paths_list(language=language, kind=kind):
    #         yield read_file_content(sample_file_path), label


class ArrDeclAccessSimple(ArrDeclAccess):

    def __init__(self, language, kinds, sample_count, lower_limit, upper_limit, **kwargs):
        # Currently only ClangTooling (i.e. C++ or C) is supported
        assert language in ["C++", "C"], language
        self.test_set_cache = {}
        project_path = os.path.join(ENVIRONMENT.data_root_path, "ArrDeclAccess")  # parent path for generated samples
        frontend = ClangToolingSimple(project_path=project_path)

        assert len(kinds) > 0, kinds
        self.kinds = kinds
        self.sample_count = sample_count
        assert lower_limit >= 0, lower_limit
        self.lower_limit = lower_limit
        assert upper_limit > lower_limit, str(lower_limit) + ", " + str(upper_limit)
        self.upper_limit = upper_limit

        # Must come last (because use of __repr__)
        super(ArrDeclAccess, self).__init__(language,
                                            frontend,
                                            project_path,  # parent path for generated samples
                                            ["faulty", "correct"],
                                            **kwargs
                                            )


class MemNet(DatasetCustodian, ABC):
    ZIP_FILE_NAME = "buffer_overrun_memory_networks"
    FILE_NAMES = ["test_1_100", "test_2_100", "test_3_100", "test_4_100", "training_100"]
    FILE_EXTENSION = "txt"
    LABELS_FILE_SUFFIX = "_labels"

    file_names_to_use: List[str]

    def __init__(self, file_names_to_use, split_info, **kwargs):
        if file_names_to_use is None:
            file_names_to_use = self.FILE_NAMES
        else:
            assert sum([1 for file_name_to_use in file_names_to_use if file_name_to_use not in self.FILE_NAMES]) == 0, \
                file_names_to_use
        self.file_names_to_use = file_names_to_use

        if split_info == "paper":
            assert self.file_names_to_use == self.FILE_NAMES
            # Special split info:
            split_info = dict(test_fraction=4000.0 / 14000.0, validation_fraction=0.15 * 10000.0 / 14000.0,
                              shuffle=False)
            kwargs["split_info"] = split_info
        project_path = os.path.join(ENVIRONMENT.data_root_path, "MemNet", MemNet.ZIP_FILE_NAME)
        frontend = ClangTooling(project_path=project_path)

        # Must come last (because use of __repr__)
        super().__init__("C++",
                         frontend,
                         project_path,
                         ["faulty", "correct"],
                         **kwargs
                         )

    def repr_list(self):
        return super().repr_list()

    def gen_sample_nodes_label_pairs(self) -> Tuple[List[object], str]:
        for file_name in self.file_names_to_use:

            samples_file_path = os.path.join(self.project_path, file_name + "." + self.FILE_EXTENSION)
            labels_file_path = os.path.join(self.project_path, file_name + self.LABELS_FILE_SUFFIX + "."
                                            + self.FILE_EXTENSION)

            print("Reading samples and labels for", file_name, "from", samples_file_path, "and", labels_file_path,
                  "...")
            with open(samples_file_path, "r") as samples_file:
                with open(labels_file_path, "r") as labels_file:
                    sample_file_line_number = 0
                    # Iterate over label file lines and read the corresponding sample from samples file:
                    for label_line_obj in labels_file:
                        label_line = label_line_obj.rstrip('\n')  # format is "<?>:=:<label_int>" (e.g. "14:=:1")
                        query_line_number = int(label_line.split(":=:")[0])
                        label_int = int(label_line[-1])
                        assert label_int in self.get_label_ints(), label_int
                        label = self.int_to_label_mapping[label_int]

                        # Read sample:
                        sample_source = ""
                        curr_line = ""

                        # Sample ends with line starting with "}"
                        while not curr_line.startswith("}"):
                            sample_line_obj = next(samples_file, None)
                            if sample_line_obj is not None:
                                sample_file_line_number += 1
                                curr_line = str(sample_line_obj)
                                # The samples are not valid C code. Transform the code to make it valid:
                                if "NULL" in curr_line:
                                    # e.g.:
                                    #   char entity_3[61] = "";
                                    #   entity_3 = NULL;"
                                    # Error: error: array type 'char [61]' is not assignable
                                    #   entity_3 = NULL;
                                    #   ~~~~~~~~ ^
                                    # NULL only occurs in such assignments:
                                    assert curr_line.lstrip().startswith("entity_") and curr_line.endswith(" = NULL;\n")
                                    # Not sure what the intention of this line is, but it is ill-formed and has no
                                    # effect as the entity is set afterwards
                                    curr_line = ""
                                elif re.search(r"char entity_\d*?\[entity_\d*?\] = \"\";\n", curr_line) is not None:
                                    # e.g.
                                    #   char entity_2[entity_0] = "";
                                    # Error: error: variable-sized object may not be initialized
                                    #   char entity_2[entity_0] = "";
                                    #        ^                    ~~
                                    # Remove the initializer:
                                    curr_line = curr_line[0:-len(" = \"\";\n")] + ";\n"
                                # Append (transformed) current line to sample
                                sample_source += curr_line
                            else:
                                # Unexpected end of file
                                assert False, "No full sample for label file line!"

                        # There should be a sample for each label:
                        assert len(sample_source) > 0
                        # Check whether each sample starts with "void fun ()"
                        assert sample_source.startswith("void fun ()")

                        # The current query line number should always be the source code snippets last statement's line
                        # number. This is just an observation and of no further importance for the data preparation
                        # since the query line is ignored anyway.
                        assert sample_file_line_number - 1 == query_line_number, \
                            (sample_file_line_number, query_line_number)

                        # Add necessary includes to make the sample code valid:
                        sample_source = "\n".join([
                            # "#include <stddef.h>",  # For NULL
                            "#include <string.h>",  # For memset, strcpy
                            "#include <stdlib.h>",  # for malloc
                            sample_source])
                        # Parse the sample code:
                        sample_ast = self.frontend.get_ast_from_source(sample_source, ".c")
                        # There are no cases in which parsing could fail:
                        assert sample_ast is not None
                        # Check ast format:
                        tu_decl = sample_ast.children[0]
                        function_decl = tu_decl.children[0]
                        assert function_decl.kind[0:2] == ["Decl", "Function"], \
                            "Expected TU's child to be a function decl"
                        assert function_decl.id.spelling.endswith(":fun"), "Expected function decl's name to be \"fun\""
                        # Extract relevant body nodes of the samples function:
                        fun_body_sub_asts = self.frontend.get_relevant_nodes_from_method_def_node(function_decl)
                        # ... and yield it together with its label
                        yield fun_body_sub_asts, label
            # print("Yielded", len(samples) - prev_count, "samples in", file_name)
        # print("Yielded", len(samples), "samples overall.")


NO_BUG_LABEL_NAME = "CWE00_NO_FLAW_FOUND"
UNKNOWN_BUG_LABEL_NAME = "CWE01_UNKNOWN_FLAW"  # for unknown code
ALL_CWE_CLASSES = ["CWE272_Least_Privilege_Violation", "CWE197_Numeric_Truncation_Error",
                   "CWE570_Expression_Always_False", "CWE511_Logic_Time_Bomb", "CWE467_Use_of_sizeof_on_Pointer_Type",
                   "CWE843_Type_Confusion", "CWE672_Operation_on_Resource_After_Expiration_or_Release",
                   "CWE482_Comparing_Instead_of_Assigning", "CWE562_Return_of_Stack_Variable_Address",
                   "CWE758_Undefined_Behavior", "CWE563_Unused_Variable", "CWE338_Weak_PRNG",
                   "CWE484_Omitted_Break_Statement_in_Switch", "CWE469_Use_of_Pointer_Subtraction_to_Determine_Size",
                   "CWE124_Buffer_Underwrite", "CWE398_Poor_Code_Quality", "CWE835_Infinite_Loop",
                   "CWE526_Info_Exposure_Environment_Variables", "CWE675_Duplicate_Operations_on_Resource",
                   "CWE506_Embedded_Malicious_Code", "CWE440_Expected_Behavior_Violation",
                   "CWE273_Improper_Check_for_Dropped_Privileges", "CWE401_Memory_Leak",
                   "CWE195_Signed_to_Unsigned_Conversion_Error", "CWE500_Public_Static_Field_Not_Final",
                   "CWE242_Use_of_Inherently_Dangerous_Function", "CWE780_Use_of_RSA_Algorithm_Without_OAEP",
                   "CWE256_Plaintext_Storage_of_Password", "CWE127_Buffer_Underread", "CWE416_Use_After_Free",
                   "CWE773_Missing_Reference_to_Active_File_Descriptor_or_Handle", "CWE665_Improper_Initialization",
                   "CWE510_Trapdoor", "CWE561_Dead_Code", "CWE321_Hard_Coded_Cryptographic_Key",
                   "CWE534_Info_Exposure_Debug_Log", "CWE123_Write_What_Where_Condition",
                   "CWE591_Sensitive_Data_Storage_in_Improperly_Locked_Memory", "CWE134_Uncontrolled_Format_String",
                   "CWE674_Uncontrolled_Recursion", "CWE571_Expression_Always_True", "CWE328_Reversible_One_Way_Hash",
                   "CWE377_Insecure_Temporary_File", "CWE391_Unchecked_Error_Condition",
                   "CWE481_Assigning_Instead_of_Comparing", "CWE427_Uncontrolled_Search_Path_Element",
                   "CWE325_Missing_Required_Cryptographic_Step",
                   "CWE666_Operation_on_Resource_in_Wrong_Phase_of_Lifetime", "CWE459_Incomplete_Cleanup",
                   "CWE196_Unsigned_to_Signed_Conversion_Error", "CWE176_Improper_Handling_of_Unicode_Encoding",
                   "CWE475_Undefined_Behavior_for_Input_to_API", "CWE367_TOC_TOU", "CWE259_Hard_Coded_Password",
                   "CWE366_Race_Condition_Within_Thread", "CWE190_Integer_Overflow", "CWE690_NULL_Deref_From_Return",
                   "CWE404_Improper_Resource_Shutdown", "CWE247_Reliance_on_DNS_Lookups_in_Security_Decision",
                   "CWE785_Path_Manipulation_Function_Without_Max_Sized_Buffer", "CWE457_Use_of_Uninitialized_Variable",
                   "CWE476_NULL_Pointer_Dereference", "CWE483_Incorrect_Block_Delimitation",
                   "CWE284_Improper_Access_Control", "CWE676_Use_of_Potentially_Dangerous_Function",
                   "CWE244_Heap_Inspection", "CWE226_Sensitive_Information_Uncleared_Before_Release",
                   "CWE188_Reliance_on_Data_Memory_Layout", "CWE761_Free_Pointer_Not_at_Start_of_Buffer",
                   "CWE546_Suspicious_Comment", "CWE15_External_Control_of_System_or_Configuration_Setting",
                   "CWE126_Buffer_Overread", "CWE397_Throw_Generic_Exception", "CWE400_Resource_Exhaustion",
                   "CWE223_Omission_of_Security_Relevant_Information", "CWE36_Absolute_Path_Traversal",
                   "CWE122_Heap_Based_Buffer_Overflow", "CWE191_Integer_Underflow", "CWE390_Error_Without_Action",
                   "CWE587_Assignment_of_Fixed_Address_to_Pointer", "CWE464_Addition_of_Data_Structure_Sentinel",
                   "CWE480_Use_of_Incorrect_Operator", "CWE681_Incorrect_Conversion_Between_Numeric_Types",
                   "CWE253_Incorrect_Check_of_Function_Return_Value", "CWE23_Relative_Path_Traversal",
                   "CWE252_Unchecked_Return_Value", "CWE369_Divide_by_Zero", "CWE396_Catch_Generic_Exception",
                   "CWE364_Signal_Handler_Race_Condition", "CWE789_Uncontrolled_Mem_Alloc",
                   "CWE194_Unexpected_Sign_Extension", "CWE606_Unchecked_Loop_Condition",
                   "CWE605_Multiple_Binds_Same_Port", "CWE667_Improper_Locking",
                   "CWE479_Signal_Handler_Use_of_Non_Reentrant_Function", "CWE617_Reachable_Assertion",
                   "CWE535_Info_Exposure_Shell_Error", "CWE78_OS_Command_Injection",
                   "CWE762_Mismatched_Memory_Management_Routines", "CWE478_Missing_Default_Case_in_Switch",
                   "CWE426_Untrusted_Search_Path", "CWE319_Cleartext_Tx_Sensitive_Info",
                   "CWE685_Function_Call_With_Incorrect_Number_of_Arguments", "CWE615_Info_Exposure_by_Comment",
                   "CWE90_LDAP_Injection", "CWE121_Stack_Based_Buffer_Overflow", "CWE114_Process_Control",
                   "CWE222_Truncation_of_Security_Relevant_Information", "CWE468_Incorrect_Pointer_Scaling",
                   "CWE590_Free_Memory_Not_on_Heap", "CWE775_Missing_Release_of_File_Descriptor_or_Handle",
                   "CWE415_Double_Free", "CWE688_Function_Call_With_Incorrect_Variable_or_Reference_as_Argument",
                   "CWE588_Attempt_to_Access_Child_of_Non_Structure_Pointer",
                   "CWE832_Unlock_of_Resource_That_is_Not_Locked", "CWE327_Use_Broken_Crypto",
                   "CWE680_Integer_Overflow_to_Buffer_Overflow", "CWE620_Unverified_Password_Change"]


class Juliet(DatasetCustodian, ABC):
    ZIP_FILE_NAME = "Juliet_Test_Suite_v1.3_for_C_Cpp"

    class DifficultyClass(Enum):
        SIMPLE = 1
        COMPLEX_CONTROL_FLOW = 2
        DUMMY = 3  # For loading old models
        COMPLEX_DATA_FLOW = 31

    testcases_path: str = None
    difficulty_classes: List[DifficultyClass]

    # list of file extensions of files that should be part of the dataset without dot (e.g. "java"):
    file_extensions: List[str] = None

    def __init__(self, language: str,
                 label_whitelist: list = None,  # None for all
                 difficulty_classes: List[DifficultyClass] = None,
                 **kwargs):
        # Initialize depending on language:

        project_path = None  # path to juliet src directory (to check whether a file is part of juliet or e.g. std library)
        if language == "Java":
            project_path = os.path.join(ENVIRONMENT.data_root_path, "Juliet", "Juliet_Test_Suite_v1.3_for_Java")
            self.file_extensions = ["java"]
            frontend = JavaLang()
        elif language in ["C++", "C"]:
            project_path = os.path.join(ENVIRONMENT.data_root_path, "Juliet", Juliet.ZIP_FILE_NAME)
            self.file_extensions = ["c", "cpp"]
            # frontend = LibClang([os.path.join(self.project_path, "testcasesupport")])
            frontend = ClangTooling([os.path.join(project_path, "testcasesupport")], project_path)
        else:
            assert False, "Invalid language " + str(language)
        self.testcases_path = os.path.join(project_path, "testcases")

        # Config values for this class (other config values are passed to super class)
        self.label_whitelist = label_whitelist

        if difficulty_classes is None:
            # All is default:
            difficulty_classes = [difficulty_class for difficulty_class in self.DifficultyClass
                                  if difficulty_class != self.DifficultyClass.DUMMY]
        self.difficulty_classes = difficulty_classes

        # Must come last (because use of __repr__)
        super().__init__(language, frontend, project_path, self.get_labels_from_test_path(), **kwargs)

    def repr_list(self):
        return super().repr_list() + [str(self.difficulty_classes), self.file_extensions]

    def get_labels_from_test_path(self) -> List[str]:
        """
        Generates labels for this Juliet instance
        :return:
        """
        result = []
        for abs_path, file_or_dir_name in get_non_rec_dir_content_names(self.testcases_path):
            if os.path.isdir(abs_path) and file_or_dir_name.startswith("CWE"):
                if self.label_whitelist is None or file_or_dir_name in self.label_whitelist:
                    result.append(file_or_dir_name)
        # Add label for "no bug" class
        result.append(NO_BUG_LABEL_NAME)
        # Add label for "unknown bug" class
        # result.append(UNKNOWN_BUG_LABEL_NAME)
        if self.label_whitelist is not None:
            # Make sure that all whitelist labels are valid:
            assert sum([1 for label in self.label_whitelist if
                        label not in result]) == 0, "At least one whitelist label is invalid."
        return result

    def gen_sample_nodes_label_pairs(self) -> Tuple[List[ParserFrontend.AstNodeType], str]:
        """
        First step in pipeline: Yields list of sub-asts label pairs (sub-ast for each method body root node)
        :return:
        """
        # For each label(=directory name)
        processed_labels_count = 0
        for bug_label in self.get_labels():
            if bug_label in [NO_BUG_LABEL_NAME, UNKNOWN_BUG_LABEL_NAME]:
                continue
            path = os.path.join(self.testcases_path, bug_label)
            print("Generate samples for " + bug_label + " (" + str(processed_labels_count + 1) + "/" + str(
                len(self.get_labels()) - 1) + ") ...")

            print("  Considering testcases with one of the following diffictulty classes:", self.difficulty_classes)

            selected_abs_path_collections = {True: [], False: []}  # True for windows=True, False analogously
            # Regexp from Juliet doc (wihtout beginning ^ because names from frontend may be fully qualified)
            relevant_method_name_to_label_mapping = {
                # Secondary good first, look for primary good afterwards:
                NO_BUG_LABEL_NAME: [r"good(\d+|G2B\d*|B2G\d*)$", r"(CWE.*_)?good$"],
                # Primary bad:
                bug_label: [r"(CWE.*_)?bad$"]
            }
            print("  Considering method nodes having a name matching one of the following: ",
                  relevant_method_name_to_label_mapping)
            collected_testcase_ids = []
            testcases_h_contents = []
            if os.path.exists(os.path.join(path, "testcases.h")):
                testcases_h_contents.append((read_file_content(os.path.join(path, "testcases.h")), path))
            else:
                for abs_path, file_or_dir_name in get_non_rec_dir_content_names(path):
                    assert file_or_dir_name.startswith("s")
                    testcases_h_contents.append((read_file_content(os.path.join(abs_path, "testcases.h")), abs_path))
            for testcases_h_content, dir_path in testcases_h_contents:
                for line in testcases_h_content.split("\n"):
                    if "void" in line:
                        if line.endswith(("good();}", "bad();}")):
                            # E.g. "namespace CWE606_Unchecked_Loop_Condition__char_connect_socket_83 { void bad();}"
                            # ==> C++
                            # Extract testcase name from whole line:
                            testcase_name = line \
                                .replace("namespace", "") \
                                .replace("{ void good();}", "") \
                                .replace("{ void bad();}", "") \
                                .strip()
                            testcase_extension = "cpp"

                        elif line.endswith(("_good();", "_bad();")):
                            # E.g. "void CWE606_Unchecked_Loop_Condition__char_environment_11_bad();"
                            # ==> C
                            # Extract testcase name from whole line:
                            testcase_name = line \
                                .replace("void", "") \
                                .replace("_good();", "") \
                                .replace("_bad();", "") \
                                .strip()
                            testcase_extension = "c"
                        else:
                            assert False, line

                        # Most testcase names appear twice (once with good and once with bad function). Some even
                        # appear four times. In this case, the same line (including good/bad suffix) appears twice
                        # in the tescases.h file. Not sure why. Example:
                        # "namespace CWE606_Unchecked_Loop_Condition__char_connect_socket_83 { void bad();}"
                        # Therefore only add the testcase if it was not collected already:
                        testcase_id = testcase_name + "_" + testcase_extension
                        if testcase_id in collected_testcase_ids:
                            continue

                        # Determine difficulty number of testcase:
                        difficulty = int(testcase_name.split("_")[-1])
                        assert difficulty > 0, difficulty
                        # Determine testcase's difficulty class:
                        if difficulty == 1:
                            # Easy sample
                            difficulty_class = self.DifficultyClass.SIMPLE
                        elif difficulty < 31:
                            # Complex with control flow
                            difficulty_class = self.DifficultyClass.COMPLEX_CONTROL_FLOW
                        else:
                            # Complex with data flow
                            difficulty_class = self.DifficultyClass.COMPLEX_DATA_FLOW

                        if difficulty_class not in self.difficulty_classes:
                            # Skip testcase:
                            continue

                        collected_testcase_ids.append(testcase_id)
                        # Determine the paths of the files that are associated with this testcase
                        testcase_file_paths = glob.glob(os.path.join(dir_path,
                                                                     testcase_name + "*." + testcase_extension))
                        # Sort files such that the main file comes last. The length of the file names can be used for
                        # this because the main one is the shortest. If two files have the same length (e.g. ..a.cpp
                        # and ..b.cpp) then sort uses alphabetical order. This is good as ..a.cpp calls function from
                        # ..b.cpp according to Juliet doc). FIXME: This order should not matter because everything is
                        # searched later!?
                        testcase_file_paths.sort(key=lambda x: (len(x), x), reverse=True)
                        # print(len(tc_files), tc_files)

                        # Check whether this file needs windows environment:
                        needs_windows = False
                        if "_w32" in testcase_name or "_wchar_t_" in testcase_name:
                            needs_windows = True

                        # Remember the current file and its needs:
                        selected_abs_path_collections[needs_windows].append(testcase_file_paths)

                        # Statistics:
                        processed_test_case_count = len(collected_testcase_ids)
                        if processed_test_case_count % 1000 == 0:
                            print(str(processed_test_case_count) + " test cases collected so far ...")

            assert len(selected_abs_path_collections[True]) + len(selected_abs_path_collections[False]) == \
                   len(collected_testcase_ids)
            print("Collected", len(collected_testcase_ids), "test cases",
                  "(" + str(len(selected_abs_path_collections[True])), "of these test cases need windows environment)")

            # Generate pairs of ast node lists and labels for files that need windows environment and these that
            # dont need it:
            for key in selected_abs_path_collections.keys():
                if len(selected_abs_path_collections[key]) > 0:
                    # Create dict with paths and context:
                    source_file_path_collections_with_context = {
                        "source_path_collections": selected_abs_path_collections[key],
                        "context": {"needs_windows": key}}
                    yield from self.gen_ast_nodes_of_relevant_methods_from_testcases_source_file_paths(
                        source_file_path_collections_with_context=source_file_path_collections_with_context,
                        relevant_method_name_to_label_mapping=relevant_method_name_to_label_mapping)
            processed_labels_count += 1
