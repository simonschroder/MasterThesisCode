from DataPreparation import PREPARED_DATASET_PATH_ROOT
from Helpers import backup_dir_to_archive

if __name__ == '__main__':
    backup_dir_to_archive(PREPARED_DATASET_PATH_ROOT,
                          exclude_files_dirs_startswith=("FeatureSequences", "Samples_as_AST_nodes_with_labels_"),
                          verbose=False)
