# Master Thesis Code Repository

Code repository for master thesis `Explainable Vulnerability Detection on Abstract Syntax Trees with Arithmetic, LSTM-based Neural Networks` written by Simon Schröder at TU Dortmund University in 2019. The thesis was supervised by Prof. Dr. Falk Howar and Simon Dierl, M. Sc. at Chair 14 - Software Engineering.

The thesis is located in [`thesis/thesis.pdf`](thesis/thesis.pdf).


## Disclaimer
I wrote this tool during my master thesis, i.e., in a time-constrained environment in which code quality and a production-ready tool were not of highest priority. As a results, the code structure and documentation is not perfect, many unit tests are missing, and the use of the tool is not as convenient as it could be. I still hope that this repository is useful and I would be glad if the code was reused in other projects. 

## Notation
In code or comments, `parameter` corresponds to the term `hyperparameter` as used in the thesis.
 
## Setup
The following setup instructions are provided without any warranty as I did not test them on a fresh system. There may be additional or fewer packages or steps necessary.
1. Preferably use Ubuntu 18.04 (or Ubuntu 16.04). Another OS may work as well.
1. Create a (virtual/conda) environment that has the following installed and setup.
    - NVIDIA drivers, CUDA 9 or 10, and cuDNN (the specific versions depend on your hardware and OS)
    - TensorFlow 1.13.1 pip/conda package (preferably with GPU support)
        - Depending on your CUDA version and your GPU's compatibility level, there may be no official pre-build TensorFlow binary. In this case, you can build TensorFlow by yourself or look for matching pre-build pip-wheels on the web.
        - Don't forget to add the respective CUDA lib directory to `LD_LIBRARY_PATH`, e.g., `export LD_LIBRARY_PATH=path/to/MasterThesisConda/cuda-9.0/lib64:$LD_LIBRARY_PATH`.  
    - Clang 7 incl. development packages. For example, install the apt (ubuntu) packages:
        - `clang-7-dev`,
        - `llvm-7-dev`, and
        - `libclang-7-dev` (optional)
    - Wine C header files: Install apt package `libwine-dev`
    - libxml2 incl. development packages
        - Install apt package `libxml2-dev`
        and 
        - pip/conda package `lxml`
    - Other pip/conda packages: `beautifulsoup4`, `dill`, `javalang`, `matplotlib`, `numpy`, `pandas`, `regex`, `scikit-learn`   
1. Create a directory where datasets, prepared feature sequences, and trained models will be stored. Let its path be `MasterThesisData` in the following.
1. Download and extract the Juliet and MemNet dataset ZIP files from
    - https://samate.nist.gov/SRD/testsuites/juliet/Juliet_Test_Suite_v1.3_for_C_Cpp.zip  
    and
    - https://codeload.github.com/mjc92/buffer_overrun_memory_networks/zip/master  
    to
    - `MasterThesisData/Juliet/Juliet_Test_Suite_v1.3_for_C_Cpp/`  
    and
    - `MasterThesisData/MemNet/buffer_overrun_memory_networks/`,  
    respectively.
1. In file `MasterThesisData/Juliet/Juliet_Test_Suite_v1.3_for_C_Cpp/testcasesupport/std_testcase.h` move `#include <stdint.h>` from line 44 to line 37.
1. Create a symlink in the Wine C header `windows` directory (in my case `/usr/include/wine/windows`) from `windows/Winldap.h` to `windows/winldap.h`. This is only necessary on Linux since the Juliet test cases import the upper case variant. 
1. Create the directory `MasterThesisData/ArrDeclAccess`.
1. In the code directory (i.e., where this README is located), open `Environment.py` and add an `EnvironmentInfo` object for your machine. The following items explain the information which must be provided.
    1. The name of your machine (based on this name, the appropriate `EnvironmentInfo` object is selected on execution). Run `import platform; platform.node()` in a Python console to retrieve the name.
    1. The absolute path to `MasterThesisData`
    1. A list of your system's include paths for compiling basic C++ code. In my case `/usr/local/include`, `/usr/include/x86_64-linux-gnu`, `/usr/include`, and `/usr/lib/llvm-7/lib/clang/7.0.0/include`.
    1. The wine include path (the path to the directory which contains the directories `msvcrt` and `windows`). In my case `/usr/include/wine`.
    1. The installed TensorFlow version. `1.13.1` in almost all cases. 
    1. The number of GPUs to use. `0` corresponds to CPU mode, `1` to single GPU mode. The use of multiple GPUs for a single training is not tested/supported well. Instead, you can start one separate training process for each GPU by employing your OS'/shell's multi-processing capabilities.
    1. The ID of the GPU to use (according to the output of `nvidia-smi`). Ignored in CPU mode.    
1. Run the unit tests to make sure that basic imports and parsing are working. For example, invoke `python -m unittest discover -s ./tests/` in the code directory.

## Structure
- To prepare datasets and to train models on them, have a look at `Main.py`. Trained models are stored on disk inside `MasterThesisData/PreparedDatasets/<Dataset Custodian Directory>/GridSearch/<Parameter Combination Directory>`.
- To test a trained model on a (different) dataset or single source code snippet and to generate attribution overlays, have a look at `Classification.py`. Attribution overlays are stored in the respective model directory.
- The C++ parser, which is written in C++ and based on Clang's LibTooling, is located in `ClangAstToJson`.

## License
MIT (see LICENSE file)
