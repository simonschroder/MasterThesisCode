"""
Parsing (of different languages)
"""
import abc
import ast
import copy
import itertools
from abc import ABC  # Abstract Base Class
from collections import OrderedDict
from typing import Tuple, Optional, Union

import javalang

from Helpers import *

"""
Feature kinds:
"""
# Value feature kinds:
VALUE_KIND_FEATS = OrderedDict(
    FEAT_VAL_NONE=0,
    FEAT_VAL_BASIC_LITERAL_KIND=30,  # convertible to float
    FEAT_VAL_STRING_LITERAL_KIND=20,
    FEAT_VAL_OBJECT_KIND=10,
)
FEAT_VAL_BASIC_LITERAL_KIND = VALUE_KIND_FEATS["FEAT_VAL_BASIC_LITERAL_KIND"]
FEAT_VAL_STRING_LITERAL_KIND = VALUE_KIND_FEATS["FEAT_VAL_STRING_LITERAL_KIND"]
FEAT_VAL_OBJECT_KIND = VALUE_KIND_FEATS["FEAT_VAL_OBJECT_KIND"]

# Variable/function feature kinds:
MEMBER_KIND_FEATS = OrderedDict(
    FEAT_MEMBER_NONE=0,
    # FEAT_MEMBER_LOCAL = 30,
    FEAT_MEMBER_FILE_OR_PROJECT=20,
    FEAT_MEMBER_SYSTEM_OR_INCLUDED=10,
)
# FEAT_MEMBER_LOCAL = MEMBER_KIND_FEATS["FEAT_MEMBER_LOCAL"]
FEAT_MEMBER_FILE_OR_PROJECT = MEMBER_KIND_FEATS["FEAT_MEMBER_FILE_OR_PROJECT"]
FEAT_MEMBER_SYSTEM_OR_INCLUDED = MEMBER_KIND_FEATS["FEAT_MEMBER_SYSTEM_OR_INCLUDED"]

# Type feature kinds:
TYPE_KIND_FEATS = OrderedDict(
    FEAT_TYPE_NONE=0,
    FEAT_TYPE_BASIC=50,
    FEAT_TYPE_OBJECT=40,
    FEAT_TYPE_REFERENCE=30,
    FEAT_TYPE_POINTER=20,
    FEAT_TYPE_ARRAY=10,
)
FEAT_TYPE_BASIC = TYPE_KIND_FEATS["FEAT_TYPE_BASIC"]
FEAT_TYPE_OBJECT = TYPE_KIND_FEATS["FEAT_TYPE_OBJECT"]
FEAT_TYPE_REFERENCE = TYPE_KIND_FEATS["FEAT_TYPE_REFERENCE"]
FEAT_TYPE_POINTER = TYPE_KIND_FEATS["FEAT_TYPE_POINTER"]
FEAT_TYPE_ARRAY = TYPE_KIND_FEATS["FEAT_TYPE_ARRAY"]

"""
Node kind features
This is still based on javalang for historic reasons
"""

LIST_OF_AST_NODE_KINDS = [
    javalang.tree.CompilationUnit, javalang.tree.Import, javalang.tree.Documented, javalang.tree.Declaration,
    javalang.tree.TypeDeclaration, javalang.tree.PackageDeclaration, javalang.tree.ClassDeclaration,
    javalang.tree.EnumDeclaration, javalang.tree.InterfaceDeclaration, javalang.tree.AnnotationDeclaration,
    javalang.tree.Type, javalang.tree.BasicType, javalang.tree.ReferenceType, javalang.tree.TypeArgument,
    javalang.tree.TypeParameter, javalang.tree.Annotation, javalang.tree.ElementValuePair,
    javalang.tree.ElementArrayValue, javalang.tree.Member, javalang.tree.MethodDeclaration,
    javalang.tree.FieldDeclaration, javalang.tree.ConstructorDeclaration, javalang.tree.ConstantDeclaration,
    javalang.tree.ArrayInitializer, javalang.tree.VariableDeclaration, javalang.tree.LocalVariableDeclaration,
    javalang.tree.VariableDeclarator, javalang.tree.FormalParameter, javalang.tree.InferredFormalParameter,
    javalang.tree.Statement, javalang.tree.IfStatement, javalang.tree.WhileStatement, javalang.tree.DoStatement,
    javalang.tree.ForStatement,
    javalang.tree.AssertStatement, javalang.tree.BreakStatement, javalang.tree.ContinueStatement,
    javalang.tree.ReturnStatement,
    javalang.tree.ThrowStatement, javalang.tree.SynchronizedStatement, javalang.tree.TryStatement,
    javalang.tree.SwitchStatement,
    javalang.tree.BlockStatement, javalang.tree.StatementExpression, javalang.tree.TryResource,
    javalang.tree.CatchClause,
    javalang.tree.CatchClauseParameter, javalang.tree.SwitchStatementCase, javalang.tree.ForControl,
    javalang.tree.EnhancedForControl,
    javalang.tree.Expression, javalang.tree.Assignment, javalang.tree.TernaryExpression, javalang.tree.BinaryOperation,
    javalang.tree.Cast,
    javalang.tree.MethodReference, javalang.tree.LambdaExpression, javalang.tree.Primary, javalang.tree.Literal,
    javalang.tree.This,
    javalang.tree.MemberReference, javalang.tree.Invocation, javalang.tree.ExplicitConstructorInvocation,
    javalang.tree.SuperConstructorInvocation, javalang.tree.MethodInvocation, javalang.tree.SuperMethodInvocation,
    javalang.tree.SuperMemberReference, javalang.tree.ArraySelector, javalang.tree.ClassReference,
    javalang.tree.VoidClassReference,
    javalang.tree.Creator, javalang.tree.ArrayCreator, javalang.tree.ClassCreator, javalang.tree.InnerClassCreator,
    javalang.tree.EnumBody,
    javalang.tree.EnumConstantDeclaration, javalang.tree.AnnotationMethod
]

# Manual node kind feature mappings for basic feature representation
END_OF_NODE_INDEX = 3
END_OF_NODE_VALUE = 20
AST_NODE_TOP_LEVEL_KINDS_TO_FEAT_VEC_MAPPING = {
    javalang.tree.Declaration: (10, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    javalang.tree.Expression: (0, 10, 0, 0, 0, 0, 0, 0, 0, 0),
    javalang.tree.Statement: (0, 0, 10, 0, 0, 0, 0, 0, 0, 0),
    # "EndOfNode":           (0, 0, 0, 20, 0, 0, 0, 0, 0, 0),
    # javalang.tree.Type: (0, 0, 0, 0, 10, 0, 0, 0, 0, 0),
    # javalang.tree.TypeArgument: (0, 0, 0, 0, 5, 0, 0, 0, 0, 0),
    # javalang.tree.TypeParameter: (0, 0, 0, 0, 5, 0, 0, 0, 0, 0),
    javalang.tree.Annotation: None,  # (0, 0, 0, 0, 0, 0, 0, 0, 0),
    javalang.tree.ElementValuePair: None,  # (0, 0, 0, 0, 0, 0, 0, 0, 0),
    javalang.tree.ElementArrayValue: None,  # (0, 0, 0, 0, 0, 0, 0, 0, 0),
    javalang.tree.CompilationUnit: None,  # (0, 0, 0, 0, 0, 0, 0, 0, 0),
    javalang.tree.Import: None,  # (0, 0, 0, 0, 0, 0, 0, 0, 0),
    javalang.tree.Documented: None,  # (0, 0, 0, 0, 0, 0, 0, 0, 0),
    javalang.tree.ArrayInitializer: None,  # (0, 5, 0, 10, 0, 0, 0, 0, 0, 0),
    javalang.tree.VariableDeclarator: (10, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    javalang.tree.InferredFormalParameter: (10, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    javalang.tree.SwitchStatementCase: (0, 0, 0, 0, 0, 10, 0, 0, 0, 0),
    javalang.tree.ForControl: (0, 0, 0, 0, 0, 0, 20, 0, 0, 0),
    javalang.tree.EnhancedForControl: (0, 0, 0, 0, 0, 0, 20, 0, 0, 0),
    javalang.tree.EnumBody: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}
AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING = {
    javalang.tree.BlockStatement: (0, 0, 10, 0, 0, 0, 0, 0, 0, 0),
    javalang.tree.IfStatement: (0, 0, 0, 0, 0, 10, 0, 0, 0, 0),
    javalang.tree.TernaryExpression: (0, 0, 0, 0, 0, 8, 0, 0, 0, 0),
    javalang.tree.BinaryOperation: (0, 20, 0, 0, 0, 0, 0, 0, 0, +0),  # not used for ClangTooling
    javalang.tree.WhileStatement: (0, 0, 0, 0, 0, 0, 20, 0, 0, 0),
    javalang.tree.DoStatement: (0, 0, 0, 0, 0, 0, 20, 0, 0, 0),
    javalang.tree.ForStatement: (0, 0, 0, 0, 0, 0, 20, 0, 0, 0),
    javalang.tree.BreakStatement: (0, 0, 0, 0, 0, 0, 0, 10, 0, 0),
    javalang.tree.ContinueStatement: (0, 0, 0, 0, 0, 0, 0, 10, 0, 0),
    javalang.tree.ReturnStatement: (0, 0, 0, 0, 0, 0, 0, 10, 0, 0),
    javalang.tree.MemberReference: (0, 20, 0, 0, 10, 0, 0, 0, 0, 0),
    javalang.tree.MethodInvocation: (0, 30, 0, 0, 0, 0, 0, 10, 0, 0)
}
# Merge the above mappings into one mapping which maps every node kind to a feature vector:
AST_NODE_KINDS_TO_FEAT_VEC_MAPPING = {}
for node_kind in LIST_OF_AST_NODE_KINDS:
    found = False
    for special_node_kind in AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING:
        if issubclass(node_kind, special_node_kind):
            AST_NODE_KINDS_TO_FEAT_VEC_MAPPING[node_kind] \
                = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[special_node_kind]
            found = True
            break
    if found:
        continue
    for top_level_node_kind in AST_NODE_TOP_LEVEL_KINDS_TO_FEAT_VEC_MAPPING:
        if issubclass(node_kind, top_level_node_kind):
            AST_NODE_KINDS_TO_FEAT_VEC_MAPPING[node_kind] = AST_NODE_TOP_LEVEL_KINDS_TO_FEAT_VEC_MAPPING[
                top_level_node_kind]
            break
# Check if all lengths are the same:
length = None
for feat_vec in AST_NODE_KINDS_TO_FEAT_VEC_MAPPING.values():
    if feat_vec is None:
        continue
    curr_length = len(feat_vec)
    if length is None:
        length = curr_length
    else:
        assert length == curr_length, "length " + str(length) + " and " + str(curr_length) + " differ!"

"""
Operator kind features
"""

# Manual operator feature mappings for basic feature representation. Use dummies to create distance between different
#  features of different operator groups
OPERATOR_STRINGS = ("<no_operator>",  # This will get index 0 and is here for the case that the node has no operator
                    "dummy", "dummy", "dummy", "dummy",
                    "dummy", "dummy", "dummy", "dummy", "dummy",
                    '=', '>>>=', '>>=', '<<=', '%=', '^=', '|=', '&=', '/=', '*=', '-=', '+=',  # assignment
                    "dummy", "dummy", "dummy", "dummy", "dummy",
                    '--', '++',  # unary post-/pre-fix
                    "dummy", "dummy", "dummy", "dummy", "dummy",
                    '||', '&&',  # binary logical comparision
                    "dummy", "dummy",
                    '!=',
                    "dummy", "dummy",
                    '==',
                    "dummy", "dummy",
                    "<", '<=',
                    "dummy", "dummy",
                    ">", '>=',  # binary logical comparision
                    "dummy", "dummy", "dummy", "dummy", "dummy",
                    '%', '^', '|', '&', '~', '!', '<<', '>>',  # bitwise
                    "dummy", "dummy", "dummy", "dummy", "dummy",
                    '/', '*', '-', '+',  # arithmetic
                    "dummy", "dummy", "dummy", "dummy", "dummy",
                    ':', '?', '=', '...', '->', '::',
                    "dummy", "dummy", "dummy", "dummy", "dummy",
                    '[]',
                    "dummy", "dummy", "dummy", "dummy", "dummy",
                    ',',
                    "dummy", "dummy", "dummy", "dummy", "dummy",
                    '?:')
OPERATOR_FEATS = {oper: OPERATOR_STRINGS.index(oper) for oper in OPERATOR_STRINGS if oper != "dummy"}


class FeatVecAnnInputInfo:
    """
    Information about a component in the feature vector which is input to an ANN
    """
    index_start: int
    index_end: int
    kind_count: int
    name: str

    def __init__(self, name, index_start, length=1, kind_count=None):
        self.index_start = index_start
        assert length > 0, length
        self.index_end = self.index_start + length
        assert self.index_start < self.index_end, self
        self.kind_count = kind_count
        self.name = name

    def __repr__(self):
        return str(self.__dict__)


class ParserFrontend(ABC):
    """
    Base class for parser frontends
    """
    string_lookup = {}
    project_var_name_lookup = {}
    included_var_name_lookup = {}
    ref_type_name_lookup = {}

    # Possible AST node types (used for typing information):
    AstNodeType = typing.Union["javalang.ast.Node", "LibClang.ClangNode", "ClangTooling.ClangAstNode"]

    def __repr__(self):
        return repr(type(self).__name__)

    @abc.abstractmethod
    def get_feat_vec_for_node(self, ast_node) -> Tuple:
        """
        Returns the feature vector of the given ast node
        :param ast_node:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_feat_vec_component_info(self, component):
        """
        Returns a list of FeatVecAnnInputInfo objects which describe the components of this frontend's feature vectors.
        :param component: Specifies whether information for all components or only some components should be returned
        :return:
        """
        pass

    @abc.abstractmethod
    def get_kind_counts(self) -> Dict:
        """
        Returns a the numbers of different node/operator/... feature kinds
        :return:
        """
        return {}

    @abc.abstractmethod
    def get_embedding_init_weights(self, feat_vec_input_info: FeatVecAnnInputInfo, pad_to_length: int):
        """
        Returns an array which can be used as initialization of an embedding layer's weights.
        :param feat_vec_input_info: The feature vector component for whose corresponding embedding layer the
         initialization should be returned
        :param pad_to_length:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_ast_from_source(self, source_code, source_file_path, verbosity=0):
        """
        Returns the ast for the given source code
        :param source_code:
        :param source_file_path:
        :param verbosity:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_asts_from_source_paths(self, source_file_paths) -> List:
        """
        Returns an ast for each of the given source code files
        :param source_file_paths:
        :return:
        """
        pass

    def get_ast_from_source_path(self, source_file_path: str) -> object:
        """
        Returns the ast of the given source code file
        :param source_file_path:
        :return:
        """
        return self.get_asts_from_source_paths([source_file_path])[0]

    def get_asts_from_source_path_collections(self, source_file_path_collections_with_context):
        """
        Returns asts for the each source code file in each source code file path collection obeying each source code
         file collection's environment/context requirements
        :param source_file_path_collections_with_context:
        :return:
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_node_as_string(cls, node, compact: bool = False) -> str:
        """
        Returns a textual representation of the given node
        :param node:
        :param compact: Whether the representation should only contain few information
        :return:
        """
        pass

    @classmethod
    def add_source_code_to_ast(cls, ast, source_code, source_extension, temp_file_path):
        """
        Adds the given source code + extension to the given node. This is useful in cases where the ast was generated
        from a temporary file.
        :param ast:
        :param source_code:
        :param source_extension:
        :param temp_file_path: If given, check whether the given path matches the one the ast was parsed from.
        :return:
        """
        pass

    @classmethod
    def get_node_as_pickable(cls, node, recursive: bool = True, context=None):
        """
        Returns a pickable version of the given node (and its children). Necessary only if libclang is used since
         libclang nodes contain unpickable objects.
        :param node:
        :param recursive:
        :param context:
        :return:
        """
        # On default, nodes are pickable:
        return node

    @classmethod
    def get_node_depth(cls, node):
        """
        Returns the lexical depth of the given node
        :param node:
        :return:
        """
        pass

    @classmethod
    @abc.abstractmethod
    def depth_search_tree(cls, node, skip_included: typing.Union[bool, str] = False, verbose: bool = False,
                          depth: int = 0, parent=None):
        """
        Yields all of the given node's children in depth search pre-order
        :param node:
        :param skip_included: If a path is provided, includes that are not below that path are skipped
        :param verbose:
        :param depth:
        :param parent:
        :return: node, depth, node's parent
        """
        yield None, -1, None

    @classmethod
    @abc.abstractmethod
    def is_included_or_imported_node(cls, ast_node, project_path: str = None):
        """
        Returns whether the given ast_node is part of a included or imported file/class
        :param ast_node:
        :param project_path:
        :return:
        """
        return

    @classmethod
    @abc.abstractmethod
    def is_method_definition_node(cls, ast_node):
        """
        Returns whether the given ast_node is the root of method definition
        :param ast_node:
        :return:
        """
        return

    @classmethod
    @abc.abstractmethod
    def get_method_name(cls, method_def_node):
        """
        Returns the name of the given method
        :param method_def_node:
        :return:
        """
        pass

    @classmethod
    def get_semantic_name(cls, fully_qualified_name_with_file_path: str):
        """
        fully_qualified_name_with_file_path is expected to be of format
         <lexical information / file path>:<semantic information / fully qualified name>. E.g.
         testcases/CWE401_Memory_Leak/s01/CWE401_Memory_Leak__int64_t_realloc_62b.cpp:CWE401_Memory_Leak__int64_
         t_realloc_62::badSource
         This returns the semantic part.
        part
        :param fully_qualified_name_with_file_path:
        :return:
        """
        # Split at single : (and not at ::). For this, lookbehind (?<...) and lookahead (?...) are used. (?<!...) and
        # (?!...) are negative/not versions of lookbehind and lookahead respectively.
        lexical_and_semantic_part = re.split("(?<!:):(?!:)", fully_qualified_name_with_file_path)
        assert len(lexical_and_semantic_part) == 2, fully_qualified_name_with_file_path
        return lexical_and_semantic_part[1]

    @classmethod
    @abc.abstractmethod
    def get_relevant_nodes_from_method_def_node(cls, method_def_node) -> List[object]:
        """
        Returns a list method_def_node children that are relevant e.g. without parameters
        :param method_def_node:
        :return:
        """
        pass

    @classmethod
    @abc.abstractmethod
    # TODO: Unify and move implementation up to here?
    def add_called_methods_to_list(cls, method_node, list_to_add, all_methods_in_content):
        """
        Adds method def nodes of all methods to list_to_add that satisfy the following:
         - They must be called directly by method_node
         - They must be in list_of_all_available_methods
        :param method_node:
        :param list_to_add:
        :param all_methods_in_content:
        :return:
        """
        pass

    def get_value_feat_vec_from_literal_value(self, node_value):
        """
        Returns a feature vector for the given literal
        :param node_value:
        :return:
        """
        pass

    @staticmethod
    def perform_common_lookup(key, dictionary: Dict) -> int:
        """
        Lookup the given key in the given dictionary and returns the key's value
        :param value:
        :param dictionary:
        :return:
        """
        LOOKUP_FACTOR = 1
        # Use "len(dictionary) + 1" because id 0 is reserved for "not available"
        return dictionary.setdefault(key, len(dictionary) + 1) * LOOKUP_FACTOR

    def lookup_string(self, string) -> int:
        return self.perform_common_lookup(string, self.string_lookup)

    def lookup_project_var_name(self, project_var_name) -> int:
        return self.perform_common_lookup(project_var_name, self.project_var_name_lookup)

    def reset_project_var_name_lookup(self):
        self.project_var_name_lookup = {}

    def lookup_included_var_name(self, included_var_name) -> int:
        return self.perform_common_lookup(included_var_name, self.included_var_name_lookup)

    def lookup_ref_type_name(self, ref_type_name) -> int:
        return self.perform_common_lookup(ref_type_name, self.ref_type_name_lookup)

    def print_tree(self, node, skip_included: typing.Union[bool, str] = False, verbose: bool = False,
                   add_feat_vec: bool = False, pre_depth: int = 0):
        """
        Prints the whole tree specified by root "node"
        :param node:
        :param skip_included:
        :param verbose:
        :param add_feat_vec:
        :param pre_depth:
        :return:
        """
        for sub_node, sub_depth, sub_parent in self.depth_search_tree(node, skip_included=skip_included,
                                                                      verbose=verbose, depth=pre_depth):
            print((("|   " * (sub_depth - 1) + "|`-> ") if sub_depth > 0 else "") + str(
                self.get_node_as_string(sub_node)))
            if add_feat_vec:
                feat_vec = self.get_feat_vec_for_node(sub_node)
                if feat_vec is None:
                    continue
                print(("|   " * (sub_depth - 1) + "|   ") if sub_depth > 0 else "", feat_vec)

    def get_sequence_as_array_from_a_sampleis_node_list(self, sample_as_node_list: List[AstNodeType]) \
            -> Union[Tuple[None, None], Tuple[np.ndarray, List[AstNodeType]]]:  # 2D ndarray
        """
        Returns a feature sequence for the given asts.
        :param sample_as_node_list:
        :return:
        """
        # sample_as_node_list contains ast nodes of one sample (statements of entry point, optionally with helper
        # methods)
        # Reset the lookup for project-wide variables/members. This way, samples having the same code structure, but
        # different variable names, are associated the same representation:
        self.reset_project_var_name_lookup()
        list_of_feature_vectors: List[np.ndarray] = []
        list_of_corresponding_ast_nodes: List[ParserFrontend.AstNodeType] = []
        # put feature vectors of entry point statements and helper methods behind each other
        for sub_ast in sample_as_node_list:
            for sub_node, depth, parent in self.depth_search_tree(sub_ast):  # pre order depth search
                # It is not necessary to distinguish between data preparation during dataset creation and
                # data preparation of an unknown sample. The lookups may be extended due to unknown source
                # but the learner should only compare values and should not look for a certain absolute
                # lookup value.
                feat_vec = self.get_feat_vec_for_node(sub_node)
                if feat_vec is None:
                    # print("no feature vector", sub_node.kind)
                    continue
                list_of_feature_vectors.append(to_numpy_array(feat_vec))
                list_of_corresponding_ast_nodes.append(sub_node)
        if len(list_of_feature_vectors) == 0:
            print("Empty list of feature vectors for",
                  [(str(sub_ast), sub_ast.tu.id.spelling) for sub_ast in sample_as_node_list])
            return None, None
        else:
            return to_numpy_array(list_of_feature_vectors), list_of_corresponding_ast_nodes

    @staticmethod
    def get_ast_nodes_location_summary(ast_nodes: List[AstNodeType]) -> Dict[str, Dict[str, Union[bool, List[int]]]]:
        """
        Returns a summary of the file paths and line numbers in which the given ast nodes are located.
        :param ast_nodes:
        :return: A dictionary with a key-value pair for each file path. Each file path's value is a dictionary
         contains the lines numbers in which one of the given ast nodes is located in. Additionally it contains whether
         the file contains the source code snippet's main entry point
        """
        file_locations = {}
        # Collect file locations and line numbers:
        for ast_node in ast_nodes:
            main_file_path = None
            if hasattr(ast_node, "tu"):
                main_file_path = ast_node.tu.id.spelling
                if isinstance(main_file_path, tuple):
                    assert len(main_file_path) == 3, main_file_path
                    # AST nodes were created from in-memory source code snippet. In this case main_file_path is a tuple
                    # of the temporary file, the source code snippet, and its extension. Use the source code with
                    # extension instead of path:
                    main_file_path = main_file_path[1] + "\n" + main_file_path[2]

            if hasattr(ast_node, "extent"):
                loc_start = ast_node.extent.start
                loc_end = ast_node.extent.end
                loc_file_path = loc_start.file
                is_main = False
                if loc_file_path == ":MN:":
                    assert main_file_path is not None
                    loc_file_path = main_file_path
                    is_main = True
                file_loc_info = file_locations.setdefault(loc_file_path, {"is_main": is_main, "line_intervals": []})
                assert file_loc_info["is_main"] == is_main, (loc_file_path, file_loc_info, is_main)
                file_loc_info["line_intervals"].append((loc_start.line, loc_end.line))

        # Aggregate line numbers:
        for loc_file_path, file_loc_info in file_locations.items():
            line_intervals = file_loc_info["line_intervals"]
            # Create set with all line numbers for current file:
            line_numbers = set()
            for line_start, line_end in line_intervals:
                line_numbers = line_numbers.union(set(range(line_start, line_end + 1)))
            # Make sorted list of line_numbers:
            line_numbers = sorted(list(line_numbers))
            # Compute intervals of line numbers such that all line numbers are covered and that the smallest amount
            # of intervals is used:
            reduced_line_intervals = []
            curr_line_interval = None
            prev_line_num = None
            for line_num in line_numbers:
                if curr_line_interval is None:
                    # Start with interval of size one:
                    curr_line_interval = [line_num, line_num]
                elif line_num - 1 == prev_line_num:
                    # Add current line number to current interval:
                    curr_line_interval[1] = line_num
                else:
                    # End of current interval:
                    reduced_line_intervals.append(curr_line_interval)
                    # Start next interval:
                    curr_line_interval = [line_num, line_num]
                prev_line_num = line_num
            # Add last interval (may be None if no line number for current file):
            reduced_line_intervals.append(curr_line_interval)
            # Replace line_intervals list content with reduced line intervals (such that reference is not changed):
            line_intervals[:] = reduced_line_intervals
        return file_locations

    @staticmethod
    def get_ast_nodes_location_summary_string(ast_nodes: List[AstNodeType], separator="\n"):
        """
        Returns a textual representation of get_ast_nodes_location_summary's result for the given ast nodes
        :param ast_nodes:
        :param separator:
        :return:
        """
        return separator.join([("(MN) " if file_loc_info["is_main"] else "")
                               + file_path + ": "
                               + ", ".join(str(line_interval)
                                           for line_interval
                                           in file_loc_info["line_intervals"])
                               for file_path, file_loc_info in
                               ParserFrontend.get_ast_nodes_location_summary(ast_nodes).items()])

    # @staticmethod
    # # TODO: Something is wrong here (lines seem to be missing)
    # def get_ast_nodes_source_code(ast_nodes: List[AstNodeType]):
    #     source_code = ""
    #     for file_path, file_loc_info in ParserFrontend.get_ast_nodes_location_summary(ast_nodes).items():
    #         if "\n" in file_path:
    #             file_source_code = "\n".join(file_path.split("\n")[:-1])
    #         else:
    #             file_source_code = read_file_content(file_path)
    #         line_numbers_to_keep = []
    #         for line_interval in file_loc_info["line_intervals"]:
    #             line_numbers_to_keep.extend(list(range(line_interval[0], line_interval[1])))
    #         line_numbers_to_keep = sorted(list(set(line_numbers_to_keep)))
    #         file_source_code_lines = file_source_code.split("\n")
    #         source_code += "\n".join([file_source_code_lines[line_number] for line_number in line_numbers_to_keep])
    #
    #     return source_code


class ClangTooling(ParserFrontend):
    include_paths = ENVIRONMENT.system_include_paths
    for include_path in include_paths:
        assert " " not in include_path, ("Include paths with spaces are currently not supported", include_path)
    compiler_args = ["-Wno-everything"]
    project_path = None
    # For each coarse kind the interval of corresponding fine kind numbers. Intervals are inclusive!
    KIND_NUMBER_INTERVALS: "ClangAstNode" = None
    # For each coarse kind there is a list of full kinds where the index is the
    # fine kind number
    KINDS: "ClangAstNode" = None

    UNIQUE_NUMBER_TO_NODE_KIND_MAPPING = None
    NODE_KIND_TO_UNIQUE_NUMBER_MAPPING = None

    # intervals are inclusive!
    @property
    def NODE_KIND_NUMBER_INTERVALS(self):
        return OrderedDict(
            Decl=self.KIND_NUMBER_INTERVALS.Decl,
            Stmt=self.KIND_NUMBER_INTERVALS.Stmt,  # Inclusing Expr
            # Expr=[15, 125],  # Not really necessary because in Stmt range
        )

    # interval is inclusive!
    @property
    def TYPE_KIND_NUMBER_INTERVAL(self):
        return self.KIND_NUMBER_INTERVALS.Type

    @property
    def TYPE_KIND_COUNT(self):
        return self.TYPE_KIND_NUMBER_INTERVAL[1] - self.TYPE_KIND_NUMBER_INTERVAL[0] + 1

    def __init__(self, include_paths=None, project_path=None):
        if include_paths is not None:
            self.include_paths.extend(include_paths)
        if project_path is not None and not project_path.endswith(os.path.sep):
            project_path += os.path.sep
        self.project_path = project_path
        super().__init__()

        # Get static kind info and create mappings out of it
        self.KIND_NUMBER_INTERVALS, self.KINDS = self.get_static_kind_info()
        self.create_node_kind_to_unique_kind_number_mappings()

    class ClangAstNode:
        exclude_for_str = ["children", "tu", "parent"]

        def __init__(self, attributes=None):
            assert attributes != {}, "Empty attributes for ClangAstNode"
            if attributes is None:
                attributes = {}
            self.__dict__ = attributes

        def __repr__(self):
            # Do not use exluded attributes for representation (because they may result in stack overflow):
            filtered = {key: val for key, val in self.get_dict().items() if
                        key not in getattr(self, "exclude_for_str", [])}
            return str(filtered)

        def get_compact_repr(self):
            compact_repr = (self.kind[1] + self.kind[0]
                            + " " + getattr(self.id, "spelling", "")
                            + " " + getattr(self, "operator", "")
                            + " " + str(getattr(self, "value", ""))
                            )
            if getattr(self, "ref_id", None) is not None:
                compact_repr += " " + str(getattr(self.ref_id, "spelling", ""))
            if getattr(self, "type", None) is not None:
                compact_repr += (
                        " " + self.type.kind[1] + self.type.kind[0]
                        + " " + str(getattr(self.type.id, "spelling", ""))
                )
            return compact_repr

        def get_dict(self):
            return self.__dict__

    def create_node_kind_to_unique_kind_number_mappings(self):
        """
        Creates mappings which allow to uniquely identify a given node_kind with an integer and vice versa. This is
         necessary because Decl's and Stmt's kind number intervals overlap. As there are no Type ast nodes in this
         context, this should not be used for Type kinds.
        :return:
        """
        self.UNIQUE_NUMBER_TO_NODE_KIND_MAPPING = {}
        index = 0
        for is_end_of in range(2):  # Once for normal nodes and once for end-of nodes
            for coarse_kind, fine_kind_interval in self.NODE_KIND_NUMBER_INTERVALS.items():
                start_index = index
                for kind in getattr(self.KINDS, coarse_kind, None):
                    assert (index - start_index) == (kind[2] - fine_kind_interval[0])
                    assert (index - start_index) < (fine_kind_interval[1] - fine_kind_interval[0] + 1)
                    if is_end_of == 1:
                        # Create new end-of kind as copy from normal kind
                        kind = list(kind)
                        kind[0] = "EndOf" + kind[0]
                    # Use tuples as they are hashable and can be used as key in dict
                    self.UNIQUE_NUMBER_TO_NODE_KIND_MAPPING[index] = tuple(kind)
                    index += 1

        self.NODE_KIND_TO_UNIQUE_NUMBER_MAPPING = {node_kind: unique_nr for unique_nr, node_kind in
                                                   self.UNIQUE_NUMBER_TO_NODE_KIND_MAPPING.items()}

    def get_embedding_init_weights(self, feat_vec_input_info: FeatVecAnnInputInfo, pad_to_length: int) \
            -> Optional[List[np.ndarray]]:
        weight_list = None
        if feat_vec_input_info.name == "node_kind_num":
            # Use hand-crafted feat-vecs as init-weights
            weight_list = self.get_node_embedding_weights_from_basic_feat_vecs()
        elif feat_vec_input_info.name == "operator_kind_num":
            # No hand-crafted feat vecs and too large for onehot
            pass
            # weight_list = self.get_embedding_onehot_init_weights_from_feat_mapping(OPERATOR_FEATS)
        elif feat_vec_input_info.name == "type_kind_num":
            # No hand-crafted feat vecs and too large for onehot
            pass
        elif feat_vec_input_info.name == "member_kind_num":
            # Use onehot feat-vecs as init weights:
            weight_list = self.get_embedding_onehot_init_weights_from_feat_mapping(MEMBER_KIND_FEATS)
        elif feat_vec_input_info.name == "value_kind_num":
            # Use onehot feat-vecs as init weights:
            weight_list = self.get_embedding_onehot_init_weights_from_feat_mapping(VALUE_KIND_FEATS)
        else:
            assert False, feat_vec_input_info

        if weight_list is not None:
            max_length_from_weights = max([len(feat_vec) for feat_vec in weight_list])
            assert max_length_from_weights <= pad_to_length, (max_length_from_weights, pad_to_length)
            weights = pad_sequences(weight_list, dtype=np.float32, maxlen=pad_to_length)
            print("Init weights for embedding", feat_vec_input_info.name + ":", "Got", len(weight_list),
                  "feat vecs of max length", max_length_from_weights, "and padded them to array of shape",
                  weights.shape)
            # Embedding layers have one weight matrix. Thus, return the embedding weights as a one-element list of
            # weights:
            return [weights]
        else:
            print("No embedding init weights for", feat_vec_input_info.name)
            return None

    def get_node_embedding_weights_from_basic_feat_vecs(self) -> List[Tuple]:
        weights = []
        for node_kind in self.NODE_KIND_TO_UNIQUE_NUMBER_MAPPING.keys():
            kind_feat, node_kind = self.get_node_kind_feat(node_kind)
            basic_feat_vec = kind_feat.basic_feat_vec
            if basic_feat_vec is None:
                # These will not appear later: Add null vector
                basic_feat_vec = (0,)  # will be expanded by pad_sequences
            weights.append(basic_feat_vec)
        # Embedding layers have one weight matrix. Thus, return the embedding weights as a one-element list of weights:
        return weights  # [pad_sequences(weights, dtype=np.float32)]

    def get_embedding_onehot_init_weights_from_feat_mapping(self, feat_mapping: Dict) -> List[np.ndarray]:
        weights = []
        one_hot_size = len(feat_mapping)
        for feat_number in range(one_hot_size):
            weights.append(int_to_onehot(feat_number, one_hot_size))
        return weights

    # # Replaces Expr with Stmt
    # def normalize_node_kind(self, node_kind):
    #     if node_kind[0].endswith("Expr"):
    #         node_kind = list(node_kind)
    #         node_kind[0] = node_kind[0][:-len("Expr")] + "Stmt"
    #     return node_kind

    def get_unique_node_kind_number(self, node_kind: List):
        """
        Returns an integer that uniquely identifies the given node_kind. This is necessary because Decl's, Stmt's,
        Type's kind number intervals overlap. As there are no Type ast nodes in this context, this should not be used
        for Type kinds.
        :param node_kind:
        :return:
        """
        # The mapping uses tuples because lists can not be used as index
        return self.NODE_KIND_TO_UNIQUE_NUMBER_MAPPING[tuple(node_kind)]

        # coarse_kind = node_kind[0]
        # assert coarse_kind != "Type"
        # fine_kind_number = node_kind[2]
        # if coarse_kind == "Expr":
        #     coarse_kind = "Stmt"
        # # End-of node's kind numbers are greater than all non-end-of node's kind numbers. Rest stays the same.
        # offset = 0 if not is_end_of_node else self.NODE_KIND_NUMBER_NON_ENDOF_MAX_VALUE + 1
        # for kind, kind_interval in self.NODE_KIND_NUMBER_INTERVALS.items():
        #     kind_interval_begin = kind_interval[0]
        #     kind_interval_end = kind_interval[1]
        #     if kind == coarse_kind:
        #         assert kind_interval_begin <= fine_kind_number <= kind_interval_end, fine_kind_number
        #         return offset + fine_kind_number - kind_interval_begin
        #     else:
        #         offset += kind_interval_end - kind_interval_begin + 1  # because intervals are inclusive
        # assert False, "Unknown kind" + coarse_kind

    def get_unique_type_kind_number(self, type_kind: List) -> int:
        """
        Returns an integer that uniquely identifies the given node_kind. This is necessary because Decl's, Stmt's,
        Type's kind number intervals overlap
        :param type_kind:
        :return:
        """
        if type_kind is None:
            return 0
        offset = 1  # because 0 is reserved for None
        assert type_kind[0] == "Type"
        fine_kind_number = type_kind[2]
        type_interval = self.TYPE_KIND_NUMBER_INTERVAL
        assert type_interval[0] <= fine_kind_number <= type_interval[1], fine_kind_number
        type_number = offset + fine_kind_number - type_interval[0]
        assert type_number != 0, (type_number, type_kind)
        return type_number

    def get_node_by_hash(self, translation_unit_node: ClangAstNode, hash: int) -> typing.Optional[ClangAstNode]:
        assert translation_unit_node is not None
        for sub_node, sub_depth, sub_parent in self.depth_search_tree(translation_unit_node):
            if sub_node.id.hash == hash:
                return sub_node
        return None

    def get_kind_counts(self) -> Dict[str, int]:
        return dict(
            node=len(self.NODE_KIND_TO_UNIQUE_NUMBER_MAPPING),
            operator=len(OPERATOR_FEATS),
            type=self.TYPE_KIND_COUNT,
            member=len(MEMBER_KIND_FEATS),
            value=len(VALUE_KIND_FEATS)
        )

    class TypeFeature:
        MAX_TYPE_FEAT_SURROUNDINGS = 5

        def __init__(self):
            # # The type kind's coarse group:
            # grouped_kind: int = 0
            # The (underlying) type kind's fine number corresponding to Clang's TypeClassId:
            self.fine_kind: int = 0
            # The (underlying) type string's looked up number:
            self.qual_name: int = 0
            # # How many times is the underlying type surrounded by references, pointers, arrays,
            # referenceness: int = 0
            # pointerness: int = 0
            # arrayness: int = 0
            # # If the (underlying) type is a constant array type, this will contain its size. If there are nested array
            # # types, the outermost size is used.
            # array_size: int = -1
            # For each type that surrounds the underlying type whether it is a references, pointer and/or array type.
            # If it is a constant array type, also store the array type's size
            self.surroundings = [{
                "referenceness": 0,
                "pointerness": 0,
                "arrayness": 0,
                "array_size": -1
            } for _ in range(self.MAX_TYPE_FEAT_SURROUNDINGS)]

        def __repr__(self):
            return str(self.__dict__)

        def to_tuple(self, add_fine_kind):
            return ((self.fine_kind,) if add_fine_kind else ()) \
                   + (self.qual_name,) \
                   + tuple(itertools.chain(*[surrounding.values() for surrounding in self.surroundings]))
            # + (self.grouped_kind, )

    def get_type_feat_vec(self, node_type: ClangAstNode, depth=0) -> TypeFeature:
        type_feat = self.TypeFeature()
        if node_type is not None:
            assert depth < self.TypeFeature.MAX_TYPE_FEAT_SURROUNDINGS, (depth, node_type)
            type_kind = node_type.kind  # node_type.kind[1] does end with "Type" anymore

            # Get fully qualified type string (sometimes without global namespace specifier)
            qualified_type_string = node_type.id.spelling
            assert node_type.kind[1] != "Elaborated", "ElaboratedType should have been eliminated by tooling"
            # References
            if getattr(node_type, "reference_type", None) is not None:
                type_feat = self.get_type_feat_vec(node_type.reference_type, depth + 1)
                type_feat.surroundings[depth]["referenceness"] = 1
            # Pointers
            elif getattr(node_type, "pointee_type", None) is not None:
                type_feat = self.get_type_feat_vec(node_type.pointee_type, depth + 1)
                type_feat.surroundings[depth]["pointerness"] = 1
            # Functions
            elif getattr(node_type, "result_type", None) is not None:
                assert node_type.kind[1] in ("FunctionNoProto", "FunctionProto"), \
                    "Expected function type"  # NoProto is for functions without params, Proto for functions with params
                # Consider the result type only:
                type_feat = self.get_type_feat_vec(node_type.result_type, depth)
            # Arrays
            elif getattr(node_type, "arr_elem_type", None) is not None:
                type_feat = self.get_type_feat_vec(node_type.arr_elem_type, depth + 1)
                type_feat.surroundings[depth]["arrayness"] = 1
                if getattr(node_type, "array_size", None) is not None:
                    type_feat.surroundings[depth]["array_size"] = node_type.array_size
            # Types to ignore
            elif (qualified_type_string in ("<bound member function type>", "<dependent type>", "<builtin fn type>",
                                            "<overloaded function type>")
                  or node_type.kind[1] in ("TemplateTypeParm", "Decltype", "Auto", "PackExpansion")):
                # ignore
                pass
            # Objects and POD and other stuff
            else:  #
                # plain old datatype (builtin types plus structs (or class without methods/constructors) constructed
                # out of these)
                type_feat.fine_kind = self.get_unique_type_kind_number(type_kind)
                type_feat.qual_name = self.lookup_ref_type_name(qualified_type_string)

                # Check for unknown types:
                if not (node_type.kind[1] in ("Record", "Enum") or node_type.is_pod or qualified_type_string == "void"):
                    if node_type.kind[1] not in ("TemplateSpecialization", "DependentName", "InjectedClassName"):
                        print("Unknown type2", type_kind, qualified_type_string, node_type)
                    if qualified_type_string.startswith("<") and qualified_type_string.endswith(">"):
                        print("Unknown type", type_kind, qualified_type_string, node_type)
        return type_feat

    class KindFeature:
        basic_feat_vec: Tuple = None
        fine_kind: int = 0

        def __repr__(self):
            return str(self.__dict__)

    def get_node_kind_feat(self, node_kind: List[typing.Union[str, int]]) \
            -> Tuple[KindFeature, List[typing.Union[str, int]]]:
        kind_feat = self.KindFeature()
        # Copy node kind because it is changed below:
        actual_node_kind = node_kind
        node_kind = list(node_kind)
        is_end_of_node = node_kind[0].startswith("EndOf")
        if is_end_of_node:
            # Remove "EndOf" prefix:
            temp = node_kind[0].split("EndOf")
            assert len(temp) == 2, temp
            node_kind[0] = temp[1]
            # Feature vector will be modified at the end.

        # Top level kinds:
        if node_kind[0] == "Decl":
            kind_feat.basic_feat_vec = AST_NODE_TOP_LEVEL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.Declaration]
        elif node_kind[0] == "Expr":
            kind_feat.basic_feat_vec = AST_NODE_TOP_LEVEL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.Expression]
        elif node_kind[0] == "Stmt":
            kind_feat.basic_feat_vec = AST_NODE_TOP_LEVEL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.Statement]
        elif node_kind[0] == "AstRoot":
            kind_feat.basic_feat_vec = None
        else:
            assert False, node_kind

        # Ignored Nodes
        if node_kind[0] == "Expr" and node_kind[1] in (
                "ImplicitCastExpr", "FullExpr", "MaterializeTemporaryExpr", "CXXBindTemporaryExpr"):
            # assert False, (node_kind, "should not be here")
            kind_feat.basic_feat_vec = None

        # Compound (e.g. if/else body or scope)
        if node_kind[0:2] == ["Stmt", "CompoundStmt"]:
            kind_feat.basic_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.BlockStatement]

        # Conditional kinds:
        elif node_kind[0:2] == ["Stmt", "IfStmt"]:
            kind_feat.basic_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.IfStatement]
        elif node_kind[0:2] == ["Expr", "ConditionalOperator"]:
            kind_feat.basic_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.TernaryExpression]

        # Loop kinds:
        elif node_kind[0:2] == ["Stmt", "WhileStmt"]:
            kind_feat.basic_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.WhileStatement]
        elif node_kind[0:2] == ["Stmt", "DoStmt"]:
            kind_feat.basic_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.DoStatement]
        elif node_kind[0:2] == ["Stmt", "ForStmt"]:
            kind_feat.basic_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.ForStatement]

        # Break/Jump kinds:
        elif node_kind[0:2] == ["Stmt", "BreakStmt"]:
            kind_feat.basic_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.BreakStatement]
        elif node_kind[0:2] == ["Stmt", "ContinueStmt"]:
            kind_feat.basic_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.ContinueStatement]
        elif node_kind[0:2] == ["Stmt", "ReturnStmt"]:
            kind_feat.basic_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.ReturnStatement]
        elif node_kind[0:2] == ["Stmt", "GotoStmt"]:
            kind_feat.basic_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[
                javalang.tree.ReturnStatement]  # TODO!

        # Member access kinds:
        elif node_kind[0] == "Expr" and node_kind[1] in ("DeclRefExpr", "MemberExpr"):
            kind_feat.basic_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.MemberReference]

        # Call kinds:
        elif node_kind[0] == "Expr" and node_kind[1] in ("CallExpr", "CXXMemberCallExpr", "CXXOperatorCallExpr"):
            kind_feat.basic_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.MethodInvocation]

        if kind_feat.basic_feat_vec is not None:
            # Set end-of node feature in kind feature vector
            if is_end_of_node:
                assert kind_feat.basic_feat_vec[END_OF_NODE_INDEX] == 0, kind_feat.basic_feat_vec
                kind_feat_vec_as_list = list(kind_feat.basic_feat_vec)
                kind_feat_vec_as_list[END_OF_NODE_INDEX] = END_OF_NODE_VALUE
                kind_feat.basic_feat_vec = tuple(kind_feat_vec_as_list)

            # kind number:
            kind_feat.fine_kind = self.get_unique_node_kind_number(actual_node_kind)

            # Remove last two elements from the basic feat vec as they should be always zero
            # (last one was operator kind before)
            assert kind_feat.basic_feat_vec[-1] == 0, kind_feat
            assert kind_feat.basic_feat_vec[-2] == 0, kind_feat
            kind_feat.basic_feat_vec = tuple(kind_feat.basic_feat_vec[0:-2])

        # Also return the adjusted node_kind
        return kind_feat, node_kind

    # C++ string literal prefixes. See https://en.cppreference.com/w/cpp/language/string_literal
    # For R<delim>( <multiline> )<delim> the "R" and delimiters are already removed by Clang.
    # Order matters as they are used as argument to startswith below and u8"foo" would be matched by both "u8" and "u"
    # prefixes
    STRING_LITERAL_PREFIXES = ("L", "u8", "u", "U")

    def get_value_feat_vec_from_literal_value(self, node_value):
        value_feat_vec = [FEAT_VAL_BASIC_LITERAL_KIND, 0]
        if isinstance(node_value, str):
            node_value: str = node_value
            assert len(node_value) >= 2 and node_value[-1] == "\"", \
                "Expected at least opening and closing quotation marks in" + node_value
            # Check for prefixes:
            for string_literal_prefix in self.STRING_LITERAL_PREFIXES:
                if node_value.startswith(string_literal_prefix):
                    # String with prefix. E.g. L"Hello \xffff"
                    # print("string with prefix", node_value)
                    # For now, just remove the prefix. This is not 100% correct as the number of characters consumed
                    # by the escape pattern "\x..." depends on the prefix (and the compiler-dependent size of wchar_t).
                    # TODO: Replace escape pattern by the correct python equivalent depending on the string literal
                    #  prefix and the compiler. E.g. L"\xffff" should be replaced by "\uffff" if sizeof wchar_t == 4.
                    node_value = node_value[len(string_literal_prefix):]
                    break
            # Parse/decode the string literal into a python string:
            # print("string", node_value, ast.literal_eval(node_value))
            parsed_node_value = ast.literal_eval(node_value)
            value_feat_vec[0] = FEAT_VAL_STRING_LITERAL_KIND
            # Convert parsed string to integer using the string lookup:
            value_feat_vec[1] = self.lookup_string(parsed_node_value)
        else:
            # Works for byte, short, int, long, float, double, bool
            value_feat_vec[1] = float(node_value)
        return value_feat_vec

    def get_feat_vec_for_node(self, node: ClangAstNode) -> typing.Optional[Tuple]:
        # ### kind ###
        kind_feat, node_kind = self.get_node_kind_feat(node.kind)  # node_kind is without EndOf and with Expr
        # Skip this node if there is no kind feat vec:
        if kind_feat.basic_feat_vec is None:
            return None

        # Operator kinds:
        operator_feat = 0
        if getattr(node, "operator", None) is not None:
            try:
                operator_feat = OPERATOR_FEATS[node.operator]
            except ValueError:
                print("Unknown operator", node.operator)
                raise

        # ### Variable, member, or method name ###
        member_feat_vec = [0, 0]
        # Either the node itself has a name (e.g. Decl) or it references something with an id (e.g. DeclRefExpr)
        member_id = None
        if node_kind[0] == "Decl":
            member_id = node.id
        elif getattr(node, "ref_id", None):
            member_id = node.ref_id
            # member_node = self.get_node_by_hash(node.tu, member_id.hash)
        if member_id is not None and hasattr(member_id, "spelling"):
            # Variable, member, or method name
            full_id = member_id.spelling
            # Check whether this identifier has been declared in the project or in an included file
            if getattr(member_id, "in_project", True):
                member_kind_feat = FEAT_MEMBER_FILE_OR_PROJECT
                # Use the lookup for project-wide members/variables, which will be reset after each sample
                member_feat = self.lookup_project_var_name(full_id)
            else:
                member_kind_feat = FEAT_MEMBER_SYSTEM_OR_INCLUDED
                # Use the lookup for included/system members/variables. The returned ids are persistent over the whole
                # preprocessing
                member_feat = self.lookup_included_var_name(full_id)
            member_feat_vec = [member_kind_feat, member_feat]

        # ### node value ###
        value_feat_vec = [0, 0]
        node_value = getattr(node, "value", None)
        if node_value is not None:
            value_feat_vec = self.get_value_feat_vec_from_literal_value(node_value)

        # ### type ###
        node_type = getattr(node, "type", None)
        type_feat = self.get_type_feat_vec(node_type)

        # Kind as integer for embedding:
        node_kind_number = kind_feat.fine_kind
        type_kind_number = type_feat.fine_kind  # self.get_unique_type_kind_number(type_kind)
        member_kind_number = list(MEMBER_KIND_FEATS.values()).index(member_feat_vec[0])
        value_kind_number = list(VALUE_KIND_FEATS.values()).index(value_feat_vec[0])
        operator_number = list(OPERATOR_FEATS.values()).index(operator_feat)

        member_number = member_feat_vec[1]  # lookup of qualified member name
        value = value_feat_vec[1]

        # total feat vec:
        return (
            # new feature vector:
                (node_kind_number, operator_number, type_kind_number, member_kind_number, value_kind_number) +
                type_feat.to_tuple(add_fine_kind=False) + (member_number, value) +
                # Old feature vector:
                kind_feat.basic_feat_vec + (operator_feat,) +
                tuple(value_feat_vec) + tuple(member_feat_vec) + type_feat.to_tuple(add_fine_kind=True)
        )

    def get_feat_vec_component_info(self, components_to_return: typing.Union[str, List[str], None] = None) \
            -> List[FeatVecAnnInputInfo]:
        # Ensure component list:
        if components_to_return is not None and not isinstance(components_to_return, list):
            components_to_return = [components_to_return]
        result = []
        last_end_index = 0
        curr_component = None

        # Creates FeatVecAnnInputInfo with given info and increasing index to result list
        def add_feat_fec_info(**kwargs):
            nonlocal last_end_index
            # index_start should not be provided:
            assert kwargs.get("index_start", None) is None
            # Create and append:
            feat_vec_info = FeatVecAnnInputInfo(index_start=last_end_index, **kwargs)
            if components_to_return is None or curr_component in components_to_return:
                result.append(feat_vec_info)
            last_end_index = feat_vec_info.index_end

        curr_component = "kind_numbers"
        kind_counts = self.get_kind_counts()
        suffix = "_kind_num"
        add_feat_fec_info(name="node" + suffix, kind_count=kind_counts["node"])
        add_feat_fec_info(name="operator" + suffix, kind_count=kind_counts["operator"])
        add_feat_fec_info(name="type" + suffix, kind_count=kind_counts["type"])
        add_feat_fec_info(name="member" + suffix, kind_count=kind_counts["member"])
        add_feat_fec_info(name="value" + suffix, kind_count=kind_counts["value"])

        curr_component = "feat_numbers"
        suffix = "_num"
        add_feat_fec_info(name="type_qual_name" + suffix)
        for i in range(self.TypeFeature.MAX_TYPE_FEAT_SURROUNDINGS):
            add_feat_fec_info(name="type_refness" + suffix + "_" + str(i))
            add_feat_fec_info(name="type_ptrness" + suffix + "_" + str(i))
            add_feat_fec_info(name="type_arrness" + suffix + "_" + str(i))
            add_feat_fec_info(name="type_arrsize" + suffix + "_" + str(i))
        add_feat_fec_info(name="member" + suffix),
        add_feat_fec_info(name="value" + suffix),

        curr_component = "basic_feat_vec"
        add_feat_fec_info(name="basic_feat_vec", length=8  # node kind
                                                        + 1  # operator
                                                        + 2  # value
                                                        + 2  # member
                                                        + (2 + self.TypeFeature.MAX_TYPE_FEAT_SURROUNDINGS * 4))
        assert len(result) > 0, ("Unknown component name(s)", components_to_return)
        return result

    @classmethod
    def is_method_definition_node(cls, node):
        return node.kind[0] == "Decl" and node.is_def and node.kind[1] in ("Function", "FunctionTemplate")

    @classmethod
    def is_included_or_imported_node(cls, node, project_path: str = None):
        # Nodes outside project were not added at all in ClangTooling
        return False

    @classmethod
    def get_method_name(cls, method_node):
        return method_node.id.spelling

    @classmethod
    def get_relevant_nodes_from_method_def_node(cls, method_def_node) -> List:
        # TODO: Consider all nodes!?
        # return [method_def_node]

        first = True
        result_nodes = []
        for sub_node, sub_depth, sub_parent in cls.depth_search_tree(method_def_node):
            # Only look at direct children:
            if sub_depth != 1:
                continue
            # Look for return value or parameter:
            if first and sub_node.kind[0:2] == ["Decl", "ParmVar"]:
                continue
            # Found normal node. Add it:
            first = False
            result_nodes.append(sub_node)
        return result_nodes  # usually only one CompoundStatement

    @classmethod
    def add_called_methods_to_list(cls, method_def_node, list_to_add, all_methods_in_test_case):
        for sub_node, sub_depth, sub_parent in cls.depth_search_tree(method_def_node):
            # Look for calls:
            if sub_depth > 0 and sub_node.kind[0:2] == ["Expr", "CallExpr"]:
                # Try to get the callee:
                callee_method_id = getattr(sub_node, "ref_id", None)
                if callee_method_id:
                    # Clang was able to determine the callee. If it has a name, spelling will contain a fully qualified
                    # id (including file path). Otherwise callee_method_name will be set to None
                    callee_method_name = getattr(sub_node.ref_id, "spelling", None)
                else:
                    # Callee not available. E.g. in case of call through function pointer
                    callee_method_name = None

                if callee_method_name is not None:
                    # Use the callee method's fully qualified semantic name (without lexical file path information) to
                    # find the corresponding method definition node if it is part of the test case's AST. This must be
                    # done without lexical information as a test case may consists of multiple translation units. Thus,
                    # a name may be declared and used in TU A and defined in another TU B. In this case, ref_id refers
                    # to the declaration in A as no definition is available in A. all_methods_in_test_case contains
                    # the corresponding definition from B. Thus, the semantic information is identical, but the lexical
                    # differs in such a case.
                    callee_method_semantic_name = cls.get_semantic_name(callee_method_name)
                    matching_callee_method_def_nodes = [method_def_node
                                                        for method_def_node, method_semantic_name
                                                        in all_methods_in_test_case
                                                        if method_semantic_name == callee_method_semantic_name]
                    if len(matching_callee_method_def_nodes) == 1:
                        # Found callee method definition:
                        callee_method_def_node = matching_callee_method_def_nodes[0]
                        # Add it to list of found callee method def nodes:
                        if callee_method_def_node not in list_to_add:
                            # print("Found call ", callee_method_name, callee_method_def_node)
                            list_to_add.append(callee_method_def_node)
                            # Recursively look for calls in definition of callee:
                            cls.add_called_methods_to_list(callee_method_def_node, list_to_add,
                                                           all_methods_in_test_case)
                    elif len(matching_callee_method_def_nodes) == 0:
                        # Call to method which is not defined in test case itself. E.g. writeLine. This should not
                        # happen for test case related methods.

                        # TODO: The following is Juliet dependent and therefore commented out. Maybe add this somewhere
                        #  else or pass check function from outside?!
                        # # Extract the actual method name (without qualifiers and file name) and assert that it is not
                        # # test case related. A test case related method have CWE and/or bad and/or good in its name
                        # if "CWE" in callee_method_semantic_name \
                        #         or "bad" in callee_method_semantic_name \
                        #         or "good" in callee_method_semantic_name:
                        #     assert False, (callee_method_semantic_name,
                        #                    callee_method_name,
                        #                    [mname for mnode, mname in all_methods_in_test_case])

                        # Do nothing
                        pass
                    else:
                        # There shouldn't be more than one matching callee method definition nodes:
                        assert False, matching_callee_method_def_nodes

    @classmethod
    def depth_search_tree(cls, node: ClangAstNode, skip_included: typing.Union[bool, str] = False,
                          verbose: bool = False, depth: int = 0, parent=None):
        if node is None:
            return

        yield node, depth, parent

        for child in node.children:
            yield from cls.depth_search_tree(child, skip_included=skip_included, verbose=verbose, depth=depth + 1,
                                             parent=node)

    @classmethod
    def get_node_as_string(cls, node: ClangAstNode, compact=False) -> str:
        if compact:
            return node.get_compact_repr()
        else:
            return str(node)

    @classmethod
    def add_source_code_to_ast(cls, ast: ParserFrontend.AstNodeType, source_code: str, source_extension: str,
                               temp_file_path: str = None):
        """
        Adds the source code to the given ast for comprehensibility since a temporary file path does not allow to
         gather the source code the ast was parsed from.
        :param ast:
        :param source_code:
        :param source_extension:
        :param temp_file_path:
        :return:
        """
        # Make sure that temp file path matches path in ast (if given);
        if isinstance(ast.tu.id.spelling, str):
            if temp_file_path is not None:
                assert ast.tu.id.spelling == temp_file_path, (ast.tu.id.spelling, temp_file_path)
            # Add the source code to the resulting ast for comprehensibility
            ast.tu.id.spelling = (ast.tu.id.spelling, source_code, source_extension)
        else:
            assert ast.tu.id.spelling[1] == source_code and ast.tu.id.spelling[2] == source_extension, \
                (ast.tu.id.spelling, source_code, source_extension)

    def get_ast_from_source(self, source_code, source_extension, *args, **kwargs) -> ClangAstNode:
        assert source_extension.startswith("."), source_extension + " must be an extension in format .<ext>"
        with tempfile.NamedTemporaryFile("w", dir=self.project_path, suffix=source_extension) as temp_input_file:
            temp_input_file.write(source_code)
            temp_input_file.flush()
            resulting_ast = self.get_ast_from_source_path(temp_input_file.name, *args, **kwargs)
            self.add_source_code_to_ast(resulting_ast, source_code, source_extension, temp_input_file.name)
            return resulting_ast

    def get_ast_from_source_path(self, source_file_path: str, context=None, verbosity=1, add_static_kind_info=False) \
            -> ClangAstNode:
        if verbosity > 2:
            print("Parsing", source_file_path, "with content\n", read_file_content(source_file_path))
        return self.get_asts_from_source_paths(
            source_file_paths_with_context={"source_path_collections": [source_file_path], "context": context},
            verbosity=verbosity,
            add_static_kind_info=add_static_kind_info
        )[0]

    def get_static_kind_info(self):
        ast_root_node = self.get_ast_from_source("int main() { }", ".cpp", verbosity=0, add_static_kind_info=True)
        assert ast_root_node, "Parse error while getting static kind info"
        return ast_root_node.kind_number_intervals, ast_root_node.kinds

    def get_asts_from_source_paths(self, source_file_paths_with_context: Dict[str, typing.Union[List[str], Dict]],
                                   **kwargs) -> List[ClangAstNode]:
        # Make one-element source file path collections out of list of source file paths:
        source_file_path_collections_with_context = copy.deepcopy(source_file_paths_with_context)
        source_file_path_collections_with_context["source_path_collections"] = \
            [[source_file_path] for source_file_path
             in source_file_path_collections_with_context["source_path_collections"]]
        list_of_result_asts = self.get_asts_from_source_path_collections(source_file_path_collections_with_context,
                                                                         **kwargs)
        # Return each collection's first ast only
        return [result_asts[0] for result_asts in list_of_result_asts]

    def annotate_ast(self, ast: ClangAstNode) -> None:
        for node, depth, parent in self.depth_search_tree(ast):
            node.tu = ast.children[0]
            node.depth = depth
            node.parent = parent

    def get_asts_from_source_path_collections(self, source_file_path_collections_with_context: Dict[
        str, typing.Union[List[str], Optional[Dict]]],
                                              annotate_ast: bool = True, verbosity=1, add_end_nodes=True,
                                              add_static_kind_info=False) -> List[List[Optional[ClangAstNode]]]:
        context_specific_includes = []
        # Get list of list of file paths:
        source_file_path_collections = source_file_path_collections_with_context["source_path_collections"]
        # Get context of file paths (e.g. requirements for successful parsing)
        context = source_file_path_collections_with_context.get("context", None)
        if context is None:
            # Ensure that context is a dict even if there is no context key or context value is None:
            context = {}

        # The amount of how much the files need windows environment (0^=no windows, 1^=a bit of windows, ...(see below))
        needs_windows = int(context.get("needs_windows", False))

        # Write list of file paths, which will be the input for tooling:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as input_path_list_file:
            # Each line contains a source file path collection (i.e., an array of source file paths)
            input_path_list_file.write("\n".join([to_json(source_file_path_collection)
                                                  for source_file_path_collection in source_file_path_collections]))
            input_path_list_file.flush()

            # Generate tooling output file path:
            with tempfile.NamedTemporaryFile("r", suffix=".jsonl") as output_json_file:
                # Arguments for call (order matters)
                args = [input_path_list_file.name,
                        output_json_file.name,
                        self.project_path if self.project_path is not None else "",
                        "true" if add_end_nodes else "false",
                        "true" if add_static_kind_info else "false",
                        "true" if verbosity > 2 else "false",
                        "--"]
                if verbosity > 2:
                    args.append("-v")

                # Windows environment handling:
                if needs_windows >= 1:
                    # Add Microsoft-related includes, defines, and compiler args depending on the windows'ness needed.
                    # ORDER OF INCLUDES MATTERS!

                    # needs_windows >= 1: Add "windows" includes and missing defines
                    context_specific_includes.append(os.path.join(ENVIRONMENT.wine_include_root_path, "windows"))
                    args.extend([
                        # It is important to explicitly specify a version
                        # (https://stackoverflow.com/questions/34531071/clang-cl-on-windows-8-1-compiling-error)
                        # Version number corresponds to cl.exe version output.
                        # See: https://en.wikipedia.org/wiki/Microsoft_Visual_C%2B%2B#Internal_version_numbering
                        "-fms-compatibility-version=16.00",  # 16.00 ^= VS 2010 (as in Juliet User Guide)
                        # For CWE90: (from https://github.com/delphij/openldap/blob/master/include/ldap.h)
                        "-DLDAP_PORT=389",
                        "-DLDAP_NO_LIMIT=0",
                        # For CWE256
                        # (from https://github.com/tpn/winsdk-10/blob/master/Include/10.0.10240.0/um/WinBase.h)
                        "-DLOGON32_PROVIDER_DEFAULT=0",
                        "-DLOGON32_LOGON_NETWORK=3",
                        # "-v"
                    ])
                    if needs_windows >= 2:
                        # Also add "msvcrt" include (this is needed by some test cases but also breaks some test cases
                        #  that only need "windows" include)
                        context_specific_includes.append(os.path.join(ENVIRONMENT.wine_include_root_path, "msvcrt"))
                    if needs_windows >= 3:
                        args.append("-fno-builtin")
                    if needs_windows >= 4:
                        args.extend(["-D_WIN32",
                                     "-DWIN32",
                                     # "-m64",
                                     # "-fshort-wchar",
                                     # "-DWINE_UNICODE_NATIVE",
                                     # "-D_REENTRANT -fPIC",
                                     # "-DWIN64",
                                     # "-D_WIN64",
                                     # "-D__WIN64",
                                     # "-D__WIN64__",
                                     # "-DWIN32",
                                     # "-D_WIN32",
                                     # "-D__WIN32",
                                     # "-D__WIN32__",
                                     # "-D__WINNT",
                                     # "-D__WINNT__",
                                     # "-D__stdcall=__attribute__((ms_abi))",
                                     # "-D__cdecl=__attribute__((ms_abi))",
                                     # "-D_stdcall=__attribute__((ms_abi))",
                                     # "-D_cdecl=__attribute__((ms_abi))",
                                     # "-D__fastcall=__attribute__((ms_abi))",
                                     # "-D_fastcall=__attribute__((ms_abi))",
                                     # "-D__declspec(x)=__declspec_##x",
                                     # "-D__declspec_align(x)=__attribute__((aligned(x)))",
                                     # "-D__declspec_allocate(x)=__attribute__((section(x)))",
                                     # "-D__declspec_deprecated=__attribute__((deprecated))",
                                     # "-D__declspec_dllimport=__attribute__((dllimport))",
                                     # "-D__declspec_dllexport=__attribute__((dllexport))",
                                     # "-D__declspec_naked=__attribute__((naked))",
                                     # "-D__declspec_noinline=__attribute__((noinline))",
                                     # "-D__declspec_noreturn=__attribute__((noreturn))",
                                     # "-D__declspec_nothrow=__attribute__((nothrow))",
                                     # "-D__declspec_novtable=__attribute__(())",
                                     # "-D__declspec_selectany=__attribute__((weak))",
                                     # "-D__declspec_thread=__thread",
                                     # "-D__int8=char",
                                     # "-D__int16=short",
                                     # "-D__int32=int",
                                     # "-D__int64=long",
                                     # "-D__WINE__", # "-v",
                                     # "-fms-compatibility-version=16.00",
                                     # "-fms-extensions",
                                     # "-fno-builtin"
                                     ])
                    if needs_windows >= 5:
                        args.append("-fms-extensions")

                    if needs_windows >= 6:
                        # Too much windows needed. Give up:
                        assert len(source_file_path_collections) == 1
                        return [[None]]

                # Add compiler args
                args.extend(self.compiler_args)
                if verbosity > 2:
                    # Show all warnings:
                    args = [arg for arg in args if arg != "-Wno-everything"]

                # Add includes:
                args.extend(["-I" + include_path + "/"
                             for include_path in context_specific_includes + self.include_paths])

                # Run ClangAstToJson tool:
                command = [os.path.join(ENVIRONMENT.code_root_path, "ClangAstToJson", "ClangAstToJson")] + args
                if verbosity > 1:
                    print("Parsing", len(source_file_path_collections), "file collection(s) with context", context,
                          "(command:", command, ") ...")
                result = subprocess.run(
                    command,
                    # check=True, # throw exception on non zero exit code
                    stdout=(subprocess.PIPE if verbosity > 0 else subprocess.DEVNULL),  # capture stdout
                    stderr=(subprocess.PIPE if verbosity > 0 else subprocess.DEVNULL),  # capture stderr
                    # input=source_code,
                    encoding="utf-8"  # encode stdout value
                )

                if verbosity > 0:
                    # Print subprocess' stdout and stderr in python (instead of in subprocess) to allow Tee to work
                    print(result.stdout, end="")  # no newline
                    print(result.stderr, file=sys.stderr, end="")  # no newline

                if result.returncode != 0:
                    # This should not happen because clang tool returns "null" line in case of parse error!
                    assert False, "Error in ClangTooling: " + str(result.returncode)

                # Read JSON: Each line contains a json string containing a list of asts
                result_json_lines_str = output_json_file.read()
                result_json_lines = result_json_lines_str.split("\n")
                assert result_json_lines[-1] == "", "Last Json line is expected to be empty"
                assert (len(result_json_lines) - 1) == len(source_file_path_collections)

                if verbosity > 2:
                    print(result_json_lines_str)

                # Read JSON lines, convert JSON objects to ClangAstNodes
                #  and annotate ast with TU, parent, and depth info if requested:
                def annotate_if_requested(asts):
                    # Check whether kind numbers match:
                    # assert self.KIND_NUMBER_INTERVALS == ast.kind_numbers,\
                    #     (ast.kind_numbers, self.KIND_NUMBER_INTERVALS)
                    for ast in asts:
                        if annotate_ast:
                            self.annotate_ast(ast)
                    return asts

                # Create ClangAstNode objects out of json objects:
                result_json_objs = [annotate_if_requested(from_json(json_line, self.ClangAstNode)) for json_line in
                                    result_json_lines if len(json_line) > 0]  # Ignore last empty line
                assert len(result_json_objs) == len(source_file_path_collections)

                # For each source path collection: Increase windows-ness if need_windows is specified and parse failed:
                for index, result_json_obj, source_file_path_collection in zip(range(len(result_json_objs)),
                                                                               result_json_objs,
                                                                               source_file_path_collections):
                    if None in result_json_obj and needs_windows > 0:
                        # Try to parse the file with more windows'ness:
                        sub_result = self.get_asts_from_source_path_collections(
                            # Only the current collection and increased windows'ness:
                            {"source_path_collections": [source_file_path_collection],
                             "context": {"needs_windows": needs_windows + 1}},
                            annotate_ast=annotate_ast, verbosity=verbosity if verbosity > 1 else 0,
                            add_end_nodes=add_end_nodes
                        )[0]
                        if None not in sub_result:
                            # Success: replace failed parse result (i.e. None) with parsed ast:
                            result_json_objs[index] = sub_result
                        elif verbosity > 0:
                            print("Unable to parse", source_file_path_collection)
                return result_json_objs


class ClangToolingSimple(ClangTooling):

    def get_feat_vec_for_node(self, node: ClangTooling.ClangAstNode) -> typing.Optional[Tuple]:
        # ### kind ###
        kind_feat, node_kind = self.get_node_kind_feat(node.kind)  # node_kind is without EndOf and with Expr
        # Skip this node if there is no kind feat vec:
        if kind_feat.basic_feat_vec is None:
            return None

        if node_kind[0] == "Expr" and node_kind[1] == "IntegerLiteral":
            # ### node value ###
            value_feat_vec = [0, 0]
            node_value = getattr(node, "value", None)
            assert node_value is not None
            value_feat_vec = self.get_value_feat_vec_from_literal_value(node_value)
            return value_feat_vec[1], value_feat_vec[1]
        elif node_kind[0] == "Decl" and node_kind[1] == "Var":
            # ### type ###
            node_type = getattr(node, "type", None)
            type_feat = self.get_type_feat_vec(node_type)
            arr_size = type_feat.surroundings[0]["array_size"]
            if arr_size >= 0:
                return arr_size, arr_size
        return None

    def get_feat_vec_component_info(self, components_to_return: typing.Union[str, List[str], None] = None):
        # Ensure component list:
        if components_to_return is not None and not isinstance(components_to_return, list):
            components_to_return = [components_to_return]
        result = []
        last_end_index = 0
        curr_component = None

        # Creates FeatVecAnnInputInfo with given info and increasing index to result list
        def add_feat_fec_info(**kwargs):
            nonlocal last_end_index
            # index_start should not be provided:
            assert kwargs.get("index_start", None) is None
            # Create and append:
            feat_vec_info = FeatVecAnnInputInfo(index_start=last_end_index, **kwargs)
            if components_to_return is None or curr_component in components_to_return:
                result.append(feat_vec_info)
            last_end_index = feat_vec_info.index_end

        curr_component = "feat_numbers"
        add_feat_fec_info(name="value_num")

        curr_component = "basic_feat_vec"
        add_feat_fec_info(name="basic_feat_vec")

        return result


# Enable functionality for parsing of C/C++ source code with libclang. This was only used in an early stage of the tool
#  and is not supported anymore
ENABLE_LIBCLANG_PARSING = False
if ENABLE_LIBCLANG_PARSING:
    import clang
    from clang.cindex import Index, CursorKind, TypeKind, TranslationUnit, Cursor, Token, SourceRange, SourceLocation

    # !apt-get install libc++-dev  # TODO: necessary?
    # !apt-get install libc-dev  # TODO: necessary?
    # !apt-get install llvm-dev  # TODO: necessary? # llvm llvm-6.0 llvm-6.0-dev llvm-6.0-runtime llvm-dev llvm-runtime
    # !apt-get install clang
    # !apt-get install libclang-dev  # libclang-6.0-dev libclang-dev
    # !apt-get install python-clang-6.0  # python-clang-6.0
    # # python-clang-6.0 only installs for python 2. Create symlink for python 3:
    # !cd /usr/lib/python3/dist-packages/ && sudo ln --symbolic /usr/lib/python2.7/dist-packages/clang
    # clang
    # See https://github.com/llvm-mirror/clang/blob/master/bindings/python/clang/cindex.py for documentation

    # ! echo -e "#include <stddef.h>\nint main(){}" > test.cpp
    # Output necessary includes (which need to be entered in include_paths in LibClang frontend class below)
    # !clang -xc -v

    INPUT_FILE_NAME = "input"


    class LibClang(ParserFrontend):
        include_paths = ENVIRONMENT.system_include_paths  # TODO: Maybe llvm-6 is necessary

        class ClangObject:

            def __init__(self, attributes=None):
                assert attributes != {}, "Empty attributes for ClangObject"
                if attributes is None:
                    attributes = {}
                self.__dict__ = attributes

            def __repr__(self):
                # Do not use exluded attributes for representation (because they may result in stack overflow):
                filtered = {key: val for key, val in self.get_dict().items() if
                            key not in getattr(self, "exclude_for_str", [])}
                return str(filtered)

            def get_dict(self):
                return self.__dict__

        class ClangKind(ClangObject):
            pass

        class ClangType(ClangObject):
            # Exclude potentially recursive attributes:
            exclude_for_str = ["canonical", "result_type", "argument_types", "named_type", "array_type", "array_size",
                               "class_type", "pointee_type", "is_const", "is_pod_type"]

            # @property
            # def kind(self):
            #     return TypeKind(self.kind_val)

            def get_result(self):
                return self.result_type

            def is_pod(self):
                return self.is_pod_type

            # def get_canonical(self):
            #     return self.canonical

            def get_pointee(self):
                return self.pointee_type

        class ClangLocation(ClangObject):
            pass

        class ClangLocationFile(ClangObject):
            pass

        class ClangNode(ClangObject):
            # Exclude potentially recursive attributes:
            exclude_for_str = ["children", "translation_unit"]

            def is_definition(self):
                return self.is_def

            def get_children(self):
                return self.children

        def __init__(self, include_paths=None):
            if include_paths is not None:
                self.include_paths.extend(include_paths)
            super().__init__()

        def get_feat_vec_for_node(self, node: ClangNode) -> Tuple:
            node_kind = CursorKind.from_id(node.kind.value)

            # kind
            kind_feat_vec = None
            if (node_kind.is_unexposed()):
                # Not part of Python LibClang Bindings / An expression whose specific kind is not exposed via libclang interface.
                # Probably implicit cast nodes?
                # E.g.: Unexposed declarations have the same operations as any other kind of
                # declaration; one can extract their location information, spelling, find their
                # definitions, etc. However, the specific kind of the declaration is not
                # reported.
                kind_feat_vec = None
            elif (node_kind.is_invalid()):
                kind_feat_vec = None
            elif (node_kind.is_attribute()):
                kind_feat_vec = None
            elif (node_kind.is_preprocessing()):
                kind_feat_vec = None  # TODO keep these?
            elif (node_kind == CursorKind.COMPOUND_STMT):  # { stmt, stmt, ... }
                kind_feat_vec = None

            # Top level kinds:
            elif (node_kind.is_declaration()):
                kind_feat_vec = AST_NODE_TOP_LEVEL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.Declaration]
            elif (node_kind.is_expression()):
                kind_feat_vec = AST_NODE_TOP_LEVEL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.Expression]
            elif (node_kind.is_statement()):
                kind_feat_vec = AST_NODE_TOP_LEVEL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.Statement]

            # Conditional kinds:
            if (node_kind == CursorKind.IF_STMT):
                kind_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.IfStatement]
            elif (node_kind == CursorKind.CONDITIONAL_OPERATOR):
                kind_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.TernaryExpression]

            # Operator kinds:
            elif (node_kind in (CursorKind.UNARY_OPERATOR, CursorKind.ARRAY_SUBSCRIPT_EXPR, CursorKind.BINARY_OPERATOR,
                                CursorKind.COMPOUND_ASSIGNMENT_OPERATOR)):
                kind_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.BinaryOperation]
                # get operator as string and replace last feat vec element
                operator_string = node.operator_str  # self.get_operator_str(node)
                kind_feat_vec = self.get_feat_fec_with_operator_feature(kind_feat_vec, operator_string)

            # Loop kinds:
            elif (node_kind == CursorKind.WHILE_STMT):
                kind_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.WhileStatement]
            elif (node_kind == CursorKind.DO_STMT):
                kind_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.DoStatement]
            elif (node_kind == CursorKind.FOR_STMT):
                kind_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.ForStatement]

            # Break/Jump kinds:
            elif (node_kind == CursorKind.BREAK_STMT):
                kind_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.BreakStatement]
            elif (node_kind == CursorKind.CONTINUE_STMT):
                kind_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.ContinueStatement]
            elif (node_kind == CursorKind.RETURN_STMT):
                kind_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.ReturnStatement]

            # Member access kinds:
            elif (node_kind in (CursorKind.DECL_REF_EXPR, CursorKind.MEMBER_REF_EXPR)):
                kind_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.MemberReference]

            # Call kinds:
            elif (node_kind == CursorKind.CALL_EXPR):
                kind_feat_vec = AST_NODE_SPECIAL_KINDS_TO_FEAT_VEC_MAPPING[javalang.tree.MethodInvocation]

            if (kind_feat_vec is None):
                return None

            # node value:
            value_feat_vec = [0, 0]
            node_value = node.value  # self.get_literal_node_value(node)
            if (node_value is not None):
                value_feat_vec = self.get_value_feat_vec_from_literal_value(node_value)

            # Variable, member, or method name
            member_feat_vec = [0, 0]
            if (node_kind in (CursorKind.DECL_REF_EXPR, CursorKind.MEMBER_REF_EXPR)):
                full_id = node.spelling
                if (node_kind == CursorKind.MEMBER_REF_EXPR):
                    node_children = list(node.get_children())
                    if (len(node_children) == 1 and node_children[0].kind == CursorKind.DECL_REF_EXPR):
                        full_id = node_children[0].spelling + "." + full_id
                # TODO: INCLUDED VS LOCAL?
                member_kind_feat = FEAT_MEMBER_FILE_OR_PROJECT  # assume TU first
                member_feat = self.lookup_project_var_name(full_id)
                member_feat_vec = [member_kind_feat, member_feat]

            # type
            def get_type_feat_vec(node_type):
                type_kind_feat, type_feat = [0, 0]
                if node_type is not None:
                    # node_type = node_type.get_canonical()  # remove sugar
                    qualified_type_string = node_type.spelling  # fully qualified but without global namespace specifier
                    if (node_type.kind.name in ("INVALID", "UNEXPOSED", "VOID")):
                        type_kind_feat = 0
                        type_feat = 0
                    elif (node_type.kind.name in ("RECORD",
                                                  "ELABORATED")):  # RECORD if just Class name is written, Elaborated if qualified Class name is written
                        type_kind_feat = FEAT_TYPE_REFERENCE
                        type_feat = len(TypeKind._kinds) + 1 + self.lookup_ref_type_name(qualified_type_string)
                    elif (node_type.kind.name == "POINTER"):
                        type_kind_feat, type_feat = get_type_feat_vec(node_type.get_pointee())
                        type_kind_feat = FEAT_TYPE_POINTER
                    elif (node_type.kind.name in ("FUNCTIONNOPROTO",
                                                  "FUNCTIONPROTO")):  # FUNCTIONNOPROTO is for functions without params, FUNCTIONPROTO for functions with params:
                        result_type = node_type.get_result()  # returns result type
                        # Consider the result type only:
                        type_kind_feat, type_feat = get_type_feat_vec(result_type)
                    elif (node_type.is_pod()):
                        # plain old datatype (basic types plus structs (or class without methods/constructors) constructed
                        # out of these)
                        type_kind_feat = FEAT_TYPE_BASIC
                        # Use id used by clang:
                        type_feat = node_type.kind.value
                    else:
                        print("Unknown type", node_type.kind.name, qualified_type_string)
                return [type_kind_feat, type_feat]

            type_feat_vec = get_type_feat_vec(node.type)

            # total feat vec:
            return kind_feat_vec + tuple(value_feat_vec + member_feat_vec + type_feat_vec)

        @classmethod
        def is_method_definition_node(cls, node):
            # is_definition() to avoid forward declarations
            return node.kind in (CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD) and node.is_definition()

        @classmethod
        # @static_vars(cache={})
        def is_included_or_imported_node(cls, node: Cursor, project_path: str = None):
            if node.location.file is not None:
                if project_path is not None:
                    # project_cache = cls.is_included_or_imported_node.cache.setdefault(project_path, {})
                    #
                    file_path = node.location.file.name
                    is_included = not file_path.startswith(project_path)
                    # assert is_included == file_path.startswith("/usr/")
                    return is_included
                    # # if project_cache.get(file_path, None) is None:
                    # # Check whether current path is in project
                    # project_path = pathlib.Path(project_path)
                    # file_path = pathlib.Path(file_path)
                    # is_included = project_path.parents not in file_path.parents
                    # return is_included
                    # project_cache[file_path] = is_included
                    # return project_cache[file_path]
                else:
                    # Check whether the current node is part of the current translation unit:
                    assert node.translation_unit.spelling is not None
                    return node.location.file.name != node.translation_unit.spelling
            else:
                # If there is no location, there should be no source range
                assert node.extent.start is None
                # No location: either included or translation unit (which is never included)
                return node.kind != CursorKind.TRANSLATION_UNIT

        @classmethod
        def get_method_name(cls, method_node):
            return method_node.spelling

        @classmethod
        def get_relevant_nodes_from_method_def_node(cls, method_def_node: Cursor) -> List[Cursor]:
            first = True
            result_nodes = []
            for sub_node, sub_depth, sub_parent in cls.depth_search_tree(method_def_node):
                # Only look at direct children:
                if sub_depth != 1:
                    continue
                # Look for return value or parameter:
                if first and sub_node.kind == CursorKind.PARM_DECL:
                    continue
                # Found normal node. Add it:
                first = False
                result_nodes.append(sub_node)
            return result_nodes  # usually only one COMPOUND_STATEMENT

        @classmethod
        def add_called_methods_to_list(cls, method_node, list_to_add, all_methods_in_content):
            for sub_node, sub_depth, sub_parent in cls.depth_search_tree(method_node):
                if (sub_depth > 0 and sub_node.kind == CursorKind.CALL_EXPR):
                    called_method_name = sub_node.spelling  # get the name of the called method
                    # check whether there is a method that has a matching name and is in content
                    called_method_node = next((method_node for method_node, method_name in all_methods_in_content if
                                               method_name == called_method_name), None)
                    if (called_method_node is None):
                        # print("Call to method which is not defined in current content: " + called_method_name) # e.g. writeLine
                        continue
                    elif (called_method_node not in list_to_add):
                        list_to_add.append(called_method_node)
                        cls.add_called_methods_to_list(called_method_node, list_to_add, all_methods_in_content)

        # Generator of all nodes in depth search pre-order
        # returns node, depth, node's parent
        # Alternatively, Clangs method could be used: node.walk_preorder() (but wihtout depth and parent)
        # skip_included can be provided a path to skip includes that are not below the given path
        @classmethod
        def depth_search_tree(cls, node, skip_included: typing.Union[bool, str] = False, verbose: bool = False,
                              depth: int = 0, parent=None):
            if node is None:
                return

            if type(skip_included) == str:
                skip = cls.is_included_or_imported_node(node, project_path=skip_included)
            elif not skip_included:
                skip = False
            elif skip_included:
                skip = cls.is_included_or_imported_node(node, project_path=None)
            else:
                assert False, str(skip_included) + " not allowed for skip_included"

            if not skip:
                yield node, depth, parent

                for child in node.get_children():
                    yield from cls.depth_search_tree(child, skip_included=skip_included, verbose=verbose,
                                                     depth=depth + 1,
                                                     parent=node)
            elif verbose:
                print("depth_search_tree: Skipped node (and subtree) " + str(cls.get_node_as_string(node)))

        @classmethod
        # Use cache by call and not per script execution (therefore do not use decorator)
        def get_type_as_pickable(cls, type: clang.cindex.Type, recursive: bool = True, cache=None) -> ClangType:
            if type.kind == TypeKind.INVALID:
                return None

            key = str(type.kind.value) + " " + type.spelling
            decl = type.get_declaration()
            if decl is not None:
                key += " " + str(decl.hash)

            if cache is None:
                cache = {}
            # Get the value for key and set it to {} if it does not exist yet
            clang_type = cache.setdefault(key, cls.ClangType())
            if not clang_type.get_dict():  # dict evaluates to false if empty
                # Mark that this type info will be filled before start to fill it to avoid stack overflow
                # Use kind (info that can be detmined without recursion  problems) as indicator whether the
                # info is already going to be filled:
                clang_type.get_dict().update({
                    "kind": cls.get_kind_as_pickable(type.kind, recursive=recursive),
                    "spelling": str(type.spelling),
                    "is_const": bool(type.is_const_qualified()),
                    "is_pod_type": bool(type.is_pod()),
                    "array_size": int(type.get_array_size()),
                })
                # No fill the recursive info:
                if recursive:
                    clang_type.get_dict().update({
                        # "canonical": cls.get_type_as_pickable(type.get_canonical(), recursive=recursive, cache=cache),
                        "result_type": cls.get_type_as_pickable(type.get_result(), recursive=recursive, cache=cache),
                        "argument_types": [cls.get_type_as_pickable(arg_type, recursive=recursive, cache=cache) for
                                           arg_type
                                           in
                                           type.argument_types()] if type.kind == TypeKind.FUNCTIONPROTO else None,
                        "named_type": cls.get_type_as_pickable(type.get_named_type(), recursive=recursive, cache=cache),
                        "array_type": cls.get_type_as_pickable(type.get_array_element_type(), recursive=recursive,
                                                               cache=cache),
                        "class_type": cls.get_type_as_pickable(type.get_class_type(), recursive=recursive, cache=cache),
                        "pointee_type": cls.get_type_as_pickable(type.get_pointee(), recursive=recursive, cache=cache)
                    })
            return clang_type

        @classmethod
        def get_location_as_pickable(cls, location, recursive: bool = True) -> ClangLocation:
            location_file = location.file
            return cls.ClangLocation({
                "file": cls.ClangLocationFile({"name": location_file.name}) if location_file is not None else None,
                "column": location.column,
                "line": location.line
            })

        @classmethod
        def get_kind_as_pickable(cls, kind, recursive: bool = True) -> ClangKind:
            return cls.ClangKind({
                "type": type(kind).__name__,
                "name": str(kind.name),
                "value": int(kind.value)
            })

        @classmethod
        def get_node_as_pickable(cls, node: Cursor, recursive: bool = True, context=None) -> ClangNode:
            assert False, "REIMPLEMENT!"
            key = str(node.hash)

            if context is None:
                context = {"cache": {}, "id": 0}
            # Get the value for key and set it to {} if it does not exist yet
            clang_node = context["cache"].setdefault(key, cls.ClangNode())
            if not clang_node.get_dict():  # dict evaluates to false if empty
                # Use id (info that can be detmined without recursion  problems) as indicator whether the
                # info is already going to be filled:
                clang_node.get_dict()["id"] = context["id"]
                context["id"] += 1

                clang_node.get_dict().update({
                    "kind": cls.get_kind_as_pickable(node.kind, recursive=recursive),
                    "hash": int(node.hash),
                    # depth does not make sense because the same node may appear multiple times in the AST at
                    #  different depths! (e.g. NAMESPACE_REF)
                    # "depth": cls.get_node_depth(node, exhaustive=False),
                    'spelling': str(node.spelling),
                    'displayname': str(node.displayname),
                    'operator_str': cls.get_operator_str(node),
                    'type': cls.get_type_as_pickable(node.type.get_canonical(), recursive=recursive),  # remove sugar
                    "value": cls.get_literal_node_value(node),
                    # "tokens": cls.get_token_strings(node),
                    "location": cls.get_location_as_pickable(node.location),
                    'extent.start': cls.get_location_as_pickable(node.extent.start, recursive=recursive),
                    'extent.end': cls.get_location_as_pickable(node.extent.end, recursive=recursive),
                    'is_def': bool(node.is_definition()),
                    # Never do recursion on translation unit because the whole TU (including included files!) will be translated then!
                    "translation_unit": cls.get_node_as_pickable(node.translation_unit.cursor, recursive=False,
                                                                 context=context),
                })
                # No fill the recursive info:
                if recursive:
                    clang_node.get_dict().update({
                        "children": [cls.get_node_as_pickable(child, recursive=recursive, context=context) for child in
                                     node.get_children()],
                    })

            return clang_node

        @classmethod
        def get_node_as_string(cls, node: typing.Union[javalang.ast.Node, Cursor]) -> str:
            if type(node) == Cursor:
                node = cls.get_node_as_pickable(node, recursive=False)
            return str(node)

        @classmethod
        # Return the lexical depth of the given node
        # exhaustive specifies whether the complete translation unit should be searched if lexical parent fails
        def get_node_depth(cls, node: Cursor, exhaustive: bool = True) -> int:
            depth = 0
            curr_node = node
            while True:
                curr_node = curr_node.lexical_parent  # return falsely None in some cases
                if curr_node is None:
                    break
                else:
                    depth += 1
            if exhaustive and depth == 0 and node.kind != CursorKind.TRANSLATION_UNIT:
                # node.lexical_parent is buggy
                # search manually starting from the nodes translation unit. THIS IS EXPENSIVE!
                assert node.translation_unit is not None, "Node without translation unit" + cls.get_node_as_string(node)
                for sub_node, sub_depth, sub_parent in cls.depth_search_tree(node.translation_unit.cursor):
                    if sub_node == node:
                        return sub_depth
                assert False, "Node not in its translation unit" + cls.get_node_as_string(node)
            else:
                return depth

        def get_asts_from_source_paths(self, source_file_paths):
            result = []
            for source_path in source_file_paths:
                result.append(self.get_ast_from_source(read_file_content(source_path), source_path))
            return result

        def get_ast_from_source(self, source_code, source_file_path=".cpp", omit_warnings=True,
                                preprocess_before_parsing=False):  # read_back_in=False,
            file_specific_includes = []

            if os.path.isfile(source_file_path):
                input_file_path = source_file_path
                extension = input_file_path.split(".")[-1]
                # Include directory of the given source file to the list of includes. This way sibling files can be included.
                file_specific_includes.append(os.path.dirname(source_file_path))
            else:
                # only extension allowed
                assert source_file_path[0] == "." and len(source_file_path.split(
                    ".")) == 2, source_file_path + " is no valid path and therefore must have format \".<ext>\""
                input_file_path = INPUT_FILE_NAME + source_file_path
                extension = source_file_path[1:]

            args = []  # ["--verbose"]
            # Add includes:
            args.extend(["-I" + include_path for include_path in self.include_paths + file_specific_includes])

            # Preprocess source before feeding it to the AST generation to make sure that macros are expanded before parsing
            #  otherwise the AST is inconsistent: Some nodes contain the macro token, other (sub-)nodes contain no
            #  token at all, and other contain the expanded token
            # Attention: This slows the parsing process down around 30/40 times
            # WARNING: THERE IS NO WAY TO DISTINGUISH BETWEEN included and non-included nodes later!? REALLY!?
            if (preprocess_before_parsing):
                result = subprocess.run(
                    ['clang'] + args + ['-E', "-x", ("c" if extension in ("c", "h") else "c++"), "-"],
                    # check=True, # throw exception on non zero exit code
                    stdout=subprocess.PIPE,  # capture stdout
                    stderr=subprocess.PIPE,  # capture stderr
                    input=source_code,
                    encoding="utf-8"  # encode stdout value
                )
                if (result.returncode != 0):
                    print("Error during preprocessing:", result.stderr)
                    return None

                source_code = result.stdout

                # source_code = get_ipython().getoutput("echo -e \"" + source_code.replace("\"", "\\\"") + "\" | clang -E -x " + ("c" if virtual_file_extension in ("c", "h") else "c++") + " -", split=False)
            #       read_back_in = True # True is a little bit faster
            #       input_temp_path = os.path.join(tempfile.gettempdir(), input_file_name)
            #       preprocessed_input_temp_path = os.path.join(tempfile.gettempdir(), "pre_" + input_file_name)
            #       # Read source to file
            #       write_file_content(input_temp_path, source_code)
            #       # Preprocess source file
            #       with pipes() as (out, err):
            #         get_ipython().system_raw("clang -E \"" + input_temp_path + "\"" + ("" if read_back_in else " > \"" + preprocessed_input_temp_path + "\""))
            #       if(read_back_in):
            #         # read preprocessed source back in
            #         source_code = out.read()

            # print(args)
            # options for parsing: https://github.com/llvm-mirror/clang/blob/master/bindings/python/clang/cindex.py#L2725
            options = TranslationUnit.PARSE_NONE  # TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD # add macro definitions to ast
            index = Index.create()

            tu = index.parse(path=input_file_path,
                             # if not preprocess_before_parsing or read_back_in else preprocessed_input_temp_path,
                             args=args, options=options,
                             unsaved_files=[(input_file_path,
                                             source_code)])  # if not preprocess_before_parsing or read_back_in else [])
            if not tu:
                print("Unable to load input")
                return None

            # print('finished parsing')
            error_occurred = False
            for diag in tu.diagnostics:
                if (diag.severity > 2):
                    error_occurred = True
                elif (omit_warnings):
                    continue
                print(diag)
                # print(cls.get_diag_info(diag))

            return None if error_occurred else tu.cursor

        @classmethod
        def get_literal_node_value(cls, node):
            if (node.kind in (CursorKind.INTEGER_LITERAL, CursorKind.FLOATING_LITERAL, CursorKind.IMAGINARY_LITERAL,
                              CursorKind.STRING_LITERAL,
                              CursorKind.CHARACTER_LITERAL, CursorKind.CXX_BOOL_LITERAL_EXPR,
                              CursorKind.CXX_NULL_PTR_LITERAL_EXPR)):
                # get literal:
                # "IntegerLiteral the first token will be your number" from https://stackoverflow.com/questions/25520945/how-to-retrieve-function-call-argument-values-using-libclang
                next_token = next(node.get_tokens(), None)
                if (next_token is None):
                    # print("Warning: No next token for literal. Returning None. Node Info:", node.kind, node.type.kind,
                    #       node.type.spelling, node.extent)
                    return None
                else:
                    return next_token.spelling
            elif (node.kind == CursorKind.GNU_NULL_EXPR):  # __null
                return "0"
            else:
                return None

        @classmethod
        def get_tokens(cls, node: Cursor) -> List[Token]:
            node_extent = node.extent
            return [tok for tok in node.get_tokens() if tok.location is not None and
                    cls.location_contained_in_source_range(tok.location, node_extent)]

        @classmethod
        def get_token_strings(cls, node: Cursor) -> List[str]:
            return [tok.spelling for tok in cls.get_tokens(node)]

        @classmethod
        def location_contained_in_source_range(cls, location: SourceLocation, range: SourceRange) -> bool:
            # Cache accesses for performance:
            loc_file_name = location.file.name
            range_start = range.start
            range_end = range.end
            loc_line = location.line
            loc_column = location.column
            range_start_line = range_start.line
            range_end_line = range_end.line
            # Check for start:
            start_ok = False
            if range_start_line <= loc_line:
                if range_start_line == loc_line:
                    if range_start.column <= loc_column:
                        start_ok = True
                else:
                    start_ok = True
            # Check for end:
            end_ok = False
            if range_end_line >= loc_line:
                if range_end_line == loc_line:
                    if range_end.column >= loc_column:
                        end_ok = True
                else:
                    end_ok = True
            if not start_ok or not end_ok:
                return False

            # Check file names last because they are usually equal:
            if (range_start.file.name != loc_file_name or
                    loc_file_name != range_end.file.name):
                return False

            return True

        @classmethod
        def source_ranges_contain(cls, range_inner: SourceRange, range_outer: SourceRange) -> bool:
            return cls.location_contained_in_source_range(range_inner.start, range_outer) \
                   and cls.location_contained_in_source_range(range_inner.end, range_outer)

        @classmethod
        def get_operator_str(cls, operator_node):
            if (operator_node.kind in (
                    CursorKind.BINARY_OPERATOR, CursorKind.COMPOUND_ASSIGNMENT_OPERATOR, CursorKind.UNARY_OPERATOR)):
                # Use own objects with an id to represent tokens because the clang token objects are not comparable!
                id = 0
                simple_oper_toks = []
                for tok in cls.get_tokens(operator_node):
                    simple_oper_toks.append({"id": id, "spelling": tok.spelling, "extent": tok.extent})
                    id += 1
                # Ids of tokens that were found in the operator's sub nodes
                sub_node_token_ids = []
                # print("\n".join([str((operator_token["spelling"], operator_token["extent"])) for operator_token in simple_oper_toks]))
                for sub_node, sub_depth, sub_parent in cls.depth_search_tree(operator_node):
                    if sub_depth == 1:  # only direct children
                        # Remove sub node token from operator_tokens:
                        # Use for loop and remove to only remove each token once (otherwise the operator itself will be removed too if it occurs multiple times)
                        # (e.g. "++(++i)")
                        for simple_oper_tok in simple_oper_toks:
                            if cls.source_ranges_contain(simple_oper_tok["extent"], sub_node.extent):
                                # This token is part of the sub token. Remember its id as it does not contain the
                                #  actual operator token
                                # print("Added", simple_oper_tok["id"])
                                sub_node_token_ids.append(simple_oper_tok["id"])
                # Remove all tokens that are part of the operator's sub node:
                simple_oper_toks = [simple_tok for simple_tok in simple_oper_toks if
                                    simple_tok["id"] not in sub_node_token_ids]

                # print("\n".join(
                #     [str((operator_token["spelling"], operator_token["extent"])) for operator_token in simple_oper_toks]))

                if len(simple_oper_toks) > 1:
                    # Probably not necessary anymore
                    print("WARNING: More than one token.")
                    allowed_operator_strings = [oper for oper in OPERATOR_STRINGS if oper != "dummy"]
                    # Remove tokens that are not a valid operator
                    simple_oper_toks = [token for token in simple_oper_toks if
                                        token["spelling"] in allowed_operator_strings]

                if len(simple_oper_toks) == 1:
                    # Everything okay:
                    return simple_oper_toks[0]["spelling"]
                else:
                    # More or less than one token left. This probably happens if a child is a UnexposedExpr and/or MACRO
                    # print("Warning: " + str(len(simple_oper_toks)) + " tokens found for operator. Tokens: " + str([token["spelling"] for token in simple_oper_toks]), "Sub AST:")
                    # for sub_node, sub_depth, sub_parent in cls.depth_search_tree(operator_node):
                    #     print("  ", sub_depth, sub_node.kind, cls.get_token_strings(sub_node), [(tok.kind, tok.spelling) for tok in cls.get_tokens(sub_node)], sub_node.extent)

                    # if len(operator_tokens) == 0:
                    return None
                # else:
                #
                #     print(cls.get_token_strings(operator_node))
                #     for sub_node, sub_depth, sub_parent in cls.depth_search_tree(operator_node):
                #         print("  ", sub_depth, sub_node.kind, cls.get_token_strings(sub_node), sub_node.extent)

                #     # TODO: Workaround by not using token strings but sourceranges (extent)!?
                #
                #     assert len(operator_tokens) <= 1, "Unexpected token count " + str(operator_tokens)
                #     if len(operator_tokens) == 0:
                #         return "<unknown>"
                #         # # Unable to recover from missing information. Display warning for now: TODO
                #         # node_token_spellings.append("<unknown>")
                #         # # Print warning and debug info:
                #         # print(
                #         #     "Warning: " + str(len(node_token_spellings) - 1) + " tokens found for operator. Tokens: " + str(
                #         #         cls.get_token_strings(operator_node)) + ". Returning " + node_token_spellings[0],
                #         #     "Sub AST:")
                #         # for sub_node, sub_depth, sub_parent in cls.depth_search_tree(operator_node):
                #         #     print("  ", sub_depth, sub_node.kind, cls.get_token_strings(sub_node), sub_node.extent)


            elif (operator_node.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR):
                return "[]"
            elif (operator_node.kind == CursorKind.CONDITIONAL_OPERATOR):
                return "?:"
            else:
                return None

# Enable functionality for parsing of Java source code. This was only used in an early stage of the tool and is not
#  supported anymore.
ENABLE_JAVA_PARSING = False
if ENABLE_JAVA_PARSING:
    LITERAL_JAVA_DATA_TYPES = ("byte", "short", "int", "long", "float", "double", "char", "String", "boolean")


    class JavaLang(ParserFrontend):

        def get_feat_vec_for_node(self, node):
            # Compute the nodes feature vector:
            # node kind:
            kind_feat_vec = AST_NODE_KINDS_TO_FEAT_VEC_MAPPING[type(node)]
            if (kind_feat_vec is None):
                return None
            # Special cases where the kind is not reflected in the type
            if isinstance(node, javalang.tree.BinaryOperation):
                kind_feat_vec = self.get_feat_fec_with_operator_feature(kind_feat_vec, node.operator)
            if isinstance(node, javalang.tree.Primary):
                if (node.prefix_operators is not None and len(node.prefix_operators)):
                    # TODO: These operators are not represented as separate node in the javalang-AST :-(
                    # Create nodes for these?
                    print("Found prefix operators: " + str(node.prefix_operators))
                elif (node.postfix_operators is not None and len(node.postfix_operators)):
                    # TODO: These operators are not represented as separate node in the javalang-AST :-(
                    # Create nodes for these?
                    print("Found postfix operators: " + str(node.postfix_operators))

            # node value:
            value_feat_vec = [0, 0]
            if isinstance(node, javalang.tree.Literal):
                value_feat_vec = self.get_value_feat_vec_from_literal_value(node.value)

            # Variable, member, or method name
            member_feat_vec = [0, 0]
            if isinstance(node, (javalang.tree.MemberReference, javalang.tree.MethodInvocation)):
                # Attributes: "member", "prefix_operators", "postfix_operators", "qualifier", "selectors"
                # Additional attributes for method invocation: "type_arguments", "arguments"
                member_kind_feat = FEAT_MEMBER_FILE_OR_PROJECT  # assume file first
                member = node.member
                if (node.qualifier is not None and len(node.qualifier) > 0):
                    member = node.qualifier + "." + member
                    # Compute full qualified name (e.g. for "System.out" compute "java.lang.System.out")
                    member = member  # Not easily possible with the current AST :-(
                    # TODO: In addition, all imports should be considered! Maybe another kind just for imported stuff!?
                    if (member.startswith("java") or
                            member.startswith("javax") or
                            member.startswith("sun")):
                        member_kind_feat = FEAT_MEMBER_SYSTEM_OR_INCLUDED
                    # Some special cases as workaround (TODO: Remove)
                    if (member.startswith("System.out")):
                        member_kind_feat = FEAT_MEMBER_SYSTEM_OR_INCLUDED
                member_feat_vec = [member_kind_feat, self.lookup_project_var_name(member)]

            # Type
            type_feat_vec = [0, 0]
            if isinstance(node, javalang.tree.Type):
                type_name = node.name
                if isinstance(node, javalang.tree.BasicType):
                    type_kind_feat = FEAT_TYPE_BASIC
                    type_feat = LITERAL_JAVA_DATA_TYPES.index(type_name)
                elif isinstance(node, javalang.tree.ReferenceType):
                    type_kind_feat = FEAT_TYPE_REFERENCE
                    type_feat = len(LITERAL_JAVA_DATA_TYPES) + 1 + self.lookup_ref_type_name(type_name)
                else:
                    raise Exception("Unknown Type subtype " + type(node).__name__)
                type_feat_vec = [type_kind_feat, type_feat]

            # total feat vec:
            return kind_feat_vec + tuple(value_feat_vec + member_feat_vec + type_feat_vec)

        @classmethod
        def get_node_as_string(cls, node):
            def get_attr_desc(description, attribute_name):
                return (" " + description + ": " + str(getattr(node, attribute_name))) if hasattr(node,
                                                                                                  attribute_name) else ""

            result = type(node).__name__ + ":"
            result += get_attr_desc("ID", "node_id")
            result += get_attr_desc("Name", "name")
            node_type = cls.get_type_of_node(node)
            if node_type is not None:
                result += " Type: " + type(node_type).__name__ + " " + node_type.name
                if hasattr(node_type.dimensions, '__iter__'):
                    result += "".join(["[]" for dim in node_type.dimensions])
            result += (" Method or Member: " + (node.qualifier + "." if node.qualifier is not None and len(
                node.qualifier) > 0 else "") + node.member if hasattr(node, "member") else "")
            result += get_attr_desc("Value", "value")
            result += get_attr_desc("Operator", "operator")
            # result += " Feature: " + str(get_feature_vector(node))
            result += get_attr_desc("Depth", "node_depth")
            result += get_attr_desc("Position", "position")
            return result

        @classmethod
        # Return the lexical depth of the given node
        def get_node_depth(cls, node):
            try:
                return node.node_depth
            except:
                return 0

        # Generator of all nodes in depth search pre-order
        # returns node, depth, node's parent
        @classmethod
        def depth_search_tree(cls, node, skip_included: typing.Union[bool, str] = False, verbose: bool = False,
                              depth: int = 0, parent=None):
            children = None
            new_parent = None
            if isinstance(node, javalang.ast.Node):
                # This is a node: print it...
                yield node, depth, parent

                # and continue with its children:
                children = node.children
                new_parent = node
            elif (isinstance(node, (
                    list, tuple))):  # but the following does not work!?: isinstance(node, (type(list), type(tuple)))
                # This is not a node (but a list or tuple (why?)) and nothing
                # can be printed. Therefore the depth must be decreased:
                depth -= 1
                # Continue with the list contents:
                children = node
                new_parent = parent
            else:
                # print("unknown type" + type(node).__name__)
                return

            for child in children:
                yield from cls.depth_search_tree(child, skip_included=skip_included, verbose=verbose, depth=depth + 1,
                                                 parent=new_parent)

        @classmethod
        def get_ast_from_source(cls, source_code, source_file_path=".java"):
            assert source_file_path == ".java", "Only .java file are supported"
            ast = javalang.parse.parse(source_code)
            cls.annotate_basic_info(ast)
            return ast

        def get_asts_from_source_paths(self, source_file_paths):
            result = []
            for source_path in source_file_paths:
                result.append(self.get_ast_from_source(read_file_content(source_path)))
            return result

        @classmethod
        def is_method_definition_node(cls, node):
            return isinstance(node, javalang.tree.MethodDeclaration)

        @classmethod
        def is_included_or_imported_node(cls, ast_node, project_path: str = None):
            # Currently the frontend should only return nodes that are not imported
            # TODO
            return False

        @classmethod
        def get_method_name(cls, method_node):
            return method_node.name

        # Remove nodes that corresponds to the return value and parameter declaration of the given MethodDeclaration node
        # e.g.
        # MethodDeclaration: ID: 49 Name: openFile Feature: (0, 1, 0, 0, 0.5, 0.5, 6, 2) Depth: 2 Position: Position(line=15, column=9)
        # |`->BasicType: ID: 50 Name: char Feature: (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5) Depth: 3 Position: None
        # |`->FormalParameter: ID: 51 Name: name Type: ReferenceType String Feature: (0, 1, 0, 0, 0.5, 0.5, 7, 2) Depth: 3 Position: Position(line=15, column=23)
        # |   |`->ReferenceType: ID: 52 Name: String Type: ReferenceType String Feature: (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5) Depth: 4 Position: None
        # |`->StatementExpression: ID: 55 Feature: (1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5) Depth: 3 Position: None [...]
        # |`->ReturnStatement: ID: 61 Feature: (1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5) Depth: 3 Position: Position(line=18, column=5) [...]
        # to list [
        #   StatementExpression: ID: 55 Feature: (1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5) Depth: 3 Position: None [...],
        #   ReturnStatement: ID: 61 Feature: (1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5) Depth: 3 Position: Position(line=18, column=5) [...]
        # ]
        @classmethod
        def get_relevant_nodes_from_method_def_node(cls, method_def_node):
            first = True
            result_nodes = []
            for sub_node, sub_depth, sub_parent in cls.depth_search_tree(method_def_node):
                # Only look at direct children:
                if (sub_depth != 1):
                    continue
                # Look for return value or parameter:
                if (first and isinstance(sub_node, (javalang.tree.Type, javalang.tree.FormalParameter))):
                    continue
                # Found normal node. Add it:
                first = False
                result_nodes.append(sub_node)
            return result_nodes

        @classmethod
        def add_called_methods_to_list(cls, method_node, list_to_add, all_methods_in_content):
            for sub_node, sub_depth, sub_parent in cls.depth_search_tree(method_node):
                if (sub_depth > 0 and isinstance(sub_node, javalang.tree.MethodInvocation)):
                    called_method_name = sub_node.member  # get the name of the called method
                    # check whether there is a method that has a matching name and is in content
                    called_method_node = next((method_node for method_node, method_name in all_methods_in_content if
                                               method_name == called_method_name), None)
                    if (called_method_node is None):
                        # print("Call to unknown method " + called_method_name) # e.g. writeLine
                        continue
                    elif (called_method_node not in list_to_add):
                        list_to_add.append(called_method_node)
                        cls.add_called_methods_to_list(called_method_node, list_to_add, all_methods_in_content)

        # Adds depth information to each node (because len(path) is wrong sometimes)
        @classmethod
        def annotate_basic_info(cls, tree):
            node_id = 1
            for node, depth, parent in cls.depth_search_tree(tree):
                node.node_depth = depth
                node.node_parent = parent
                node.node_kind = type(node).__name__
                node.node_id = node_id
                node_id += 1

        @classmethod
        def get_type_of_node(cls, node):
            # type attribute of Assignment is just a string (e.g. "=")
            if (hasattr(node, "type") and not isinstance(node, javalang.tree.Assignment)):
                return node.type
            #     elif(node.node_parent is not None):
            #       return cls.get_type_of_node(node.node_parent)
            else:
                return None

if __name__ == "__main__":
    project_path = os.path.join(ENVIRONMENT.data_root_path, "Juliet", "Juliet_Test_Suite_v1.3_for_C_Cpp")
    clangTooling = ClangTooling([os.path.join(project_path, "testcasesupport")], project_path)
    # print(clangTooling.get_static_kind_info())
    # print(clangTooling.UNIQUE_NUMBER_TO_NODE_KIND_MAPPING[148])

    content = """
        #include <stdlib.h>
        int main() {
            int arr[3], *arr2[4];
            int brr[3]; int *brr2[4];
            char* buffer = (char*) malloc (2+1);
        }
       """
    clangTooling.print_tree(clangTooling.get_ast_from_source(content, ".cpp", verbosity=10))
