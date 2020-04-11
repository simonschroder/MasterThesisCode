import os
import re
import tempfile
import unittest

from DataPreparation import Juliet
from Environment import ENVIRONMENT
from Frontends import ClangTooling
from tests.Utils import captured_output


class ClangToolingTestCase(unittest.TestCase):

    def setUp(self):
        self.clangTooling = ClangTooling()

    def test_source_content_parsing(self):
        content = """
               #define NULL (void*)0
               //#include <stddef.h>
               int main() {
                 int* data;
                 data = NULL;

               }
               """
        with captured_output() as (sysout, syserr):
            self.clangTooling.print_tree(self.clangTooling.get_ast_from_source(content, ".c"))

        # No error:
        self.assertEqual(syserr.getvalue(), "")
        # Expected output:
        output = sysout.getvalue()
        # Replace ast-wide hashes with fixed numbers. TODO: It would be more correct to map each hash to increasing
        #  number
        output = re.sub(r"hash':\s\d*?(?!\d)", "hash': 1234", output, flags=re.MULTILINE)
        # Replace temp dir path and random part in tmpdir:
        output = re.sub(tempfile.gettempdir() + "/tmp.*?\\.", "spelling': 'tmpXXX.", output, flags=re.MULTILINE)
        self.assertEqual(output, """{'kind': ['AstRoot', '', -1], 'depth': 0}
|`-> {'extent': None, 'id': {'hash': 1234, 'in_project': False, 'spelling': ('spelling': 'tmpXXX.c', '\\n               #define NULL (void*)0\\n               //#include <stddef.h>\\n               int main() {\\n                 int* data;\\n                 data = NULL;\\n\\n               }\\n               ', '.c')}, 'is_def': False, 'kind': ['Decl', 'TranslationUnit', 72], 'type': None, 'depth': 1}
|   |`-> {'extent': {'end': {'col': 16, 'line': 8}, 'start': {'col': 16, 'file': ':MN:', 'line': 4}}, 'id': {'hash': 1234, 'spelling': 'spelling': 'tmpXXX.c:main'}, 'is_def': True, 'kind': ['Decl', 'Function', 48], 'type': {'id': {'hash': 1234, 'spelling': 'int ()'}, 'is_const': False, 'is_pod': False, 'is_vola': False, 'kind': ['Type', 'FunctionNoProto', 17], 'result_type': {'id': {'hash': 1234, 'spelling': 'int'}, 'is_const': False, 'is_pod': True, 'is_vola': False, 'kind': ['Type', 'Builtin', 0]}}, 'depth': 2}
|   |   |`-> {'extent': {'end': {'col': 16, 'line': 8}, 'start': {'col': 27, 'file': ':MN:', 'line': 4}}, 'id': {'hash': 1234}, 'kind': ['Stmt', 'CompoundStmt', 9], 'type': None, 'depth': 3}
|   |   |   |`-> {'extent': {'end': {'col': 27, 'line': 5}, 'start': {'col': 18, 'file': ':MN:', 'line': 5}}, 'id': {'hash': 1234}, 'kind': ['Stmt', 'DeclStmt', 13], 'type': None, 'depth': 4}
|   |   |   |   |`-> {'extent': {'end': {'col': 23, 'line': 5}, 'start': {'col': 18, 'file': ':MN:', 'line': 5}}, 'id': {'hash': 1234, 'spelling': 'spelling': 'tmpXXX.c:main()::data'}, 'is_def': True, 'kind': ['Decl', 'Var', 56], 'type': {'id': {'hash': 1234, 'spelling': 'int *'}, 'is_const': False, 'is_pod': True, 'is_vola': False, 'kind': ['Type', 'Pointer', 2], 'pointee_type': {'id': {'hash': 1234, 'spelling': 'int'}, 'is_const': False, 'is_pod': True, 'is_vola': False, 'kind': ['Type', 'Builtin', 0]}}, 'depth': 5}
|   |   |   |`-> {'id': {'hash': 1234}, 'kind': ['EndOfStmt', 'DeclStmt', 13], 'depth': 4}
|   |   |   |`-> {'extent': {'end': {'col': 25, 'line': 6}, 'start': {'col': 18, 'file': ':MN:', 'line': 6}}, 'id': {'hash': 1234}, 'kind': ['Expr', 'BinaryOperator', 24], 'operator': '=', 'type': {'id': {'hash': 1234, 'spelling': 'int *'}, 'is_const': False, 'is_pod': True, 'is_vola': False, 'kind': ['Type', 'Pointer', 2], 'pointee_type': {'id': {'hash': 1234, 'spelling': 'int'}, 'is_const': False, 'is_pod': True, 'is_vola': False, 'kind': ['Type', 'Builtin', 0]}}, 'depth': 4}
|   |   |   |   |`-> {'extent': {'end': {'col': 18, 'line': 6}, 'start': {'col': 18, 'file': ':MN:', 'line': 6}}, 'id': {'hash': 1234}, 'kind': ['Expr', 'DeclRefExpr', 67], 'ref_id': {'hash': 1234, 'spelling': 'spelling': 'tmpXXX.c:main()::data'}, 'type': {'id': {'hash': 1234, 'spelling': 'int *'}, 'is_const': False, 'is_pod': True, 'is_vola': False, 'kind': ['Type', 'Pointer', 2], 'pointee_type': {'id': {'hash': 1234, 'spelling': 'int'}, 'is_const': False, 'is_pod': True, 'is_vola': False, 'kind': ['Type', 'Builtin', 0]}}, 'depth': 5}
|   |   |   |   |`-> {'extent': {'end': {'col': 25, 'line': 6}, 'start': {'col': 25, 'file': ':MN:', 'line': 6}}, 'id': {'hash': 1234}, 'kind': ['Expr', 'CStyleCastExpr', 53], 'type': {'id': {'hash': 1234, 'spelling': 'int *'}, 'is_const': False, 'is_pod': True, 'is_vola': False, 'kind': ['Type', 'Pointer', 2], 'pointee_type': {'id': {'hash': 1234, 'spelling': 'int'}, 'is_const': False, 'is_pod': True, 'is_vola': False, 'kind': ['Type', 'Builtin', 0]}}, 'depth': 5}
|   |   |   |   |   |`-> {'extent': {'end': {'col': 25, 'line': 6}, 'start': {'col': 25, 'file': ':MN:', 'line': 6}}, 'id': {'hash': 1234}, 'kind': ['Expr', 'IntegerLiteral', 83], 'type': {'id': {'hash': 1234, 'spelling': 'int'}, 'is_const': False, 'is_pod': True, 'is_vola': False, 'kind': ['Type', 'Builtin', 0]}, 'value': 0, 'depth': 6}
|   |   |   |   |`-> {'id': {'hash': 1234}, 'kind': ['EndOfExpr', 'CStyleCastExpr', 53], 'depth': 5}
|   |   |   |`-> {'id': {'hash': 1234}, 'kind': ['EndOfExpr', 'BinaryOperator', 24], 'depth': 4}
|   |   |`-> {'id': {'hash': 1234}, 'kind': ['EndOfStmt', 'CompoundStmt', 9], 'depth': 3}
|   |`-> {'id': {'hash': 1234, 'spelling': 'spelling': 'tmpXXX.c:main'}, 'kind': ['EndOfDecl', 'Function', 48], 'depth': 2}
|`-> {'id': {'hash': 1234, 'in_project': False, 'spelling': 'spelling': 'tmpXXX.c'}, 'kind': ['EndOfDecl', 'TranslationUnit', 72], 'depth': 1}
""")


class JulietTestCase(unittest.TestCase):
    def setUp(self):
        project_path = os.path.join(ENVIRONMENT.data_root_path, "Juliet", Juliet.ZIP_FILE_NAME)
        self.clangTooling = ClangTooling([os.path.join(project_path, "testcasesupport")], project_path)

    def test_environment(self):
        """
        Tests whether ClangAstToJson and header file includes are working
        :return:
        """
        self.assertIsNotNone(self.clangTooling.get_ast_from_source("#include \"std_testcase.h\"", ".cpp"))
        self.assertIsNotNone(self.clangTooling.get_ast_from_source_path(
            os.path.join(self.clangTooling.project_path, "testcases", "CWE197_Numeric_Truncation_Error", "s01",
                         "CWE197_Numeric_Truncation_Error__int_listen_socket_to_short_02.c")))

    def test_wine_environment(self):
        """
        Tests whether wine includes are working and whether "Winldap.h to winldap.h" symlink has been created
        :return:
        """
        self.assertIsNotNone(self.clangTooling.get_ast_from_source("#include <windows.h>\n"
                                                                   + "#include <Winldap.h>",
                                                                   ".cpp", context={"needs_windows": True}))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
