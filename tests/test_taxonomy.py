import unittest

from CweTaxonomy import get_cwe_taxonomy_trees
from tests.Utils import captured_output


class TaxonomyTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_source_content_parsing(self):
        with captured_output() as (sysout, syserr):
            get_cwe_taxonomy_trees(verbose=True)

        # No error:
        self.assertEqual(syserr.getvalue(), "")
        # Expected output:
        output = sysout.getvalue()

        self.assertEqual(output, """Building CWE taxonomy trees from xml ...
- Looking for relations and roots ...
  - Found 1608 relations between CWEs.
  - Found 11 roots.
- Root cwe 118 ...
  - Max. depth: 3 CWE count 30
- Root cwe 330 ...
  - Max. depth: 3 CWE count 23
- Root cwe 435 ...
  - Max. depth: 3 CWE count 15
- Root cwe 664 ...
  - Max. depth: 6 CWE count 472
- Root cwe 682 ...
  - Max. depth: 1 CWE count 12
- Root cwe 691 ...
  - Max. depth: 4 CWE count 81
- Root cwe 693 ...
  - Max. depth: 7 CWE count 349
- Root cwe 697 ...
  - Max. depth: 3 CWE count 20
- Root cwe 703 ...
  - Max. depth: 4 CWE count 52
- Root cwe 707 ...
  - Max. depth: 7 CWE count 270
- Root cwe 710 ...
  - Max. depth: 4 CWE count 197
""")


if __name__ == '__main__':
    unittest.main()
