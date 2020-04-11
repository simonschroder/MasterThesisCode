import os
from typing import List, Union, Dict, Any, Tuple

from bs4 import BeautifulSoup

import DataPreparation
from Environment import ENVIRONMENT


def create_new_node(identifier: str, parent_node=None):
    """
    Creates a CWE taxonomy tree node
    :param identifier:
    :param parent_node:
    :return:
    """
    node = {"id": identifier, "children": [], "parent": parent_node}
    return node


def add_childs(node, relations: List[Tuple[str, str]], nodes: Dict[str, object] = None, depth=0):
    """
    Recursively builds a CWE taxonomy tree with root "node" from parent-child-relations in "relations"
    :param node: root node object whose children list will be recursively extended
    :param relations: list of pairs where the pair's elements are the parent cwe and child cwe of a relation
    :param nodes: Dict of all already created nodes
    :param depth:
    :return:
    """
    if nodes is None:
        nodes = {}
    node_id = node["id"]
    # Look for a relation whose parent is the current node
    for relation in relations:
        if relation[0] == node_id:
            # Get or create the relation's child node
            child_node = nodes.get(relation[1], None)
            if child_node is None:
                child_node = create_new_node(relation[1], node)
                nodes[relation[1]] = child_node
            # ... and add it to node's children:
            if child_node not in node["children"]:
                node["children"].append(child_node)
    max_depth = depth
    node_count = 1
    for child in node["children"]:
        # Recursively add child nodes:
        child_max_depth, child_node_count = add_childs(child, relations, nodes, depth + 1)
        max_depth = max(max_depth, child_max_depth)
        node_count += child_node_count
    return max_depth, node_count


def depth_search_tree(node, depth: int = 0, parent=None):
    """
    Yields all taxonomy tree nodes
    :param node:
    :param depth:
    :param parent:
    :return:
    """
    if node is None:
        return

    yield node, depth, parent

    for child in node["children"]:
        yield from depth_search_tree(child, depth=depth + 1, parent=node)


def print_cwe(cwe, depth=0):
    print(cwe_to_string(cwe, depth))


def cwe_to_string(cwe, depth=0):
    weakness_node = get_cwe_db().get(cwe, None)
    return "  " * depth + cwe + " " + (weakness_node["Name"] if weakness_node is not None else "<NA>")


def find_node(root_node, node_id_to_find):
    """
    Returns the depth-first-wise first occurrence of a node with the given id
    :param root_node:
    :param node_id_to_find:
    :return:
    """
    for sub_node, depth, parent in depth_search_tree(root_node):
        # print(sub_node["id"], node_id_to_find)
        if sub_node["id"] == node_id_to_find:
            return sub_node
    return None


def get_cwe_taxonomy_xml_ast(verbose=False):
    """
    Reads the CWE taxonomy xml file and return its tree structure.
    :param verbose:
    :return:
    """
    if verbose:
        print("Parsing CWE info xml ...")
    with open(os.path.join(ENVIRONMENT.code_root_path, "auxiliary", "CweCatalog.xml")) as taxonomy_xml_file:
        return BeautifulSoup(taxonomy_xml_file, "xml")


def get_cwe_taxonomy_trees(verbose=False):
    """
    Returns a list of the cwe taxonomy tree roots
    :param verbose:
    :return:
    """
    if verbose:
        print("Building CWE taxonomy trees from xml ...\n- Looking for relations and roots ...")
    relation_nodes = get_cwe_taxonomy_xml_ast(False).find_all("Related_Weakness", Nature="ChildOf")
    # Sort for deterministic behavior:
    relations = sorted(
        [(relation_node["CWE_ID"], relation_node.parent.parent["ID"]) for relation_node in relation_nodes])
    childs = sorted(list(set([relation[1] for relation in relations])))
    roots = sorted(list(set([relation[0] for relation in relations if relation[0] not in childs])))
    if verbose:
        print("  - Found", len(relation_nodes), "relations between CWEs.")
        print("  - Found", len(roots), "roots.")

    root_nodes = []
    for root in roots:
        if verbose:
            print("- Root cwe", root, "...")
        root_node = create_new_node(root)
        max_depth, node_count = add_childs(root_node, relations)
        root_nodes.append(root_node)
        if verbose:
            print("  - Max. depth:", max_depth, "CWE count", node_count)
        # print("root_node", root, root_node)
    return root_nodes


def get_cwe_db(verbose=False):
    """
    Builds a cached lookup of cwe information and returns it.
    :param verbose:
    :return:
    """
    if not hasattr(get_cwe_db, "cwe_db_cache"):
        if verbose:
            print("Building CWE info lookup ...")
        weakness_nodes = get_cwe_taxonomy_xml_ast().find_all("Weakness")
        cwe_db_cache = {}
        for weakness_node in weakness_nodes:
            cwe_db_cache[weakness_node["ID"]] = weakness_node

        if verbose:
            print("Found", len(cwe_db_cache), "cwes.")
        setattr(get_cwe_db, "cwe_db_cache", cwe_db_cache)
    return getattr(get_cwe_db, "cwe_db_cache")


def get_cwe_name(cwe: str):
    """
    Returns the name of the given cwe
    :param cwe:
    :return:
    """
    cwe_db = get_cwe_db()
    cwe_weakness_node = cwe_db.get(cwe, None)
    if cwe_weakness_node is None:
        return None
    return cwe_weakness_node["Name"]


def get_juliet_cwes():
    """
    Returns a dictionary of all Juliet CWE ids and names
    :return:
    """
    return {cwe_name.split("_")[0][3:]: cwe_name for cwe_name in DataPreparation.ALL_CWE_CLASSES}


def get_cwe_groups(cwes_to_group: Union[List[str], Dict[str, Any]], verbose: bool = False):
    """
    Group the given cwes depending on the CWE taxonomy. In the resulting dict, there is a key for each CWE taxonomy tree
    root. Each key's value is a subset of that given cwes which are located in the CWE taxonomy root specified by the
    key. A given cwe may be part of multiple CWE groups.
    :param cwes_to_group:
    :param verbose:
    :return:
    """
    # 247 is deprecated in favour of 350
    # 534 is deprecated in favour of 532
    # 398 is CWE *category* "7PK - Code Quality" and not a CWE. 710 "Improper Adherence to Coding Standards" is best
    # replacement afaik.
    aliases = {"247": "350",
               "534": "532",
               "398": "710"}
    # Build CWE taxonomy trees:
    root_nodes = get_cwe_taxonomy_trees(verbose)
    if verbose:
        print("\nGrouping", len(cwes_to_group), "cwes", cwes_to_group, "...")
    cwe_groups = {}
    not_found = []
    for cwe in cwes_to_group:
        found = False
        # A cwe can appear as child of multiple root_nodes!
        for root_node in root_nodes:
            # Use alias -- if necessary -- when searching for parent in taxonomy tree:
            found_node = find_node(root_node, aliases.get(cwe, cwe))
            if found_node is not None:
                # Found cwe in current taxonomy tree. Determine the trees root:
                found_parent = found_node
                while True:
                    found_parent_or_none = found_parent["parent"]
                    if found_parent_or_none is None:
                        break
                    else:
                        found_parent = found_parent_or_none
                cwe_groups.setdefault(found_parent["id"], []).append(cwe)
                found = True
                # Do not break here, as there may be more occurrences of cwe in other root_nodes
        if not found:
            not_found.append(cwe)
            if verbose:
                print("<not found>")
    if verbose:
        print("Created", len(cwe_groups), "CWE groups:")
        cwe_count = 0
        for group_top_level_parent_cwe, group_member_cwe_list in cwe_groups.items():
            print("- Group of root cwe", cwe_to_string(group_top_level_parent_cwe), "has", len(group_member_cwe_list),
                  "cwes:")
            for group_member_cwe in group_member_cwe_list:
                print("  `->", cwe_to_string(group_member_cwe), cwes_to_group[group_member_cwe])
                cwe_count += 1
        print(cwe_count, "CWEs in groups.")
    if len(not_found) > 0:
        print("Warning: Unable to group the following cwe as they were not found in taxonomy:", not_found)
    return cwe_groups


if __name__ == '__main__':
    # Test:
    get_cwe_groups(get_juliet_cwes(), verbose=True)
