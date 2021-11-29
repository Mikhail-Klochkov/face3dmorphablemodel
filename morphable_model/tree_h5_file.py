import h5py
from anytree import Node, RenderTree


class TreeH5file():


    def __init__(self, h5file):
        self._h5file = h5file
        self._root = Node('File')


    def build_tree_struct(self):
        with h5py.File(self._h5file, 'r') as reader:
            try:
                self.add_nodes(reader, self._root)
            except Exception as e:
                assert False, f'Error: {e}'

    @property
    def tree_structure_data(self):
        return self._root

    def show_tree_structure(self):
        for pre, fill, node in RenderTree(self._root):
            print("%s%s" % (pre, node.name))

    @staticmethod
    def add_nodes(h5obj, parent):
        for obj in h5obj:
            if isinstance(h5obj[obj], h5py._hl.group.Group):
                name_curr_node = TreeH5file.get_name_node_group(key=obj, type_node='group')
                new_node = Node(name_curr_node, parent)
                TreeH5file.add_nodes(h5obj[obj], new_node)
            elif isinstance(h5obj[obj], h5py._hl.dataset.Dataset):
                name_curr_node = TreeH5file.get_name_node_group(key=obj, type_node='dataset')
                Node(name_curr_node, parent)
            else:
                assert False, 'Undefined Node!'

    @staticmethod
    def get_name_node_group(key, type_node = 'group', suffix=False):
        name = key
        if suffix:
            name += TreeH5file.get_suffix(type_node)
        return name

    @staticmethod
    def get_suffix(type='group'):
        if type == 'group':
            return '_sgroup'
        elif type == 'dataset':
            return '_dataset'