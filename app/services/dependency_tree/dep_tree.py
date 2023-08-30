from anytree import NodeMixin


class DepNode(NodeMixin):
    def __init__(self, text, index, dep_label, parent=None, children=None):
        super(DepNode, self).__init__()

        self.text = text
        self.index = index
        self.dep_label = dep_label
        self.parent = parent
        if children:
            self.children = children
