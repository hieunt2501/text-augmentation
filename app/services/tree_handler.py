import random

from loguru import logger

from app.services.dependency_tree.dep_tree import DepNode
from app.services.dependency_tree.base_service import BaseService
from app.services.utils import tokenize, revert_segmented_tokens
from app.services.word_segment.word_segment import TextProcessor
from app.core.config import PHO_NLP_URL


fmt = "{time} - {name} - {level} - {message}"
logger.add("logs/tree.log", level="DEBUG", format=fmt, backtrace=True)
text_processor = TextProcessor()


class TreeHandler(BaseService):
    def __init__(self):
        super(TreeHandler, self).__init__()
        self.session = None

    def annotate(self, text):
        print(text)
        def annotate_request():
            return self.session.post(PHO_NLP_URL, json={"text": text}).json()

        result = self.make_request(annotate_request, key=text)
        return result

    @staticmethod
    def create_tree(annotations):
        node_dict = {}
        root_index = None

        for annotation in annotations:
            node = DepNode(text=annotation["form"],
                           index=annotation["index"],
                           dep_label=annotation["depLabel"])
            node_dict[annotation["index"]] = node

        for annotation in annotations:
            index = annotation["index"]
            head = annotation["head"]

            if head == 0:
                root_index = index
                continue

            node_dict[index].parent = node_dict[head]

        return node_dict, root_index

    @staticmethod
    def random_drop_phrase(node_dict, root_index):
        root_children = list(node_dict[root_index].children)
        root_avail_children = [child for child in root_children if child.children]

        if not root_avail_children:
            return node_dict, False

        chosen_child = random.choice(root_avail_children)
        while not chosen_child.children:
            chosen_child = random.choice(root_children)
        root_children.remove(chosen_child)
        node_dict[root_index].children = tuple(root_children)

        remove_node = [chosen_child]
        indices = [chosen_child.index]

        while remove_node:
            for node in remove_node:
                children = node.children
                for child in children:
                    indices.append(child.index)
                remove_node.remove(node)
                remove_node.extend(children)

        for idx in indices:
            node_dict.pop(idx, None)

        return node_dict, True

    def augment(self, text, exclude, is_segmented, segment, **kwargs):
        transform_text = text
        
        if exclude:
            return [text]
        
        if is_segmented:
            tokens = tokenize(text)
            transform_text = " ".join(revert_segmented_tokens(tokens))
            
        try:
            annotations = self.annotate(transform_text)

            node_dict, root_index = self.create_tree(annotations)
            node_dict, change = self.random_drop_phrase(node_dict, root_index)

            if not change:
                return [text]

            transform_text = [" ".join([value.text.replace("_", " ") if len(
                value.text) > 1 else value.text for value in node_dict.values()])]

            if segment:
                transform_text = [text_processor.process(t) for t in transform_text]
        except Exception as e:
            logger.info(text)
            logger.error(f"Exception {e}", exc_info=True)
            raise e
            
        return transform_text
