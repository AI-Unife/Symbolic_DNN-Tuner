import tensorflow as tf

"""
methods used to build a dict containing flops values for each operation in the neural network
"""

def get_next_node(node: tf.compat.v1.profiler.MultiGraphNodeProto):
    """
    method used to obtain the first child of a certain node in the graph
    :param node: current node from which to extrapolate the child
    :return: node first child
    """
    return node.children[0]


def get_name(node):
    """
    method used to obtain the name of a node in the graph
    :param node: current node from which you want to extract the name
    :return: node's name
    """
    return node.name


def get_flops(node):
    """
    method used to obtain the flops of a node in the graph
    :param node: current node from which you want to extract the flops
    :return: node flops
    """
    return node.float_ops


def get_next(node):
    """
    method used to esplore the graph, extracting the name and number of flops of the next node
    :param node: node from which to obtain informations of the next
    :return: name and flops of the next node
    """
    node = get_next_node(node)
    return get_name(node), get_flops(node)


def to_dict(node):
    """
    method used for recursive dict construction containing flops information
    :param node: current node with which build the dictionary
    return: dict containing flops values for each operation in the neural network
    """
    # if there's at least one child node
    if len(node.children) > 0:
        # obtain the next node and recursively call the method
        node = get_next_node(node)
        dict = to_dict(node)
        # use the current node name as the key and the number of flops as the value
        dict[get_name(node)] = get_flops(node)
        return dict
    else:
        return {}
