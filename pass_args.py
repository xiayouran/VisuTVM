# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    liyanpeng@tsingmicro.com
# Datetime: 2022/10/10 10:44
# Filename: pass_args.py
from tvm import relay
from tvm.relay.dataflow_pattern import TupleGetItemPattern, is_op, wildcard


def make_add_sub_mul_pattern():
    r"""Create a pattern to match the following graph.
    add  sub
     \   /
      \ /
      mul
    """
    x = wildcard()
    y = wildcard()
    return (x + y) * (x - y)


def make_add_relu_pattern():
    r"""Create a pattern to match the following graph.
     add
      |
    relu
    """
    add_node = wildcard() + wildcard()
    r = is_op("nn.relu")(add_node)
    return r


def make_conv_bias_relu_pattern():
    r"""Create a pattern to match the following graph.
     conv2d
       |
    bias_add
       |
     relu
    """
    x = wildcard()
    y = wildcard()
    z = wildcard()
    conv_node = is_op("nn.conv2d")(x, y)
    bias_node = is_op("nn.bias_add")(conv_node, z)
    r = is_op("nn.relu")(bias_node)
    return r


def make_pattern_with_optional():
    r"""Create a pattern to match the following graph. Note that relu is optinal.
     conv2d
       |
    bias_add
       |
     (relu)
    """
    x = wildcard()
    y = wildcard()
    z = wildcard()
    conv_node = is_op("nn.conv2d")(x, y)
    bias_node = is_op("nn.bias_add")(conv_node, z)
    r = bias_node.optional(lambda x: is_op("nn.relu")(x))
    return r


def make_add_add_add_pattern():
    r"""Create a pattern to match the following graph.
       Useful for testing re-using a call node.
        x    y
      /  \  /
      |  add
       \  |  \
         add |
          | /
         add
    """
    x = wildcard()
    y = wildcard()
    add_node = is_op("add")(x, y)
    add_node_1 = is_op("add")(x, add_node)
    r = is_op("add")(add_node_1, add_node)
    return r


def make_bn_relu_pattern():
    r"""Create a pattern to match the following graph.
     batch_norm
         |
    TupleGetItem(0)
         |
       relu
    """
    x = wildcard()
    gamma = wildcard()
    beta = wildcard()
    moving_mean = wildcard()
    moving_var = wildcard()
    bn_node = is_op("nn.batch_norm")(x, gamma, beta, moving_mean, moving_var)
    tuple_get_item_node = TupleGetItemPattern(bn_node, 0)
    r = is_op("nn.relu")(tuple_get_item_node)
    return r


def fskip(expr):
    if isinstance(expr, relay.expr.Call) and expr.op.name == "add":
        return True
    return False


pattern_table = [
    ("conv2d_bias_relu", make_conv_bias_relu_pattern()),
    ("add_relu", make_add_relu_pattern()),
]

desired_layouts = {
    "nn.conv2d": ["NCHW", "default"]
    }