# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    liyanpeng@tsingmicro.com
# Datetime: 2022/10/10 10:44
# Filename: pass_args.py
from tvm.relay.dataflow_pattern import is_op, wildcard


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


pattern_table = [
    ("conv2d_bias_relu", make_conv_bias_relu_pattern()),
    ("add_relu", make_add_relu_pattern()),
]

desired_layouts = {
    "nn.conv2d": ["NCHW", "default"]
    }