# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    liyanpeng@tsingmicro.com
# Datetime: 2022/10/9 17:36
# Filename: tvm_case.py
import tvm
from tvm import relay
from tvm.relay import transform, testing

from utils import relay_ir2txt
from pass_args import desired_layouts, pattern_table


def cc_case(shape=(1, 16, 7, 7)):
    """CanonicalizeCast"""
    data = relay.var("data", shape=shape, dtype="int8")
    conv_weight = relay.var("weight")
    bias1 = relay.var("bias1", shape=(16, 1, 1), dtype="int32")
    bias2 = relay.var("bias2", shape=(16, 1, 1), dtype="int32")

    x = relay.nn.conv2d(
            data, conv_weight, channels=16, kernel_size=(3, 3), padding=(1, 1), out_dtype="int8"
        )
    x1 = relay.cast(x, dtype="int32")
    y1 = relay.add(x1, bias1)
    y2 = relay.add(x1, bias2)
    y = relay.add(y1, y2)
    return relay.Function([data, conv_weight, bias1, bias2], y)


def cl_case():
    """ConvertLayout"""
    # transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]})
    # 优化前后一样
    x = relay.var("x", shape=(1, 64, 56, 56))
    weight = relay.var("weight", shape=(64, 64, 3, 3))
    y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
    y = relay.nn.relu(y)
    y = relay.Function([x, weight], y)
    return y


def mc_case():
    """MergeComposite"""
    # transform.MergeComposite(pattern_table)
    data = relay.var("data", shape=(1, 512, 28, 28))
    kernel = relay.var("kernel", shape=(256, 512, 1, 1))
    bias = relay.var("bias", shape=(256,))
    a = relay.var("a", shape=(1, 256, 28, 28))
    b = relay.var("b", shape=(1, 256, 28, 28))

    conv_node = relay.nn.conv2d(
        data, kernel, kernel_size=(1, 1), padding=(0, 0), strides=(1, 1)
    )

    bias_node = relay.nn.bias_add(conv_node, bias)
    relu_node = relay.nn.relu(bias_node)
    add_node = relay.add(relu_node, a)
    relu_node_2 = relay.nn.relu(add_node)
    r = relay.multiply(relu_node_2, b)

    return relay.Function([data, kernel, bias, a, b], r)


if __name__ == '__main__':
    f = mc_case()
    mod = tvm.IRModule.from_expr(f)
    mod_func = mod["main"]
    relay_ir2txt(mod_func, file_name='tvm_case', is_ap=False)

    mod_opt = testing.run_opt_pass(mod_func, transform.MergeComposite(pattern_table))
    relay_ir2txt(mod_opt, file_name='tvm_case', is_ap=True)