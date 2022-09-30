# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    youran.xia@foxmail.com
# Datetime: 2022/9/23 16:20
# Filename: custom_net.py
"""
可用于 EliminateCommonSubexpr、FoldConstant 可视化
"""
import tvm
from tvm import relay
from tvm.relay import transform, testing

import numpy as np

from utils import relay_ir2txt
from pass_map import PASS_MAP
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--passname', type=str, required=True, help="use Pass's name")
args = parser.parse_args()

pass_dict = PASS_MAP.get(args.passname, {})
assert pass_dict, "not support {} pass to visu now!".format(args.passname)

file_name = 'custom_net_' + pass_dict['name']
pass_func = pass_dict['pass']


def example_net(dshape):
    data = np.empty(dshape).astype("float32")
    input_data = relay.const(data)

    x = relay.var("x", shape=dshape)

    conv = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(1, 1), padding=(0, 0), channels=16)

    y = relay.add(input_data, input_data)
    y = relay.multiply(y, relay.const(2.0, "float32"))

    y = relay.add(conv, y)
    z = relay.add(y, input_data)
    z1 = relay.add(y, input_data)
    z2 = relay.add(z, z1)

    # [x]
    return relay.Function(relay.analysis.free_vars(z2), z2)


if __name__ == '__main__':
    f = example_net(dshape=(1, 16, 64, 64))
    mod = tvm.IRModule.from_expr(f)
    mod_func = mod["main"]

    relay_ir2txt(mod_func, file_name, is_ap=False)

    mod_opt = testing.run_opt_pass(mod_func, pass_func)
    relay_ir2txt(mod_opt, file_name, is_ap=True)
