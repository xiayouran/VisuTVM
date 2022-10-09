# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    liyanpeng@tsingmicro.com
# Datetime: 2022/10/9 17:36
# Filename: tvm_case.py
import tvm
from tvm import relay
from tvm.relay import transform, testing

import numpy as np

from utils import relay_ir2txt


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
