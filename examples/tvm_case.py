# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    liyanpeng@tsingmicro.com
# Datetime: 2022/10/9 17:36
# Filename: tvm_case.py
import tvm
from tvm import relay
from tvm.relay import transform, testing

import numpy as np

from utils import relay_ir2txt, _get_positive_scale
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


def mc_case1():
    """MergeComposite"""
    # add_relu
    # pattern_table: [("add_relu", make_add_relu_pattern())]
    # transform.MergeComposite(pattern_table)
    a = relay.var("a", shape=(10, 10))
    b = relay.var("b", shape=(10, 10))
    c = relay.const(0.1)
    add_node = relay.add(a, b)
    r = relay.nn.relu(add_node)
    r = relay.add(r, c)

    return relay.Function([a, b], r)


def mc_case2():
    # add_sub_mul
    # pattern_table: [("add_sub_mul", make_add_sub_mul_pattern())]
    a = relay.var("a", shape=(10, 10))
    b = relay.var("b", shape=(10, 10))
    c = relay.var("c", shape=(10, 10))
    add_node = relay.add(a, b)
    sub_node = relay.subtract(a, b)
    mul_node = relay.multiply(add_node, sub_node)
    add_node_2 = relay.add(c, mul_node)
    sub_node_2 = relay.subtract(c, mul_node)
    mul_node_2 = relay.multiply(add_node_2, sub_node_2)
    r = relay.nn.relu(mul_node_2)

    return relay.Function([a, b, c], r)


def mc_case3():
    # add_add_add
    # pattern_table: [("add_add_add", make_add_add_add_pattern())]
    a = relay.var("a", shape=(10, 10))
    b = relay.var("b", shape=(10, 10))
    sub_node = relay.subtract(a, b)

    # pattern
    add_node = relay.add(sub_node, b)
    add_node_1 = relay.add(sub_node, add_node)
    r = relay.add(add_node_1, add_node)

    return relay.Function([a, b], r)


def mc_case4():
    # conv_bias_relu + add_relu
    # pattern_table: [("conv2d_bias_relu", make_conv_bias_relu_pattern()), ("add_relu", make_add_relu_pattern()),]
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


def mc_case5():
    # bn_relu
    # pattern_table: [("bn_relu", make_bn_relu_pattern())]
    x = relay.var("x", shape=(1, 8))
    gamma = relay.var("gamma", shape=(8,))
    beta = relay.var("beta", shape=(8,))
    moving_mean = relay.var("moving_mean", shape=(8,))
    moving_var = relay.var("moving_var", shape=(8,))
    c = relay.const(0.1)
    bn_node = relay.nn.batch_norm(x, gamma, beta, moving_mean, moving_var)
    tuple_get_item_node = bn_node[0]
    r = relay.nn.relu(tuple_get_item_node)
    r = relay.add(r, c)

    return relay.Function([x, gamma, beta, moving_mean, moving_var], r)


def fo_case1():
    """FuseOps"""
    # transform.FuseOps()
    x = relay.var("x", shape=(10, 20))
    y = relay.add(x, relay.const(1, "float32"))
    z = relay.exp(y)
    w = relay.squeeze(z)

    return relay.Function([x], w)


def fo_case2(shape=(1, 16, 64, 64)):
    """FuseOps"""
    # transform.FuseOps(fuse_opt_level=2)
    x = relay.var("x", shape=shape)
    x = relay.add(x, relay.const(1, "float32"))
    y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=16)
    # this is the next dominator.
    y1 = relay.add(relay.const(1, "float32"), y)
    y = relay.add(y, y1)
    # second path
    z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
    z3 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(3, 3), padding=(1, 1), channels=16)
    # add can only be fused to z1
    z = relay.add(z2, z3)

    return relay.Function(relay.analysis.free_vars(z), z)


def fo_case3(shape=(1, 16, 64, 64)):
    x = relay.var("x", shape=shape)
    pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
    upsampled = relay.nn.upsampling(pooled, scale_h=2, scale_w=2, layout="NCHW")
    concat = relay.concatenate((upsampled, x), axis=1)
    out = relay.add(concat, relay.const(1, "float32"))

    return relay.Function(relay.analysis.free_vars(out), out)


def ecs_case():
    """EliminateCommonSubexpr"""
    # transform.EliminateCommonSubexpr(fskip)
    x = relay.var("x", shape=(1, 16))
    y1 = relay.nn.relu(x)
    y2 = relay.nn.relu(x)
    y1 = relay.add(y1, relay.const(1.0, "float32"))
    y2 = relay.add(y2, relay.const(1.0, "float32"))
    y = relay.add(y1, y2)
    f = relay.Function([x], y)

    return f


def fc_case():
    c_data = np.array([1, 2, 3]).astype("float32")
    t = relay.TensorType([1, 2, 3], "float32")

    c = relay.const(c_data)
    x = relay.var("x", t)
    y = relay.add(c, c)
    y = relay.multiply(y, relay.const(2, "float32"))
    y = relay.add(x, y)
    z = relay.add(y, c)

    return relay.Function([x], z)


def si_case(dim=4, axis=1, nstep=1, dtype="float32"):
    # transform.SimplifyInference()
    eps = 0.01
    ttype1 = relay.TensorType(tuple(10 for i in range(dim)), dtype)
    ttype2 = relay.TensorType((10,), dtype)
    x = relay.var("x", ttype1)
    beta = relay.var("beta", ttype2)
    gamma = relay.var("gamma", ttype2)
    moving_var = relay.var("moving_var", ttype2)
    moving_mean = relay.var("moving_mean", ttype2)
    y = x

    for _ in range(nstep):
        y, _, _ = relay.nn.batch_norm(
            y + relay.const(1, dtype),
            gamma,
            beta,
            moving_mean,
            moving_var,
            epsilon=eps,
            axis=axis,
        )
        # y = relay.nn.dropout(y)

    return relay.Function([x, beta, gamma, moving_var, moving_mean], y)


def cpc2d_case1(x_shape=(1, 4, 16, 16), channels1=4, channels2=8, channels3=4, channels4=7):
    """CombineParallelConv2D"""
    # transform.CombineParallelConv2D(min_num_branches=2)
    x = relay.var("x", shape=x_shape)
    in_c = x_shape[1]
    w1 = relay.var("w1", shape=(channels1, in_c, 1, 1))
    w2 = relay.var("w2", shape=(channels2, in_c, 1, 1))
    w3 = relay.var("w3", shape=(channels3, in_c, 3, 3))
    w4 = relay.var("w4", shape=(channels4, in_c, 1, 1))

    args = [x, w1, w2, w3, w4]
    y1 = relay.nn.conv2d(x, w1)
    y2 = relay.nn.conv2d(x, w2)
    # y3 cannot be combined
    y3 = relay.nn.conv2d(x, w3)
    y4 = relay.nn.conv2d(x, w4)
    y5 = relay.nn.max_pool2d(x)
    y = relay.Tuple((y1, y2, y3, y4, y5))

    return relay.Function(args, y)


def cpc2d_case2(x_shape=(1, 4, 16, 16), channels1=4, channels2=8):
    # combining conv2d + scale + relu
    # transform.CombineParallelConv2D(min_num_branches=2)
    x = relay.var("x", shape=x_shape)
    in_c = x_shape[1]
    w1 = relay.var("w1", shape=(channels1, in_c, 1, 1))
    w2 = relay.var("w2", shape=(channels2, in_c, 1, 1))
    scale1 = relay.var("scale1", shape=(channels1, 1, 1))
    scale2 = relay.var("scale2", shape=(channels2, 1, 1))
    bias = relay.var("bias", shape=(channels2, 1, 1))

    args = [x, w1, w2, scale1, scale2, bias]
    y1 = relay.nn.conv2d(x, w1)
    y1 = relay.multiply(y1, scale1)
    y1 = relay.nn.relu(y1)
    y2 = relay.nn.conv2d(x, w2)
    y2 = relay.multiply(y2, scale2)
    y2 = relay.nn.relu(y2)
    y2 = relay.add(y2, bias)
    y = relay.Tuple((y1, y2))

    return relay.Function(args, y)


def cpc2d_case3(x_shape=(1, 4, 16, 16), channels1=4, channels2=8):
    # un-combinable scale
    x = relay.var("x", shape=x_shape)
    in_c = x_shape[1]
    w1 = relay.var("w1", shape=(channels1, in_c, 1, 1))
    w2 = relay.var("w2", shape=(channels2, in_c, 1, 1))
    scale1 = relay.var("scale1", shape=(1,))
    scale2 = relay.var("scale2", shape=(1,))

    args = [x, w1, w2, scale1, scale2]
    y1 = relay.nn.conv2d(x, w1)
    y1 = relay.multiply(y1, scale1)
    y2 = relay.nn.conv2d(x, w2)
    y2 = relay.multiply(y2, scale2)
    y = relay.Tuple((y1, y2))

    return relay.Function(args, y)


def cpc2d_case4(x_shape=(1, 4, 16, 16), repeat=4):
    x = relay.var("x", shape=x_shape)
    in_c = x_shape[1]
    out_c = in_c // 2
    w = relay.var("w", shape=(out_c, in_c, 1, 1))

    args = [x, w]
    y = x
    for i in range(repeat):
        y1 = relay.nn.conv2d(y, w)
        y2 = relay.nn.conv2d(y, w)
        y = relay.concatenate((y1, y2), axis=1)
    return relay.Function(args, y)


def cpd_case1(i=100, j=200, k=300):
    """CombineParallelDense"""
    # transform.CombineParallelDense(min_num_branches=2)
    x = relay.var("x", shape=(i, k))
    w1 = relay.var("w1", shape=(j, k))
    w2 = relay.var("w2", shape=(j, k))
    w3 = relay.var("w3", shape=(j + 1, k))
    w4 = relay.var("w4", shape=(j, k))

    args = [x, w1, w2, w3, w4]
    y1 = relay.nn.dense(x, w1)
    y2 = relay.nn.dense(x, w2)

    # y3 cannot be combined due to shape mismatch
    y3 = relay.nn.dense(x, w3)

    y4 = relay.nn.dense(x, w4)
    y = relay.Tuple((y1, y2, y3, y4))

    return relay.Function(args, y)


def cpd_case2(i=100, j=200, k=300, scale1=0.5, scale2=0.25, newshape=(1, 1, 20000)):
    # combining dense + 1d biasadd + multiply with non-fused reshape
    x = relay.var("x", shape=(i, k))
    w1 = relay.var("w1", shape=(j, k))
    w2 = relay.var("w2", shape=(j, k))
    b1 = relay.var("b1", shape=(j,))
    b2 = relay.var("b2", shape=(j,))
    scale1 = relay.var("scale1", shape=(1,))
    scale2 = relay.var("scale2", shape=(1,))

    args = [x, w1, w2, b1, b2, scale1, scale2]
    y1 = relay.nn.dense(x, w1)
    y2 = relay.nn.dense(x, w2)
    y1 = relay.add(y1, b1)
    y2 = relay.add(y2, b2)
    y1 = relay.multiply(y1, scale1)
    y2 = relay.multiply(y2, scale2)
    y1 = relay.reshape(y1, newshape=newshape)
    y2 = relay.reshape(y2, newshape=newshape)
    y = relay.Tuple((y1, y2))

    return relay.Function(args, y)


def cpd_case3(i=100, j=200, k=300):
    # All matmul of different output dim can be combined
    # transform.CombineParallelDense(min_num_branches=3, to_batch=False)
    x = relay.var("x", shape=(i, k))
    w1 = relay.var("w1", shape=(j, k))
    w2 = relay.var("w2", shape=(2 * j, k))
    w3 = relay.var("w3", shape=(3 * j, k))

    args = [x, w1, w2, w3]
    y1 = relay.nn.dense(x, w1)
    y2 = relay.nn.dense(x, w2)
    y3 = relay.nn.dense(x, w3)
    y = relay.Tuple((y1, y2, y3))

    return relay.Function(args, y)


def cpd_case4(i=3, j=5, k=4, bias_shape1=(5,), bias_shape2=(1,)):
    # combining dense + 1d biasadd with different out dims
    # transform.CombineParallelDense(min_num_branches=2, to_batch=False)
    x = relay.var("x", shape=(i, k))
    w1 = relay.var("w1", shape=(j, k))
    w2 = relay.var("w2", shape=(2 * j, k))
    b1 = relay.var("b1", shape=bias_shape1)
    b2 = relay.var("b2", shape=bias_shape2)

    args = [x, w1, w2, b1, b2]
    y1 = relay.nn.dense(x, w1)
    y2 = relay.nn.dense(x, w2)
    y1 = relay.add(y1, b1)
    y2 = relay.add(y2, b2)
    y = relay.Tuple((y1, y2))

    return relay.Function(args, y)


def cpd_case5(i=100, j=200, k=300, scale1=0.5, scale2=0.25, newshape1=(1, 1, 20000), newshape2=(1, 1, 40000)):
    # combining dense with different out dims following bias add, scale, reshape ops
    # transform.CombineParallelDense(min_num_branches=2, to_batch=False)
    x = relay.var("x", shape=(i, k))
    w1 = relay.var("w1", shape=(j, k))
    w2 = relay.var("w2", shape=(2 * j, k))
    b1 = relay.var("b1", shape=(j,))
    b2 = relay.var("b2", shape=(2 * j,))
    scale1 = relay.var("scale1", shape=(1,))
    scale2 = relay.var("scale2", shape=(1,))

    args = [x, w1, w2, b1, b2, scale1, scale2]
    y1 = relay.nn.dense(x, w1)
    y2 = relay.nn.dense(x, w2)
    y1 = relay.add(y1, b1)
    y2 = relay.add(y2, b2)
    y1 = relay.multiply(y1, scale1)
    y2 = relay.multiply(y2, scale2)
    y1 = relay.reshape(y1, newshape=newshape1)
    y2 = relay.reshape(y2, newshape=newshape2)
    y = relay.Tuple((y1, y2))

    return relay.Function(args, y)


def cpbm_case1(b=1, i=100, j=200, k=300):
    """CombineParallelBatchMatmul"""
    # transform.CombineParallelBatchMatmul(min_num_branches=2)
    x = relay.var("x", shape=(b, i, k))
    w1 = relay.var("w1", shape=(b, j, k))
    w2 = relay.var("w2", shape=(b, j, k))
    w3 = relay.var("w3", shape=(b, j, k))

    args = [x, w1, w2, w3]
    y1 = relay.nn.batch_matmul(x, w1)
    y2 = relay.nn.batch_matmul(x, w2)
    y3 = relay.nn.batch_matmul(x, w3)
    y = relay.Tuple((y1, y2, y3))

    return relay.Function(args, y)


def cpbm_case2(b=1, i=100, j=200, k=300):
    x = relay.var("x", shape=(b, i, k))
    w1 = relay.var("w1", shape=(b, j, k))
    w2 = relay.var("w2", shape=(b, j, k))
    w3 = relay.var("w3", shape=(b, j, k))
    b1 = relay.var("b1", shape=(j,))
    b2 = relay.var("b2", shape=(j,))
    b3 = relay.var("b3", shape=(j,))

    args = [x, w1, w2, w3, b1, b2, b3]
    y1 = relay.nn.batch_matmul(x, w1)
    y2 = relay.nn.batch_matmul(x, w2)
    y3 = relay.nn.batch_matmul(x, w3)
    y1 = relay.add(y1, b1)
    y2 = relay.add(y2, b2)
    y3 = relay.add(y3, b3)
    y = relay.Tuple((y1, y2, y3))

    return relay.Function(args, y)


def fsa_case1(shape=(2, 4, 10, 10), channels=2, blocking=None):
    # (2, 2, 10, 10, 2), 8, (2, 4)
    # transform.ForwardFoldScaleAxis()
    # FoldScaleAxis会调用ForwardFoldScaleAxis和BackwardFoldScaleAxis
    x = relay.var("x", shape=shape)
    conv_weight = relay.var("weight")
    if blocking:
        in_channels = shape[1] * shape[4]
        in_bias = relay.var("in_bias", shape=(1, in_channels // blocking[0], 1, 1, blocking[0]))
        in_scale = relay.const(
            _get_positive_scale((1, in_channels // blocking[0], 1, 1, blocking[0]))
        )
    else:
        in_channels = shape[1]
        in_bias = relay.var("in_bias", shape=(in_channels, 1, 1))
        in_scale = relay.const(_get_positive_scale((in_channels, 1, 1)))

    args = [x, conv_weight, in_bias]
    x = relay.multiply(x, in_scale)
    x = relay.nn.relu(x)
    x = relay.add(x, in_bias)
    y = relay.nn.conv2d(
        x,
        conv_weight,
        channels=channels,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
        kernel_layout="OIHW2i{}o".format(blocking[1]) if blocking else "OIHW",
    )

    return relay.Function(args, y)


def fsa_case2(dshape=(2, 4, 10, 3), channels=3, blocking=None):
    # scale axis being consumed by two consumers
    # (2, 4, 10, 2, 2), 4, (2, 2)
    x = relay.var("x", shape=dshape)
    if blocking:
        in_channels = dshape[3] * dshape[4]
        wshape = (3, 3, 1, channels // blocking[1], 1, blocking[1])  # HWIOio
        conv_weight = relay.var("weight", shape=wshape)
        in_bias = relay.var("in_bias", shape=(in_channels // blocking[0], blocking[0]))
        in_scale = relay.const(_get_positive_scale((in_channels // blocking[0], blocking[0])))
    else:
        in_channels = dshape[-1]
        wshape = (3, 3, 1, channels)  # HWIO
        conv_weight = relay.var("weight", shape=wshape)
        in_bias = relay.var("in_bias", shape=(in_channels,))
        in_scale = relay.const(
            _get_positive_scale(
                in_channels,
            )
        )

    # test depthwise
    assert in_channels == channels

    args = [x, conv_weight, in_bias]
    x = relay.multiply(in_scale, x)
    x = relay.nn.relu(x)
    x = relay.subtract(x, in_bias)
    y1 = relay.nn.conv2d(
        x,
        conv_weight,
        channels=channels,
        kernel_size=(3, 3),
        data_layout="NHWC{}c".format(blocking[0]) if blocking else "NHWC",
        kernel_layout="HWIO1i{}o".format(blocking[1]) if blocking else "HWIO",
        groups=channels,
        padding=(1, 1),
    )
    y2 = relay.nn.conv2d(
        x,
        conv_weight,
        channels=channels,
        kernel_size=(3, 3),
        data_layout="NHWC{}c".format(blocking[0]) if blocking else "NHWC",
        kernel_layout="HWIO1i{}o".format(blocking[1]) if blocking else "HWIO",
        groups=channels,
        padding=(1, 1),
    )
    z = relay.add(y1, y2)

    return relay.Function(args, z)


def fsa_case3(shape=(2, 4, 10, 10), channels=4, blocking=None):
    # folding negative scale
    # (2, 2, 10, 10, 2), 8, (2, 2)
    x = relay.var("x", shape=shape)
    if blocking:
        in_channels = shape[1] * shape[4]
        in_scale = relay.const(-_get_positive_scale((1, shape[1], 1, 1, shape[4])))
    else:
        in_channels = shape[1]
        in_scale = relay.const(-_get_positive_scale((in_channels, 1, 1)))
    conv_weight = relay.var("weight")

    args = [x, conv_weight]
    x = relay.multiply(x, in_scale)
    y = relay.nn.conv2d(
        x,
        conv_weight,
        channels=channels,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
        kernel_layout="OIHW4i{}o".format(blocking[1]) if blocking else "OIHW",
    )

    return relay.Function(args, y)


def fsa_case4(data_shape=(3, 5), weight_shape=(4, 5)):
    # dense
    x = relay.var("x", shape=data_shape)
    weight = relay.var("weight", shape=weight_shape)
    in_channels = data_shape[1]
    in_bias = relay.var("in_bias", shape=(in_channels,))
    in_scale = relay.const(_get_positive_scale((in_channels,)))

    args = [x, weight, in_bias]
    x = relay.multiply(x, in_scale)
    x = relay.nn.relu(x)
    x = relay.add(x, in_bias)
    y = relay.nn.dense(x, weight)

    return relay.Function(args, y)


def fsa_case5(shape=(2, 4, 10, 10), in_channels=4, channels=8, blocking=None):
    """BackwardFoldScaleAxis"""
    # transform.BackwardFoldScaleAxis()
    # (2, 2, 10, 10, 16), 32, 64, (16, 16)
    x = relay.var("x", shape=shape)
    conv_weight = relay.var("weight")
    out_bias = relay.var("out_bias", shape=(channels,))
    if blocking:
        out_scale = relay.const(_get_positive_scale((channels,)))
    else:
        out_scale = relay.const(_get_positive_scale((channels, 1, 1)))

    args = [x, conv_weight, out_bias]
    if blocking:
        out_bias = relay.reshape(out_bias, (1, channels // blocking[1], 1, 1, blocking[1]))
    else:
        out_bias = relay.expand_dims(out_bias, axis=1, num_newaxis=2)
    y = relay.nn.conv2d(
        x,
        conv_weight,
        channels=channels,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
        kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
    )
    y = relay.add(y, out_bias)
    y = relay.nn.relu(y)
    if blocking:
        out_scale = relay.reshape(out_scale, (1, channels // blocking[1], 1, 1, blocking[1]))
    y = relay.multiply(y, out_scale)

    return relay.Function(args, y)


def fsa_case6(shape=(2, 4, 10, 10), in_channels=4, channels=8, blocking=None):
    # (2, 2, 10, 10, 2), 4, 8, (2, 2)
    x = relay.var("x", shape=shape)
    conv_weight = relay.var("weight")
    if blocking:
        out_bias = relay.var("out_bias", shape=(channels // blocking[1], 1, 1, blocking[1]))
        out_scale = relay.const(
            _get_positive_scale((channels // blocking[1], 1, 1, blocking[1]))
        )
    else:
        out_bias = relay.var("out_bias", shape=(channels,))
        out_scale = relay.const(_get_positive_scale((channels, 1, 1)))

    args = [x, conv_weight, out_bias]
    y1 = relay.nn.conv2d(
        x,
        conv_weight,
        channels=channels,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
        kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
    )
    y1 = relay.nn.relu(y1)
    y2 = relay.nn.conv2d(
        x,
        conv_weight,
        channels=channels,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
        kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
    )
    y2 = relay.nn.relu(y2)
    y = relay.add(y1, y2)
    y = relay.multiply(y, out_scale)

    return relay.Function(args, y)


def fsa_case7(shape=(2, 4, 10, 10), channels=8, blocking=None):
    # (2, 2, 10, 10, 2), 8, (2, 2)
    x = relay.var("x", shape=shape)
    conv_weight = relay.var("weight")
    if blocking:
        out_scale = relay.const(
            -_get_positive_scale((1, channels // blocking[1], 1, 1, blocking[1]))
        )
    else:
        out_scale = relay.const(-_get_positive_scale((channels, 1, 1)))

    args = [x, conv_weight]
    y = relay.nn.conv2d(
        x,
        conv_weight,
        channels=channels,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
        kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
    )
    y = relay.multiply(y, out_scale)

    return relay.Function(args, y)


def fsa_case8(data_shape=(3, 5), weight_shape=(4, 5)):
    # dense
    x = relay.var("x", shape=data_shape)
    weight = relay.var("weight", shape=weight_shape)
    out_channels = weight_shape[0]
    in_bias = relay.var("in_bias", shape=(out_channels,))
    in_scale = relay.const(_get_positive_scale((out_channels,)))

    args = [x, weight, in_bias]
    x = relay.nn.dense(x, weight)
    x = relay.add(x, in_bias)
    x = relay.nn.relu(x)
    y = relay.multiply(x, in_scale)

    return relay.Function(args, y)


def fsa_case9(shape=(2, 4, 10, 10), channels=4):
    # bias add
    x = relay.var("x", shape=shape)
    conv_weight = relay.var("weight")
    out_bias = relay.var("out_bias", shape=(channels,))
    out_scale = relay.const(_get_positive_scale((channels, 1, 1)))

    args = [x, conv_weight, out_bias]
    y = relay.nn.conv2d(
        x,
        conv_weight,
        channels=channels,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW",
    )
    y = relay.nn.bias_add(y, out_bias)
    y = relay.nn.relu(y)
    y = relay.multiply(y, out_scale)

    return relay.Function(args, y)


def se_case1():
    """SimplifyExpr"""
    # simplify reshape
    # transform.SimplifyExpr()
    x = relay.var("x", shape=(1, 16, 16, 16), dtype="float32")
    w = relay.var("w", shape=(32, 16, 3, 3), dtype="float32")
    y = relay.nn.conv2d(x, w, padding=(1, 1))
    y = relay.reshape(y, newshape=(1, 16, -1))
    y = relay.reshape(y, newshape=(4, 8, -1, 16))
    y = relay.reverse_reshape(y, newshape=(32, 0, -1))

    return relay.Function([x, w], y)


def se_case2():
    # simplify transpose
    x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")  # NCHW
    c = relay.const(0.1)
    y = relay.nn.relu(x)
    y = relay.transpose(y, axes=[0, 2, 3, 1])  # To NHWC
    y = relay.transpose(y, axes=[1, 2, 3, 0])  # To HWCN
    y = relay.transpose(y, axes=[3, 2, 0, 1])  # To NCHW
    y = relay.add(y, c)

    return relay.Function([x], y)


def se_case3():
    # concretize ones_like/zeros_like
    dtype = "int32"
    shape_like = relay.var("shape_like", shape=(3, 4, 5), dtype=dtype)
    c = relay.const(1)
    x = relay.ones_like(shape_like)
    y = relay.zeros_like(shape_like)
    z = relay.add(x, y)
    z = relay.add(z, c)

    return relay.Function([shape_like], z)


def se_case4():
    # simplify consecutive add
    shape = (32, 1, 1)
    c_data = np.empty(shape).astype("float32")
    c1 = relay.const(c_data)
    c2 = relay.const(c_data)

    x = relay.var("x", shape=(1, 16, 16, 16), dtype="float32")
    w = relay.var("w", shape=(32, 16, 3, 3), dtype="float32")
    y = relay.nn.conv2d(x, w, padding=(1, 1))
    y = relay.add(y, c1)
    y = relay.add(y, c2)
    y = relay.nn.relu(y)

    return relay.Function([x, w], y)


def fac_case():
    """FlattenAtrousConv"""
    # transform.FlattenAtrousConv()
    # pattern entry with block_shape=[2, 2]
    shape_x = [1, 5, 5, 4]
    shape_w = [3, 3, 4, 1]

    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")
    c = relay.const(0.1)

    weight = relay.const(w_np)
    data = relay.var("data", shape=shape_x, dtype="float32")
    op1 = relay.nn.space_to_batch_nd(data, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
    op2 = relay.nn.conv2d(
        op1,
        weight,
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
    z = relay.nn.batch_to_space_nd(op2, block_shape=[2, 2], crops=[[0, 1], [0, 1]])
    z = relay.add(z, c)

    return relay.Function([data], z)


def fm_case():
    """FastMath"""
    # transform.FastMath()
    x = relay.var("x", shape=(1, 16, 16, 16), dtype="float32")
    y = relay.exp(x)
    y = relay.tanh(y)
    y = relay.erf(y)
    y = relay.nn.softmax(y)

    return relay.Function([x], y)


def cl_case1():
    """ConvertLayout"""
    # conv convert layout
    # transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]})
    x = relay.var("x", shape=(1, 56, 56, 64))
    weight = relay.var("weight", shape=(3, 3, 64, 64))
    y = relay.nn.conv2d(
        x,
        weight,
        channels=64,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWIO",
    )
    y = relay.nn.relu(y)
    y = relay.Function([x, weight], y)

    return y


def cl_case2():
    # conv nhwc convert layout
    # transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})
    x = relay.var("x", shape=(1, 64, 56, 56))
    weight = relay.var("weight", shape=(64, 64, 3, 3))
    y = relay.nn.conv2d(
        x,
        weight,
        channels=64,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW",
    )
    y = relay.nn.relu(y)
    y = relay.Function([x, weight], y)

    return y


def cl_case3():
    # conv_transpose convert layout
    # transform.ConvertLayout({"nn.conv2d_transpose": ["NCHW", "IOHW"]})
    x = relay.var("x", shape=(1, 56, 56, 64))
    weight = relay.var("weight", shape=(3, 3, 64, 64))
    y = relay.nn.conv2d_transpose(
        x,
        weight,
        channels=64,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWIO",
    )
    y = relay.nn.relu(y)
    y = relay.Function([x, weight], y)

    return y


def cl_case4():
    # conv_bias_pool uses specified convert layout
    # transform.ConvertLayout({"nn.conv2d": ["NCHW", "OIHW"], "nn.max_pool2d": ["NHWC"]})
    x = relay.var("x", shape=(1, 56, 56, 64))
    bias = relay.var("bias", shape=(64,))
    weight = relay.var("weight", shape=(3, 3, 64, 64))
    y = relay.nn.conv2d(
        x,
        weight,
        channels=64,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWIO",
    )
    y = relay.nn.bias_add(y, bias, axis=3)
    # a useless tuple, which will be eliminated
    y = relay.Tuple([y])[0]
    y = relay.nn.relu(y)
    y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout="NHWC")
    y = relay.cast(y, "int32")
    y = relay.nn.batch_flatten(y)
    y = relay.Function(relay.analysis.free_vars(y), y)

    return y


if __name__ == '__main__':
    f = mc_case4()
    mod = tvm.IRModule.from_expr(f)
    mod_func = mod["main"]
    # mod_func = testing.run_opt_pass(mod_func, transform.InferType())
    relay_ir2txt(mod_func, file_name='tvm_case', is_ap=False)

    mod_opt = testing.run_opt_pass(mod_func, transform.MergeComposite(pattern_table))
    relay_ir2txt(mod_opt, file_name='tvm_case', is_ap=True)