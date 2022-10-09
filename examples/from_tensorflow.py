# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    liyanpeng@tsingmicro.com
# Datetime: 2022/10/9 12:52
# Filename: from_tensorflow.py
"""
tensorflow version: 2.10.0
"""
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

import numpy as np

from tvm import relay
from tvm.relay import transform, testing

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

from utils import relay_ir2txt
from pass_map import PASS_MAP
import argparse

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf


seed = 10086
tf.random.set_seed(seed)
np.random.seed(seed)


def save_pb_model():
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        # NHWC
        # input_data = tf.random.normal(shape=(10, 32, 32, 16))
        # weight_data = tf.random.normal(shape=(3, 3, 16, 64))

        input_data = np.random.normal(size=(10, 32, 32, 16)).astype(np.float32)
        weight_data = np.random.normal(size=(3, 3, 16, 64)).astype(np.float32)

        # x = tf.space_to_batch_nd(input_data, block_shape=[2, 2], paddings=[[0, 0], [0, 0]])
        # x = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')
        # output = tf.compat.v1.batch_to_space_nd(x, block_shape=[2, 2], crops=[[0, 0], [0, 0]])

        # output1 = tf.nn.conv2d(input_data, weight, strides=[1, 1, 1, 1], padding='SAME', dilations=2)
        # print(output.shape)
        # print(output == output1)

        x = tf.compat.v1.placeholder(tf.float32, shape=[10, 32, 32, 16], name='x')
        weight = tf.Variable(weight_data, name='weight')

        x1 = tf.space_to_batch_nd(x, block_shape=[2, 2], paddings=[[0, 0], [0, 0]])
        x2 = tf.nn.conv2d(x1, weight, strides=[1, 1, 1, 1], padding='SAME')     # 空洞卷积不改变feature map大小
        output = tf.compat.v1.batch_to_space_nd(x2, block_shape=[2, 2], crops=[[0, 0], [0, 0]], name='output')

        sess.run(tf.compat.v1.global_variables_initializer())

        # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])

        # 测试 OP
        feed_dict = {x: input_data}
        print(sess.run(output, feed_dict))

        # 写入序列化的 PB 文件
        with tf.compat.v1.gfile.FastGFile('atrous_model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())


def load_pb_model():
    pb_file_path = 'atrous_model.pb'
    sess = tf.compat.v1.Session()
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  # 导入计算图

    input_x = sess.graph.get_tensor_by_name('x:0')
    output = sess.graph.get_tensor_by_name('output:0')

    input_data = np.random.normal(size=(10, 32, 32, 16)).astype(np.float32)
    ret = sess.run(output, feed_dict={input_x: input_data})
    print(ret.shape)


parser = argparse.ArgumentParser()
parser.add_argument('--passname', type=str, required=True, help="use Pass's name")
args = parser.parse_args()

pass_dict = PASS_MAP.get(args.passname, {})
assert pass_dict, "not support {} pass to visu now!".format(args.passname)

file_name = 'atrous_' + pass_dict['name']
pass_func = pass_dict['pass']


if __name__ == '__main__':
    save_pb_model()

    model_path = 'atrous_model.pb'
    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name="")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        # Add shapes to the graph.
        # with tf_compat_v1.Session() as sess:
        #     graph_def = tf_testing.AddShapesToGraphDef(sess, "softmax")

    shape_dict = {"x": [10, 32, 32, 16]}
    mod, params = relay.frontend.from_tensorflow(graph_def, layout=None, shape=shape_dict)

    mod_func = mod["main"]
    # print('Relay IR before Pass:\n', mod_func)
    relay_ir2txt(mod_func, file_name, is_ap=False)

    mod_opt = testing.run_opt_pass(mod_func, pass_func)
    # print('Relay IR after Pass:\n', mod_opt)
    relay_ir2txt(mod_opt, file_name, is_ap=True)
