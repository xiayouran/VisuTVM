# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    youran.xia@foxmail.com
# Datetime: 2022/9/19 20:49
# Filename: from_onnx.py
import onnx

from tvm import relay
from tvm.relay import transform, testing

from utils import relay_ir2txt
from pass_map import PASS_MAP
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--passname', type=str, required=True, help="use Pass's name")
args = parser.parse_args()

pass_dict = PASS_MAP.get(args.passname, {})
assert pass_dict, "not support {} pass to visu now!".format(args.passname)

file_name = 'resnet18_onnx_' + pass_dict['name']
pass_func = pass_dict['pass']


if __name__ == '__main__':
    # https://github.com/onnx/models/tree/main/vision/classification/resnet
    model_path = 'resnet18-v1-7.onnx'
    onnx_model = onnx.load(model_path)

    input_shape = [1, 3, 224, 224]
    input_name = "data"
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    mod_func = mod["main"]
    # print('Relay IR before Pass:\n', mod_func)
    relay_ir2txt(mod_func, file_name, is_ap=False)

    mod_opt = testing.run_opt_pass(mod_func, pass_func)
    # print('Relay IR after Pass:\n', mod_opt)
    relay_ir2txt(mod_opt, file_name, is_ap=True)
