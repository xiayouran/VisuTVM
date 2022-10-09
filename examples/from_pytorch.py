# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    youran.xia@foxmail.com
# Datetime: 2022/9/19 17:10
# Filename: from_pytorch.py
"""
pytorch version: 1.7.1
"""
import torch
from torchvision import models

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

file_name = 'resnet18_' + pass_dict['name']
pass_func = pass_dict['pass']


if __name__ == '__main__':
    model_name = "resnet18"
    # pth_file = 'resnet18-f37072fd.pth'
    model = models.resnet18()
    # ckpt = torch.load(pth_file)
    # model.load_state_dict(ckpt)
    model = model.eval()

    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    mod_func = mod["main"]
    # print('Relay IR before Pass:\n', mod_func)
    relay_ir2txt(mod_func, file_name, is_ap=False)

    mod_opt = testing.run_opt_pass(mod_func, pass_func)
    # print('Relay IR after Pass:\n', mod_opt)
    relay_ir2txt(mod_opt, file_name, is_ap=True)
