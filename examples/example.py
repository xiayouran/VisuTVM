# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    youran.xia@foxmail.com
# Datetime: 2022/9/19 17:10
# Filename: example.py
"""
可用于 FuseOps 可视化
"""
import torch
from torch import nn

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

file_name = 'example_' + pass_dict['name']
pass_func = pass_dict['pass']


class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = x + 0.1
        x = self.conv1(x)
        x0 = x + 0.1
        x = x + x0
        x1 = self.conv2(x)
        x2 = self.conv3(x)
        out = x1 + x2

        return out


if __name__ == '__main__':
    model = ExampleModel()
    model.eval()

    input_shape = [1, 16, 64, 64]
    input_data = torch.randn(input_shape, dtype=torch.float32)
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
