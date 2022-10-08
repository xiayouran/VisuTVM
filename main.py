# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    youran.xia@foxmail.com
# Datetime: 2022/9/19 16:59
# Filename: main.py
import argparse
from utils import visu_relay_ir


parser = argparse.ArgumentParser()
parser.add_argument('--before_pass', '-bp', type=str, default='relay_ir/example_fo_bp.txt',
                    help='relay ir before pass txt file')
parser.add_argument('--after_pass', '-ap', type=str, default='relay_ir/example_fo_ap.txt',
                    help='relay ir after pass txt file')
parser.add_argument('--save_name', '-sn', type=str, default='example',
                    help='png save name')
args = parser.parse_args()


if __name__ == '__main__':
    save_name = args.save_name
    before_pass = args.before_pass
    after_pass = args.after_pass

    visu_relay_ir(before_pass, after_pass, save_name)
    print('Finshed!')
