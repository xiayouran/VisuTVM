# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    youran.xia@foxmail.com
# Datetime: 2022/9/19 20:34
# Filename: utils.py
import os
import glob
import warnings
import numpy as np

from visu_tvm import VisuGraph, VisuGraphFuseOps, VisuGraphRUF, VisuGraphMC


def relay_ir2txt(context, file_name='example', is_ap=False):
    save_path = 'relay_ir'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if is_ap:
        file_name += '_ap.txt'
    else:
        file_name += '_bp.txt'

    with open(os.path.join(save_path, file_name), 'w', encoding='utf-8') as f:
        f.writelines(str(context))


def visu_relay_ir(bp_file, ap_file, save_name, with_info=False):
    g = VisuGraph(txt_file=bp_file, save_name=save_name, with_info=with_info)
    g.codegen()

    if '_fo_' in ap_file:
        g = VisuGraphFuseOps(txt_file=ap_file, save_name=save_name, with_info=with_info)
    elif '_ruf_' in ap_file or '_fc_' in ap_file or '_ecs_' in ap_file or '_si_' in ap_file or '_fm_' in ap_file or \
            '_se_' in ap_file or '_fac_' in ap_file or '_cc_' in ap_file or '_cl_' in ap_file or '_fsa_' in ap_file or \
            '_cpc2d_' in ap_file or '_cpd_' in ap_file or '_cpbm_' in ap_file:
        g = VisuGraphRUF(txt_file=ap_file, save_name=save_name, with_info=with_info)
    elif '_mc_' in ap_file:
        g = VisuGraphMC(txt_file=ap_file, save_name=save_name, with_info=with_info)
    else:
        warnings.warn("not support the pass to visu now! ==> {}".format(ap_file))
        # TODO 由于没有合适的case，部分Pass优化后的Relay IR可视化可能会失败
        #  有些Pass在优化神经网络(目前只在resnet18上进行了测试)的时候可能不起作用，因此Pass优化前后的可视化结果是一样的
        g = VisuGraphRUF(txt_file=ap_file, save_name=save_name, with_info=with_info)
    g.codegen()


def run_all_examples(scan_dir='relay_ir', with_info=False):
    bp_list = glob.glob(os.path.join(scan_dir, '*_bp.txt'))
    for bp_file in bp_list:
        ap_file = bp_file.replace('_bp', '_ap')
        save_name = bp_file.replace('.txt', '')

        print("Parsing {} and {}".format(bp_file, ap_file))
        visu_relay_ir(bp_file, ap_file, save_name, with_info)


def _get_positive_scale(size):
    return np.random.uniform(0.5, 1, size=size).astype("float32")


if __name__ == '__main__':
    run_all_examples()
    run_all_examples(scan_dir='relay_ir/tvm_case')
