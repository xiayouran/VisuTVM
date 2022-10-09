# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    liyanpeng@tsingmicro.com
# Datetime: 2022/9/30 10:46
# Filename: pass_map.py
from tvm.relay import transform


PASS_MAP = {
    'FuseOps': {'name': 'fo', 'pass': transform.FuseOps(fuse_opt_level=2)},
    'RemoveUnusedFunctions': {'name': 'ruf', 'pass': transform.RemoveUnusedFunctions()},
    'ToBasicBlockNormalForm': {'name': 'tbbnf', 'pass': transform.ToBasicBlockNormalForm()},
    'EliminateCommonSubexpr': {'name': 'ecs', 'pass': transform.EliminateCommonSubexpr()},
    'FoldConstant': {'name': 'fc', 'pass': transform.FoldConstant()},
    'SimplifyInference': {'name': 'si', 'pass': transform.SimplifyInference()},
    'CombineParallelConv2D': {'name': 'cpc2d', 'pass': transform.CombineParallelConv2D(min_num_branches=3)},
    'CombineParallelDense': {'name': 'cpd', 'pass': transform.CombineParallelDense(min_num_branches=3)},
    'CombineParallelBatchMatmul': {'name': 'cpbm', 'pass': transform.CombineParallelBatchMatmul(min_num_branches=3)},
    'FoldScaleAxis': {'name': 'fsa', 'pass': transform.FoldScaleAxis()},
    'SimplifyExpr': {'name': 'se', 'pass': transform.SimplifyExpr()},
    'CanonicalizeCast': {'name': 'cc', 'pass': transform.CanonicalizeCast()},
    'CanonicalizeOps': {'name': 'co', 'pass': transform.CanonicalizeOps()},
    'FlattenAtrousConv': {'name': 'fac', 'pass': transform.FlattenAtrousConv()},
    'FastMath': {'name': 'fm', 'pass': transform.FastMath()},
    'ConvertLayout': {'name': 'cl', 'pass': transform.ConvertLayout()},
}