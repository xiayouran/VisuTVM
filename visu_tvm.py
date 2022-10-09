# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    youran.xia@foxmail.com
# Datetime: 2022/9/19 17:07
# Filename: visu_tvm.py
import re
import random


__all__ = ["VisuGraph", "VisuGraphFuseOps", "VisuGraphRUF", "VisuGraphCPC2D"]


class PNode(object):
    def __init__(self, name='', type='', inputs=None, body='') -> None:
        if inputs is None:
            inputs = []
        self.name = name
        self.type = type
        self.inputs = inputs
        self.body = body
        self.color = ''


class IRNode(object):
    def __init__(self, name='None', label='None', color='', style='', inputs=None) -> None:
        if inputs is None:
            inputs = []
        self.name = name
        self.label = label
        self.color = color
        self.style = style
        self.inputs = inputs


class IREdge(object):
    def __init__(self, tail_name, head_name) -> None:
        self.tail_name = tail_name
        self.head_name = head_name


class VisuGraph(object):
    """Visu TVM Relay IR"""
    def __init__(self, txt_file, save_name='example') -> None:
        self.graph_code = ''
        self.nodes = dict()
        self.edges = list()
        self.node_code = ''
        self.edge_code = ''
        self.parse_res = list()
        self.save_name = 'output/visu_{}_relay_ir'.format(save_name)
        self.txt_file = txt_file

    def random_color(self):
        colors = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        color = random.sample(colors, k=6)
        color = ''.join(color)

        return '#' + color

    def parse_txt(self, txt_file=''):
        assert txt_file, "You must have a txt file!!!"

        with open(txt_file, 'r') as f:
            lines = f.readlines()
        # pattern_iargs = re.compile(r'(%[a-z]+\d+\.*[a-z]*):')
        # '%0 = add(%input0, 0.1f);'
        pattern1 = re.compile(r'\s/\*.+?\*/')
        pattern2 = re.compile(r'(%\d+) = (.+)')
        for line in lines[1:-1]:
            line = re.sub(pattern1, '', line)
            line = line.strip().strip(';')
            match_op = re.findall(pattern2, line)
            if match_op:
                self.parse_res.extend(match_op)
            else:
                self.parse_res.append(('', line))

    def init_node(self):
        # graph.node(name='%input0', label='%input0', color='white', style='filled')
        # base_node = "graph.node(name='{}', label='{}', color='{}', style='{}')\n"
        for k, v in self.nodes.items():
            self.node_code += "graph.node(name='{}', label='{}', color='{}', style='{}')\n".format(k, v.label, v.color,
                                                                                                   v.style)

    def init_edge(self):
        # graph.edge(tail_name='%input0', head_name='%0', label='')
        # base_eage = "graph.edge(tail_name='{}', head_name='{}')\n"
        for edge in self.edges:
            self.edge_code += "graph.edge(tail_name='{}', head_name='{}')\n".format(edge.tail_name, edge.head_name)

    def codegen(self):
        from graphviz import Digraph

        self.parse_txt(self.txt_file)
        self.parse_node()
        self.parse_edge()
        self.init_node()
        self.init_edge()

        graph = Digraph(name='RelayIR')
        exec(self.node_code)
        exec(self.edge_code)
        graph.render(filename=self.save_name, format='png')

    def parse_node(self):
        # pattern1 = re.compile(r'(%[a-z]*(\d*\.?_?[a-z]*\d*)*)')
        pattern1 = re.compile(r'(%[a-z]*(\d*\.?_?[a-z]*\d*)*|meta\[relay\.Constant]\[\d*])')

        node_map = dict()

        for info in self.parse_res:
            assert len(info) == 2, 'length of info must be 2!!!'
            if '(%' not in info[1] and '%' in info[1]:
                # '%16 = %15.0;
                node_map[info[0]] = info[1][:-2]
                continue

            index = info[1].find('(')
            if 'add(' in info[1] or 'multiply(' in info[1]:
                # add multiply 两数之间的运算
                args_list = info[1][index+1:-1].split(', ')
                args_list = [node_map.get(arg, arg) for arg in args_list]
            else:
                args_list = re.findall(pattern1, info[1])
                args_list = [node_map.get(arg[0], arg[0]) for arg in args_list]
            self.nodes[info[0]] = IRNode(name=info[0], label=info[1][:index], inputs=args_list)
            for n in args_list:
                if not self.nodes.get(n, ''):
                    self.nodes[n] = IRNode(name=n, label=n, color='white')

    def parse_edge(self):
        for k, v in self.nodes.items():
            if len(v.inputs) > 0:
                for n in v.inputs:
                    self.edges.append(IREdge(tail_name=n, head_name=k))
            # elif not k and len(v.inputs) > 0:
            #     for n in v.inputs:
            #         self.edges.append(IREdge(tail_name=n, head_name=v.label))


class VisuGraphFuseOps(VisuGraph):
    """Visu FuseOP Pass Relay IR"""
    def __init__(self, txt_file, save_name='example') -> None:
        super(VisuGraphFuseOps, self).__init__(txt_file, save_name)
        self.op_args_map = dict()
        self.save_name = 'output/visu_{}_relay_ir_pass'.format(save_name)

    def parse_txt(self, txt_file=''):
        assert txt_file, "You must have a txt file!!!"

        with open(txt_file, 'r') as f:
            lines = f.readlines()

        fn_flag = False
        fn_str = ''
        pattern = re.compile(r'\s/\*.+?\*/')
        for line in lines[1:-1]:
            line = line.strip()
            line = re.sub(pattern, '', line)
            if ' fn ' in line:
                fn_str = line
                fn_flag = True
            elif '}' not in line and fn_flag:
                fn_str += line
            elif '}' in line and fn_flag:
                fn_str += line
                fn_flag = False
                self.parse_res.append(fn_str)
            else:
                self.parse_res.append(line)

    def parse_node(self):
        pattern1 = re.compile(r'(%\d+).+{(.+)}')
        pattern1_ = re.compile(r'(%[a-z]*\d+):')
        pattern2 = re.compile(r'(%\d+).+?(%\d+)\((.+)\)')
        pattern3 = re.compile(r'(%\d+)\((.+)\)')
        pattern4 = re.compile(r'(%[a-z]*\d+),?')

        # 对解析的结果进一步划分成fn和op
        pnodes = dict()
        for fn_str in self.parse_res:
            if ' fn ' in fn_str:
                match_op = re.search(pattern1, fn_str).groups(0)
                args = re.findall(pattern1_, fn_str)    # fn的输入参数
                pnodes[match_op[0]] = PNode(name=match_op[0], type='fn', inputs=args, body=match_op[-1])
            elif ' = ' in fn_str:
                match_op = re.search(pattern2, fn_str).groups(0)
                args = match_op[-1].split(', ')     # op的输入参数
                pnodes[match_op[0]] = PNode(name=match_op[0], type='op', inputs=args, body=match_op[1])
            else:
                match_op = re.search(pattern3, fn_str).groups(0)
                args = match_op[-1].split(', ')
                pnodes[''] = PNode(name='', type='op', inputs=args, body=match_op[0])

        # 将op进行细化
        node_map = dict()
        for k, v in pnodes.items():
            if v.type == 'fn':
                v.color = self.random_color()
                continue
            pre_info = pnodes[v.body]
            ops = pre_info.body
            fn_args = pre_info.inputs
            color = pre_info.color

            op_args = v.inputs

            # 做实参与形参映射
            for i, args in enumerate(fn_args):
                self.op_args_map[args] = op_args[i]

            # FunseOP --> 分析fn
            ops_list = ops.split(';')
            for ops_ in ops_list:
                if ' = ' in ops_:
                    # fn中含有多个op
                    match_op = ops_.split(' = ')

                    if '(%' not in ops_:
                        # '%29 = %p052.0'
                        node_map[match_op[0]] = match_op[1][:-2]
                        continue

                    # index = ops_.find('(')
                    index = match_op[-1].find('(')
                    if 'add(' in ops_:
                        # 含=的add
                        args_list = match_op[-1][index+1:-1].split(', ')
                        # args_list = re.findall(pattern4, match_op[-1])
                        args_list = [node_map.get(arg, arg) for arg in args_list]
                    else:
                        # 含=的op
                        args_list = re.findall(pattern4, match_op[-1])
                        args_list = [node_map.get(arg, arg) for arg in args_list]

                    args_list = [self.op_args_map.get(arg, arg) for arg in args_list]

                    self.nodes[match_op[0]] = IRNode(name=match_op[0], label=match_op[-1][:index], inputs=args_list,
                                                     color=color, style='filled')
                    for n in args_list:
                        if not self.nodes.get(n, ''):
                            self.nodes[n] = IRNode(name=n, label=n, color='white')

                else:
                    # op中不含=
                    index = ops_.find('(')
                    if 'add(' in ops_:
                        # add 加常数
                        args_list = ops_[index+1:-1].split(', ')
                    else:
                        # conv2d, batchnorm, relu, maxpool2d, adaptive_avg_pool2d, squeeze
                        args_list = re.findall(pattern4, ops_)
                    args_list = [node_map.get(arg, arg) for arg in args_list]
                    args_list = [self.op_args_map.get(arg, arg) for arg in args_list]

                    self.nodes[k] = IRNode(name=k, label=ops_[:index], inputs=args_list, color=color, style='filled')
                    for n in args_list:
                        if not self.nodes.get(n, ''):
                            self.nodes[n] = IRNode(name=n, label=n, color='white')


class VisuGraphRUF(VisuGraph):
    """Visu RemoveUnusedFunctions /
            ToBasicBlockNormalForm /
            EliminateCommonSubexpr /
            FoldConstant /
            SimplifyInference /
            FastMath /
            SimplifyExpr Pass Relay IR"""
    def __init__(self, txt_file, save_name='example') -> None:
        super(VisuGraphRUF, self).__init__(txt_file, save_name)
        self.op_args_map = dict()
        self.save_name = 'output/visu_{}_relay_ir_pass'.format(save_name)

    def parse_node(self):
        # pattern1 = re.compile(r'(%[a-z]*(\d*\.?_?[a-z]*\d*)*)')
        pattern1 = re.compile(r'(%[a-z]*(\d*\.?_?[a-z]*\d*)*|meta\[relay\.Constant]\[\d*])')

        node_map = dict()

        for info in self.parse_res:
            assert len(info) == 2, 'length of info must be 2!!!'
            # if '(%' not in info[1] and '%' in info[1]:
            if '(%' not in info[1] and '.0' in info[1]:
                # '%16 = %15.0;
                node_map[info[0]] = info[1][:-2]
                continue

            index = info[1].find('(')
            if 'add(' in info[1] or 'multiply(' in info[1] or 'divide(' in info[1]:
                args_list = info[1][index+1:-1].split(', ')
                args_list = [node_map.get(arg, arg) for arg in args_list]
            else:
                args_list = re.findall(pattern1, info[1])
                args_list = [node_map.get(arg[0], arg[0]) for arg in args_list]
            self.nodes[info[0]] = IRNode(name=info[0], label=info[1][:index], inputs=args_list, color=self.random_color(), style='filled')
            for n in args_list:
                if not self.nodes.get(n, ''):
                    self.nodes[n] = IRNode(name=n, label=n, color='white')


class VisuGraphCPC2D(VisuGraph):
    """Visu CombineParallelConv2D /
            CombineParallelDense /
            CombineParallelBatchMatmul Pass Relay IR"""
    def __init__(self, txt_file, save_name='example') -> None:
        super(VisuGraphCPC2D, self).__init__(txt_file, save_name)
        self.op_args_map = dict()
        self.save_name = 'output/visu_{}_relay_ir_pass'.format(save_name)

    def parse_node(self):
        pattern1 = re.compile(r'(%[a-z]*(\d*\.?_?[a-z]*\d*)*|meta\[relay\.Constant]\[\d*])')
        pattern2 = re.compile(r'(%\d+\.\d+)')

        node_map = dict()

        for info in self.parse_res:
            assert len(info) == 2, 'length of info must be 2!!!'
            # if '(%' not in info[1] and '%' in info[1]:
            # if '(%' not in info[1] and '.0' in info[1]:
            if '(%' not in info[1]:
                # '%16 = %15.1;
                match_op = re.search(pattern2, info[1])
                if match_op:
                    node_map[info[0]] = info[1][:-2]
                    continue

            index = info[1].find('(')
            if index == 0:
                # %0 = (%conv1.weight, %conv1.weight, %conv1.weight)
                args_list = info[1][1:-1].split(', ')
                node_map[info[0]] = args_list
                continue
            if 'add(' in info[1] or 'multiply(' in info[1] or 'divide(' in info[1]:
                args_list = info[1][index + 1:-1].split(', ')
                args_list = [node_map.get(arg, arg) for arg in args_list]
            else:
                args_list = re.findall(pattern1, info[1])
                args_list = [node_map.get(arg[0], arg[0]) for arg in args_list]

                if isinstance(args_list[0], list):
                    # 输入参数已经是列表，说明上一个op只有参数，没有具体的运算
                    args_list = args_list[0]

            self.nodes[info[0]] = IRNode(name=info[0], label=info[1][:index], inputs=args_list,
                                         color=self.random_color(), style='filled')
            for n in args_list:
                if not self.nodes.get(n, ''):
                    self.nodes[n] = IRNode(name=n, label=n, color='white')