# -*- coding:utf-8 -*-
# Author:   liyanpeng
# Email:    youran.xia@foxmail.com
# Datetime: 2022/9/19 17:07
# Filename: visu_tvm.py
import re
import random


__all__ = ["VisuGraph", "VisuGraphFuseOps", "VisuGraphRUF", "VisuGraphMC"]


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
    def __init__(self, name='None', label='None', color='', style='', inputs=None, fixedsize='false') -> None:
        if inputs is None:
            inputs = []
        self.name = name
        self.label = label
        self.color = color
        self.style = style
        self.inputs = inputs
        self.fixedsize = fixedsize
        self.height = '0.1'


class IREdge(object):
    def __init__(self, tail_name, head_name, shape='', dtype='') -> None:
        self.tail_name = tail_name
        self.head_name = head_name
        self.shape = shape
        self.dtype = dtype


class VisuGraph(object):
    """Visu TVM Relay IR"""
    def __init__(self, txt_file, save_name='example', with_info=False) -> None:
        self.graph_code = ''
        self.nodes = dict()
        self.edges = list()
        self.node_code = ''
        self.edge_code = ''
        self.parse_res = list()
        self.save_name = 'output/visu_{}_relay_ir'.format(save_name)
        self.txt_file = txt_file
        self.node_map = dict()
        self.tensor_info = dict()
        self.with_info = with_info

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
                self.parse_res.append(('}', line))

    def init_node(self):
        # graph.node(name='%input0', label='%input0', color='white', style='filled')
        # base_node = "graph.node(name='{}', label='{}', color='{}', style='{}')\n"
        for k, v in self.nodes.items():
            self.node_code += "graph.node(name='{}', label='{}', color='{}', style='{}', " \
                              "fixedsize='{}', height='{}')\n".format(k, v.label, v.color, v.style, v.fixedsize, v.height)

    def init_edge(self):
        # graph.edge(tail_name='%input0', head_name='%0', label='')
        # base_eage = "graph.edge(tail_name='{}', head_name='{}')\n"
        for edge in self.edges:
            self.edge_code += "graph.edge(tail_name='{}', head_name='{}', label='{}')\n".format(edge.tail_name,
                                                                                                edge.head_name,
                                                                                                edge.shape)

    def codegen(self):
        from graphviz import Digraph

        if '_bp' not in self.txt_file and self.with_info:
            self.get_tensor_info(self.txt_file)
        self.parse_txt(self.txt_file)
        self.parse_node()
        self.parse_edge()
        self.init_node()
        self.init_edge()

        graph = Digraph(name='RelayIR')
        exec(self.node_code)
        exec(self.edge_code)
        graph.render(filename=self.save_name, format='png', cleanup=True)

    def get_node_args(self, output_node, body_node):
        pattern1 = re.compile(r'(%[a-zA-Z]*([\d._a-z]*\d*)*|meta\[relay\.Constant]\[\d*])')
        pattern2 = re.compile(r'(%[a-z]?\d+\.\d+)')

        if '(%' not in body_node:
            # '%16 = %15.0;
            # '%29 = %p052.0'
            # 'ones(shape=[3, 4, 5], dtype="int32")'
            match_op = re.search(pattern2, body_node)
            if match_op:
                self.node_map[output_node] = body_node[:-2]
                return None, None

        index = body_node.find('(')
        if index == 0:
            # %0 = (%conv1.weight, %conv1.weight, %conv1.weight)
            args_list = body_node[1:-1].split(', ')
            self.node_map[output_node] = args_list
            return None, None

        if 'add(' in body_node or 'multiply(' in body_node or 'divide(' in body_node:
            # add multiply 两数之间的运算
            args_list = body_node[index + 1:-1].split(', ')[:2]
            args_list = [self.node_map.get(arg, arg) for arg in args_list]
        else:
            args_list = re.findall(pattern1, body_node)
            args_list = [self.node_map.get(arg[0], arg[0]) if arg[0] else arg[1] for arg in args_list]

            if args_list and isinstance(args_list[0], list):
                # 输入参数已经是列表，说明上一个op只有参数，没有具体的运算
                args_list = args_list[0]

        return args_list, index

    def parse_node(self):
        self.node_map = dict()

        for info in self.parse_res:
            assert len(info) == 2, 'length of info must be 2!!!'
            args_list, index = self.get_node_args(info[0], info[1])
            if not args_list and not index:
                continue

            self.nodes[info[0]] = IRNode(name=info[0], label=info[1][:index], inputs=args_list)
            for n in args_list:
                if not self.nodes.get(n, ''):
                    self.nodes[n] = IRNode(name=n, label=n, color='white')

        # 在图的末尾添加一个空节点
        # TODO Multi-output bug
        # a short term fix
        if self.nodes.get('}', ''):
            self.nodes[''] = IRNode(name='', label='', color='white', fixedsize='true')

    def parse_edge(self):
        for k, v in self.nodes.items():
            if len(v.inputs) > 0:
                for n in v.inputs:
                    info = self.tensor_info.get(n, '')
                    if info:
                        info = info['shape']
                    self.edges.append(IREdge(tail_name=n, head_name=k, shape=info))
            # elif not k and len(v.inputs) > 0:
            #     for n in v.inputs:
            #         self.edges.append(IREdge(tail_name=n, head_name=v.label))

        # 添加一条指向空节点的边
        info = self.tensor_info.get('}', '')
        if info:
            info = info['shape']
        is_end = self.nodes.get('}', '')
        if is_end:
            self.edges.append(IREdge(tail_name='}', head_name='', shape=info))

    def get_tensor_info(self, txt_file=''):
        # only ap_file
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        pattern1 = re.compile(r'(%[a-z\d_.]*):.+?\[(.+?)]')
        pattern2 = re.compile(r'->.+?\[(.+?)]')
        pattern3 = re.compile(r'\)\s.+?\[(.+?)]')
        pattern4 = re.compile(r'(meta\[relay\.Constant]\[\d*]).+?\[(.+?)]')

        self.tensor_info = dict()
        flag_None = ''
        for tmp_str in lines:
            if tmp_str[:2] == 'fn':  # match head
                match_res = re.findall(pattern1, tmp_str)
                for res in match_res:
                    info = res[1].split(', ')
                    self.tensor_info[res[0]] = {
                        'shape': ', '.join(info[:-1]),
                        'dtype': info[-1]
                    }
            elif tmp_str[0] == '}':  # match tail
                match_res = re.findall(pattern2, tmp_str)
                info = match_res[0].split(', ')
                self.tensor_info['}'] = {
                    'shape': ', '.join(info[:-1]),
                    'dtype': info[-1]
                }
            else:  # match body
                if ' = ' in tmp_str and 'fn' not in tmp_str:
                    split_node = tmp_str.split(' = ')
                    if split_node[-1][0] == '(':
                        continue
                    match_res = re.findall(pattern3, split_node[-1])
                    if not match_res:
                        continue
                    info = match_res[-1].split(', ')
                    self.tensor_info[split_node[0].strip()] = {
                        'shape': ', '.join(info[:-1]),
                        'dtype': info[-1]
                    }
                    # 提取常量meta[relay.Constant][0]
                    match_res = re.findall(pattern4, split_node[-1])
                    if not match_res:
                        continue
                    for res in match_res:
                        info = res[-1].split(', ')
                        self.tensor_info[res[0].strip()] = {
                            'shape': ', '.join(info[:-1]),
                            'dtype': info[-1]
                        }
                elif ' = ' in tmp_str and 'fn' in tmp_str:
                    split_node = tmp_str.split(' = ')
                    self.tensor_info[split_node[0].strip()] = {
                        'shape': None,
                        'dtype': None
                    }
                    flag_None = split_node[0].strip()
                elif tmp_str.strip()[0] == '(':
                    continue
                else:
                    match_res = re.findall(pattern3, tmp_str)
                    info = match_res[0].split(', ')
                    self.tensor_info[flag_None if flag_None else ''] = {
                        'shape': ', '.join(info[:-1]),
                        'dtype': info[-1]
                    }
                    flag_None = ''
                    # 提取常量meta[relay.Constant][0]
                    match_res = re.findall(pattern4, tmp_str)
                    if not match_res:
                        continue
                    for res in match_res:
                        info = res[-1].split(', ')
                        self.tensor_info[res[0].strip()] = {
                            'shape': ', '.join(info[:-1]),
                            'dtype': info[-1]
                        }


class VisuGraphFuseOps(VisuGraph):
    """Visu FuseOps /
            MergeComposite Pass Relay IR"""
    def __init__(self, txt_file, save_name='example', with_info=False) -> None:
        super(VisuGraphFuseOps, self).__init__(txt_file, save_name, with_info)
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
        self.node_map = dict()
        pnodes = self.split_fn_op()

        # 将op进行细化
        for k, v in pnodes.items():
            pnode_info = self.get_pnode_info(v, pnodes)
            if not pnode_info[0]:
                continue
            else:
                ops, fn_args, color, op_args = pnode_info

            # 做实参与形参映射
            assert len(fn_args) == len(op_args), "The number of elements must be the same!"
            for i, args in enumerate(fn_args):
                if args == op_args[i]:
                    continue
                self.op_args_map[args] = self.node_map.get(op_args[i], op_args[i])

            # FuseOps --> 分析fn
            ops_list = ops.split(';')
            for ops_ in ops_list:
                if ' = ' in ops_:
                    # fn中含有多个op
                    match_op = ops_.split(' = ')
                    assert len(match_op) == 2, 'length of info must be 2!!!'
                    args_list, index = self.get_node_args(match_op[0], match_op[1])
                    if not args_list and not index:
                        continue
                    args_list = [self.op_args_map.get(arg, arg) for arg in args_list]

                    self.nodes[match_op[0]] = IRNode(name=match_op[0], label=match_op[-1][:index], inputs=args_list,
                                                     color=color, style='filled')
                    for n in args_list:
                        if not self.nodes.get(n, ''):
                            self.nodes[n] = IRNode(name=n, label=n, color='white')
                else:
                    # op中不含=
                    args_list, index = self.get_node_args('', ops_)
                    if not args_list and not index:
                        continue
                    if index == -1:
                        # body中不含()
                        args_list = fn_args
                    args_list = [self.op_args_map.get(arg, arg) for arg in args_list]

                    self.nodes[k] = IRNode(name=k, label=ops_[:index] if index != -1 else ops_, inputs=args_list,
                                           color=color, style='filled')
                    for n in args_list:
                        if not self.nodes.get(n, ''):
                            self.nodes[n] = IRNode(name=n, label=n, color='white')

        # 在图的末尾添加一个空节点
        if self.nodes.get('}', ''):
            self.nodes[''] = IRNode(name='', label='', color='white', fixedsize='true')

    def split_fn_op(self):
        pattern1 = re.compile(r'(%\d+).+{(.+)}')
        pattern1_ = re.compile(r'(%[a-z]*\d+):|(%[a-zA-Z_]*\d+_\d+):')
        pattern2 = re.compile(r'(%\d+).+?(%\d+)\((.+)\)|(%\d+).+?(\s[a-z]+)\((.+)\)')
        pattern3 = re.compile(r'([%\da-z]+)\((.+)\)')

        # 对解析的结果进一步划分成fn和op
        pnodes = dict()
        for fn_str in self.parse_res:
            if ' fn ' in fn_str:
                match_op = re.search(pattern1, fn_str).groups(0)
                args_list = re.findall(pattern1_, fn_str)  # fn的输入参数
                args_list = [self.node_map.get(arg[0], arg[0]) if arg[0] else arg[1] for arg in args_list]
                pnodes[match_op[0]] = PNode(name=match_op[0], type='fn', inputs=args_list, body=match_op[-1])
            elif ' = ' in fn_str:
                if '(' not in fn_str:
                    match_op = fn_str.split(' = ')
                    self.node_map[match_op[0]] = match_op[-1][:-3]
                    continue
                match_op = re.search(pattern2, fn_str).groups(0)
                # '%2 = subtract(%a, %b);'
                match_op = [arg for arg in match_op if arg]
                args = match_op[-1].split(', ')  # op的输入参数
                pnodes[match_op[0]] = PNode(name=match_op[0], type='op', inputs=args, body=match_op[1])
            else:
                match_op = re.search(pattern3, fn_str).groups(0)
                # '%3(%2, %b)'
                args = match_op[-1].split(', ')
                pnodes['}'] = PNode(name='', type='op', inputs=args, body=match_op[0])

        return pnodes

    def get_pnode_info(self, v, pnodes):
        if v.type == 'fn':
            v.color = self.random_color()
            return (None, )

        pre_info = pnodes.get(v.body, '')
        if not pre_info:
            ops = v.body
            fn_args = v.inputs
            color = self.random_color()
        else:
            ops = pre_info.body
            fn_args = pre_info.inputs
            color = pre_info.color
        op_args = v.inputs

        return ops, fn_args, color, op_args


class VisuGraphRUF(VisuGraph):
    """Visu RemoveUnusedFunctions /
            ToBasicBlockNormalForm /
            EliminateCommonSubexpr /
            FoldConstant /
            SimplifyInference /
            FastMath /
            SimplifyExpr /
            FlattenAtrousConv /
            CanonicalizeCast /
            ConvertLayout /
            CombineParallelConv2D /
            CombineParallelDense /
            CombineParallelBatchMatmul Pass Relay IR"""
    def __init__(self, txt_file, save_name='example', with_info=False) -> None:
        super(VisuGraphRUF, self).__init__(txt_file, save_name, with_info)
        self.save_name = 'output/visu_{}_relay_ir_pass'.format(save_name)

    def parse_node(self):
        self.node_map = dict()

        for info in self.parse_res:
            assert len(info) == 2, 'length of info must be 2!!!'
            args_list, index = self.get_node_args(info[0], info[1])
            if not args_list and not index:
                continue

            self.nodes[info[0]] = IRNode(name=info[0], label=info[1][:index], inputs=args_list,
                                         color=self.random_color(), style='filled')
            for n in args_list:
                if not self.nodes.get(n, ''):
                    self.nodes[n] = IRNode(name=n, label=n, color='white')

        # 在图的末尾添加一个空节点
        if self.nodes.get('}', ''):
            self.nodes[''] = IRNode(name='', label='', color='white', fixedsize='true')


class VisuGraphMC(VisuGraphFuseOps):
    """Visu MergeComposite Pass Relay IR"""
    # 显示算子融合后的名称
    def __init__(self, txt_file, save_name='example', with_info=False) -> None:
        super(VisuGraphMC, self).__init__(txt_file, save_name, with_info)
        self.save_name = 'output/visu_{}_relay_ir_pass'.format(save_name)

    def parse_txt(self, txt_file=''):
        assert txt_file, "You must have a txt file!!!"
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        pattern1 = re.compile(r'\s/\*.+?\*/')
        pattern2 = re.compile(r'(%\d+) = (.+)')
        pattern3 = re.compile(r'Com.+?"([a-z\d_]+)"')
        skip_flag = False
        for line in lines[1:-1]:
            if '}' in line:
                skip_flag = False
                continue
            elif 'fn' in line and 'Com' in line:
                skip_flag = True
                composite = re.search(pattern3, line).groups(0)
                line = line.replace('fn', 'fn_' + composite[0])
            elif skip_flag:
                continue
            line = re.sub(pattern1, '', line)
            line = line.strip().strip(';')
            match_op = re.findall(pattern2, line)
            if match_op:
                self.parse_res.extend(match_op)
            else:
                self.parse_res.append(('', line))

    def split_fn_op(self):
        pattern1 = re.compile(r'(%[a-z]*\d+):|(%[a-zA-Z_]*\d+_\d+):')
        pattern2 = re.compile(r'([%\da-z]+)\((.+)\)')

        pnodes = dict()
        for output_str, body_str in self.parse_res:
            if 'fn_' in body_str:
                args_list = re.findall(pattern1, body_str)  # fn的输入参数
                args_list = [self.node_map.get(arg[0], arg[0]) if arg[0] else arg[1] for arg in args_list]
                index = body_str.find(' ')
                pnodes[output_str] = PNode(name=output_str, type='fn', inputs=args_list, body=body_str[3:index])
            else:
                match_op = re.search(pattern2, body_str).groups(0)
                # '%2 = subtract(%a, %b);'
                args = match_op[-1].split(', ')  # op的输入参数
                pnodes[output_str if output_str else '}'] = PNode(name=output_str if output_str else '}', type='op',
                                                                  inputs=args, body=match_op[0])

        return pnodes