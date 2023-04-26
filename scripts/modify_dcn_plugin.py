#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
import onnx_graphsurgeon as gs
import argparse
import onnx
import json

def process_graph(graph):
    dcn_nodes = [node for node in graph.nodes if node.op == "PythonOp"]
    for node in dcn_nodes:
        weight_shape = node.inputs[3].shape
        node.op = "DCNv2_TRT"
        attrs = {"dilation": [1, 1], "padding": [1, 1], "stride": [1, 1], "deformable_groups": 1, "oc": weight_shape[0]}
        node.attrs.update(attrs)
        del node.attrs["module"]
        del node.attrs["inplace"]
    return graph


def main():
    parser = argparse.ArgumentParser(description="Modify DCNv2 plugin node into ONNX model")
    parser.add_argument("-i", "--input",
            help="Path to ONNX model with 'Plugin' node to replace with DCNv2_TRT",
            default="models/centertrack_DCNv2_named.onnx")
    parser.add_argument("-o", "--output",
            help="Path to output ONNX model with 'DCNv2_TRT' node",
            default="models/modified.onnx")

    args, _ = parser.parse_known_args()
    graph = gs.import_onnx(onnx.load(args.input))
    graph = process_graph(graph)
    onnx.save(gs.export_onnx(graph), args.output)

if __name__ == '__main__':
    main()
