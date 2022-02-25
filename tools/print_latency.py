import argparse
import torch
import os
import math
import numpy as np
import shutil
from torch.autograd import Variable
import time
from tqdm import tqdm
import matplotlib
from mmcv.runner import load_checkpoint
import onnx
import onnxruntime as rt
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pdb import set_trace as bp
import warnings
from tools.genotypes import PRIMITIVES
from mmseg.models import build_segmentor
from mmcv import Config, DictAction
from mmseg.apis import init_segmentor
from thop import profile
from tools.utils_dist import create_exp_dir, plot_op, plot_path_width, objective_acc_lat
import os

try:
    from tools.utils_dist import compute_latency_ms_tensorrt as compute_latency
    print("use TensorRT for latency test")
except:
    from tools.utils_dist import compute_latency_ms_pytorch as compute_latency
    print("use PyTorch for latency test")

def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('config',help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    cfg = Config.fromfile(args.config)
    checkpoint = args.checkpoint
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    print(f'Config:\n{cfg.pretty_text}')
    # dump modecls graph
    inputSize = '960*768'
    inputDimension = (1, 3, 960, 768)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    # print(f'Model graph:\n{str(model)}')

    # model = BiSeNet(depth=18)
    print('begin')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = 12345

    print('set seed:', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print('loading parameters...')
    # checkpoint = load_checkpoint(model, checkpoint) # 实践中，导入checkpoint与否并不影响推理速度

    print('ONNX model...')

    # Test code 检测onnx转换是否成功
    # model = model.cpu()
    # model.eval()
    # dummy_input = torch.randn(1, 3, 1024, 1024, device='cpu')
    # torch.onnx.export(model, dummy_input, "deeplabv3.onnx", verbose=False, input_names=["input"], output_names=["output"],
    #                   opset_version=11)
    # #
    # print('loading ONNX model and check that the IR is well formed')
    # onnx_model = onnx.load('deeplabv3.onnx')
    # onnx.checker.check_model(onnx_model)
    # print('ONNX model graph...')
    # model_graph = onnx.helper.printable_graph(onnx_model.graph)
    #
    # print('Onnxruntime for inference...')
    # image = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
    # sess = rt.InferenceSession('model.onnx')
    # input_name1 = sess.get_inputs()[0].name
    # #
    # print('Check the performance of Pytorch and ONNX')
    # output = sess.run(None, {input_name1: image})[0]  # 1, C, 1024, 1024
    # image_tensor = torch.tensor(image)
    # output_tensor = model(image_tensor)
    # diff = torch.max(torch.abs(output_tensor - torch.tensor(output)))
    # print('different is :', diff)

    model = model.cuda()
    latency = compute_latency(model, inputDimension)

    print("{} FPS:".format(inputSize) + str(1000. / latency))

    print('Params & FLOPs...')

    # 参数和计算量
    # img_metas = None
    # gt_semantic_seg = None
    # return_loss = True
    # model = model.cpu()
    # flops, params = profile(model, inputs=(torch.randn(inputDimension), img_metas, False), verbose=False)
    # print("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))