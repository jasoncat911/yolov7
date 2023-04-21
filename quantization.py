import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
from utils.datasets import create_dataloader
from utils.general import check_img_size,colorstr
from utils.torch_utils import intersect_dicts
from models.experimental import attempt_load
# need to create model with modified definition of layers
from models.yolo_quant import Model
from test_quant import test as test_quant
from tqdm import tqdm
import yaml
from pathlib import Path
# modified yolo.py -> yolo_quant.py, moved unsupported function outside to post processing
# modified common.py, replace Silu -> Relu/Hardswish
# modified test.py -> test_quant.py to account for moved operation 


def quantization(quant_mode,batch_size,inspect,deploy,config_file,output_dir,opt):
    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batch_size != 1):
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1
        
    # Create model via model creation, not load, since model layers are modified.
    # later intersect the state_dict with official deploy model : ckpt 
    print("debug created model  ____________________",args.weights , "\n")
    model = Model(cfg=args.model,ch=3,nc=80).to(device)
    # model.info(verbose=True)
    # load model dict
    ckpt = torch.load(args.weights[0], map_location=device)
    state_dict = ckpt['model'].float().state_dict()
    # exclude = [] # ['anchor']
    # state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    # intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(state_dict, strict=False)
    input = torch.randn([batch_size, 3, 640, 640]).to(device)
    # print(model)
    if quant_mode == 'float':

        quant_model = model
        if inspect:
            import sys
            from pytorch_nndct.apis import Inspector
            
            torch.jit.trace(quant_model,input,strict=False)
            print("jit trace test passed")
            # create inspector
            # inspector = Inspector("0x603000b16013831") # by fingerprint
            inspector = Inspector("0x101000016010407") # by fingerprint
            # inspector = Inspector("DPUCAHX8L_ISA0_SP")  # by name
            # start to inspect
            inspector.inspect(quant_model, (input,), device=device, output_dir=Path(output_dir/'quantize_result'/'inspect').as_posix(),image_format="png")
            print("inspection finished")
            sys.exit()
    else:
        ## new api
        ####################################################################################
        quantizer = torch_quantizer(
            quant_mode, model, (input), device=device, quant_config_file=config_file)
        print("debug creating quantizer  ____________________")
        quant_model = quantizer.quant_model
        #####################################################################################
    # evaluate quantized model using float model evaluation scripts( modified to accomadate change in model at model initialization) 
    output,maps,time = test_quant(data='/data/yolov7/data/coco.yaml',batch_size = batch_size,model=quant_model,device=device,opt=opt, output_dir=output_dir)
    print("print some result :",output, "\n", maps,"\n", time)
    # handle quantization result
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_xmodel(deploy_check=False)
        quantizer.export_onnx_model()
        
        
# sys.argv[1:] = ["--model","/data/yolov7/cfg/deploy/yolov7.yaml",\
#     "--weights","/data/yolov7/cfg/deploy/yolov7.pt",\
#     "--output-dir","/data/cat_yolov7/yolov7/quantize_result/inspect",\
#         "--quant_mode","calib","--batch_size","1"]

parser = argparse.ArgumentParser(sys.argv)
parser.add_argument(
    '--model',
    default="/path/to/trained_model/xxx.yaml",
    help='model definition yaml'
)
parser.add_argument(
    '--weights', nargs='+', type=str, 
    default='/vitis_ai_home/pretrained_models/yolov7.pt', 
    help='model.pt path(s)')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--inspect', 
    dest='inspect',
    action='store_true',
    help='inspect model')
parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')
parser.add_argument(
    '--single-cls', 
    action='store_true', 
    help='treat as single-class dataset')
parser.add_argument(
    '--output-dir',
    type=Path,
    default="/path/to/save/dir",
    help='path to save directory'
)

args, _ = parser.parse_known_args()
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    # calibration or evaluation
    quantization(\
        batch_size=args.batch_size,\
            quant_mode=args.quant_mode,\
                inspect=args.inspect,\
                    deploy=args.deploy,\
                        config_file=args.config_file,\
                            output_dir=args.output_dir,\
                                opt = args)