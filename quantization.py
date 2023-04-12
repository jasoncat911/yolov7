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
# need to create model with modified definition of layers
from models.experimental import attempt_load
from models.yolo_quant import Model
# from models.yolo import Model as Model_float
from test_quant import test as test_quant
# from test_float import test as test_float
from tqdm import tqdm
import yaml
# modified yolo.py -> yolo_quant.py, moved unsupported function outside to post processing
# modified common.py, replace Silu -> leakyRelu

def evaluate(coco_data_yaml,subset_len,sample_method,batch_size,quant_model):
    output,maps,time = test_quant(data=coco_data_yaml,opt=args,subset_len=subset_len,sample_method = sample_method,batch_size=batch_size,model=quant_model,device=device)
    return output,maps,time

def quantization(title='optimize',
                 model_name='',
                 batch_size = 32):
    quant_mode = args.quant_mode
    finetune = args.fast_finetune
    deploy = args.deploy
    batch_size = batch_size
    subset_len = args.subset_len
    inspect = args.inspect
    config_file = args.config_file
    
    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batch_size != 1 or subset_len != 1):
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1
        subset_len = 1
        
    # Create model via model creation, not load, since model layers are modified.
    # later intersect the state_dict with official deploy model : ckpt 
    print("debug created model  ____________________",args.weights , "\n")
    model = Model(cfg=args.model,ch=3,nc=80).to(device)
    
    # model.to(device)
    # print(model)
    # model.info(verbose=True)
    # print('debug  #1 quantization model initialization ')
    # model.float().fuse().eval().to(device)
    # model.float().fuse().eval()
    # print(model.model[0].conv.bias)
    # print("debug quantization model[105].m[0].bias :",model.model[105].m[0].bias)
    # load model dict
    ckpt = torch.load(args.weights[0], map_location=device)
    state_dict = ckpt['model'].float().state_dict()
    # exclude = [] # ['anchor']
    # state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    # intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(state_dict, strict=False)
    # model.float().fuse()
    # model.names = ckpt.names

    # print('printout \n',ckpt,'end printout\n')
    # model.nc = ckpt.nc
    # torch.save(model,"quantize_result/inspect/yolo_for_quant.pt")
    input = torch.randn([batch_size, 3, 640, 640]).to(device)
    print(model)
    if 1:
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
                inspector.inspect(quant_model, (input,), device=device, output_dir="/data/cat_yolov7/yolov7/quantize_result/inspect",image_format="png")
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
        output,maps,time = evaluate(coco_data_yaml='/data/yolov7/data/coco.yaml',subset_len=subset_len,sample_method='random',batch_size = batch_size,quant_model=quant_model)
        print("print some result :",output, "\n", maps,"\n", time)
        # handle quantization result
        if quant_mode == 'calib':
            quantizer.export_quant_config()
        if deploy:
            quantizer.export_xmodel(deploy_check=False)
            quantizer.export_onnx_model()
        
        
sys.argv[1:] = ["--model","/data/yolov7/cfg/deploy/yolov7.yaml",\
    "--weights","/data/yolov7/cfg/deploy/yolov7.pt",\
    "--output_name","quantization_test_output",\
        "--quant_mode","float","--batch_size","1",\
            "--subset_len","100","--inspect"]

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
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
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

args, _ = parser.parse_known_args()
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    model_name = 'yolov7'

    feature_test = ' float model evaluation'
    if args.quant_mode != 'float':
        feature_test = ' quantization'
        # force to merge BN with CONV for better quantization accuracy
        args.optimize = 1
        feature_test += ' with optimization'
    else:
        feature_test = ' float model evaluation'
    title = model_name + feature_test

    print("-------- Start {} test ".format(model_name))

    # calibration or evaluation
    quantization(
        title=title,
        model_name=model_name,
        batch_size=args.batch_size)

    print("-------- End of {} test ".format(model_name))