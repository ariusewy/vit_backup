import os
import sys
import _init_paths
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import sys 
import optim
sys.path.append("../") 
from vit_pytorch_loc.vit_pytorch import ViT
from utils.utils import set_gpu, seed_all, _pil_interp, load_partial_weight
from smart_exchange import smart_net
from tqdm import tqdm
import argparse
import math
from IPython.core.debugger import Pdb
ipdb = Pdb()
import datetime
import deepshift
from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type,update_B,update_weight_tensor,update_weight_tensor_old
import torchsummary

#Result Report
from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header


parser = argparse.ArgumentParser(description='running parameters',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# general parameters for data and model
# data parameters
parser.add_argument('--data_path', default='../datasets/cifar10/', type=str, help='path to ImageNet data')
parser.add_argument('--ckpt_path_old', default='../save/save_8_2/final_epoch_15.pth', type=str, help='path to ImageNet data')
parser.add_argument('--ckpt_path', default='../save/save_8_2/final_epoch_15.pth', type=str, help='path to load checkpoint')
parser.add_argument('--workers', default=8, type=int, help='number of workers for data loader') 
parser.add_argument('--crop_pct', default=0.9, type=float, help='crop ratio')
parser.add_argument('--interpolation', default='bicubic', type=str, help='interpolation method')

# model parameters
parser.add_argument('--input_size', default=224, type=int, help='size of input')
parser.add_argument('--patch_size', default=16, type=int, help='size of patch') 
parser.add_argument('--num_classes', default=10, type=int, help='num_classes') 
parser.add_argument('--dim', default=768, type=int, help='dim') 
parser.add_argument('--depth', default=12, type=int, help='depth') 
parser.add_argument('--heads', default=12, type=int, help='heads') 
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp_dim') 
parser.add_argument('--dropout', default=0.1, type=float, help='dropout') 
parser.add_argument('--emb_dropout', default=0.1, type=float, help='emb_dropout') 
parser.add_argument('--qkv_bias', default=True, type=bool, help='use qkv_bias')

# training parameters
parser.add_argument('--max_epoch', default=128, type=int, help='max epoch')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--val_per', default=1, type=int, help='validate per epochs')
parser.add_argument('--val_begin',  action='store_true', help='validate before training')
parser.add_argument('--save_path', default='./save/cifar10_new', type=str, help='path to save checkpoints')
parser.add_argument('--save_per', default=10, type=int, help='save ckpt per epochs')
parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size for data loader')

# other parameters
parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
parser.add_argument('--gpu', default='0', type=str, help='gpu')

#Smart Exchange
parser.add_argument('--quantize', action='store_true',
                    help='quantize the network layers, activations and inputs.')
parser.add_argument('--num_bits', default=None, type=int,
                    help='the number of bits that the input and activations '
                         'will be quantized to.')
parser.add_argument('--num_bits_weight', default=None, type=int,
                    help='the number of bits that the weights will be '
                         'quantized to.')
# parser.add_argument('--decompose_iternum', default=30, type=int,
#                     help='number of maximum iterations in smart exchange.')
# parser.add_argument('--decompose_threshold', default=0.1, type=float,
#                     help='threshold that promotes sparsity in smart exchange.')#0.001
# parser.add_argument('--decompose_decay', default=0.7, type=float,
#                     help='decay rate if the threshold is too large.')
# parser.add_argument('--decompose_scale', default=True, type=bool,
#                     help='scale the Ce matrix for better quantization.')
# parser.add_argument('--decompose_tol', default=1e-10, type=float,
#                     help='tolerance of stopping condition in smart exchange.')
# parser.add_argument('--decompose_rcond', default=1e-10, type=float,
#                     help='rcond for the least squares in smart exchange')
# parser.add_argument('--init_method', default='trivial', type=str,
#                     help='initialization method in smart exchange. '
#                          'select between `trivial` | `ksvd` (not implemented)')
# parser.add_argument('--threshold_row', default=True, action='store_true',
#                     help='threshold by rows before thresholding emelentwisely.')
# parser.add_argument('--decompose_first', default=False, action='store_true',
#                     help='execute smart exchange decomposition before training.')

#deepshift
parser.add_argument('--type', default='linear',
                    choices=['linear', 'conv'],
                    help='model architecture type: ' +
                    ' | '.join(['linear', 'conv']) +
                    ' (default: linear)')
parser.add_argument('--model', default='', type=str, metavar='MODEL_PATH',
                    help='path to model file to load both its architecture and weights (default: none)')
parser.add_argument('--weights', default='', type=str, metavar='WEIGHTS_PATH',
                    help='path to file to load its weights (default: none)')
parser.add_argument('--shift-depth', type=int, default=3,
                    help='how many layers to convert to shift')
parser.add_argument('-st', '--shift-type', default='PS', choices=['Q', 'PS'],
                    help='type of DeepShift method for training and representing weights (default: PS)')
parser.add_argument('-r', '--rounding', default='deterministic', choices=['deterministic', 'stochastic'],
                    help='type of rounding (default: deterministic)')
parser.add_argument('-wb', '--weight-bits', type=int, default=5,
                    help='number of bits to represent the shift weights') 
parser.add_argument('-ab', '--activation-bits', nargs='+', default=[16,16],
                    help='number of integer and fraction bits to represent activation (fixed point format)')   
parser.add_argument('-opt', '--optimizer', metavar='OPT', default="radam", 
                    help='optimizer algorithm')
parser.add_argument('--resume', default='', type=str, metavar='CHECKPOINT_PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='only evaluate model on validation set')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--pretrained', dest='pretrained', default=False, type=lambda x:bool(distutils.util.strtobool(x)), 
                    help='use pre-trained model of full conv or fc model')
parser.add_argument('--updata_per', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')


decompose_opts = None

def main():
    args = parser.parse_args()

    global decompose_opts
    decompose_opts = dict(decompose_iternum=args.decompose_iternum,
                          decompose_threshold=args.decompose_threshold,
                          decompose_decay=args.decompose_decay,
                          decompose_scale=args.decompose_scale,
                          decompose_tol=args.decompose_tol,
                          decompose_rcond=args.decompose_rcond,
                          threshold_row=args.threshold_row,
                          init_method=args.init_method)
    
    f = open('log.txt','w+')
    print('Called With Args:')
    for k,v in sorted(vars(args).items()):
        print('    ', k,'=',v)
    print()
    
    print('batch_size =', args.batch_size , '\n'
          'shift_depth = ',args.shift_depth ,'\n'
     'shift_type =',args.shift_type,'\n',file = f)
    
    seed_all(args.seed)
    set_gpu(args.gpu)

    # build validation dataset
    data_path = args.data_path
    batch_size = args.batch_size
    workers = args.workers
    img_size = args.input_size # set img_size = input_size
    crop_pct = args.crop_pct
    interpolation = args.interpolation

    scale_size = int(math.floor(img_size / crop_pct))
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])

    train_transform = transforms.Compose([
            transforms.Resize(scale_size, _pil_interp(interpolation)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
            ])
    val_transform = transforms.Compose([
            transforms.Resize(scale_size, _pil_interp(interpolation)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            # normalize,
            ])

    train_dataset = datasets.CIFAR10(
        root=data_path, 
        train=True, 
        transform=train_transform)
    val_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )

    # ipdb.set_trace()


    # build ViT model
    input_size = args.input_size
    patch_size = args.patch_size
    num_classes = args.num_classes
    dim = args.dim 
    depth = args.depth
    heads = args.heads
    mlp_dim = args.mlp_dim
    dropout = args.dropout
    emb_dropout = args.emb_dropout
    qkv_bias = args.qkv_bias

    v = ViT(
        image_size = input_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = dropout,
        emb_dropout = emb_dropout,
        qkv_bias= qkv_bias
    )
    print('Building ViT Model:\n{}'.format(v))
    print(v.state_dict)

    # initialize save_path
    save_path = args.save_path
    if not os.path.isdir(save_path):
        print('Creating Saving Path: \'{}\''.format(save_path))
        os.makedirs(save_path)
    else:
        print('\033[1;31mWARNING: Saving Path \'{}\' Already Exist. May Cover Saved Checkpoints\033[0m'.format(save_path))

    # load weight
    ckpt_path_old = args.ckpt_path_old
    ckpt_path = args.ckpt_path
    print('Loading Weights from \'{}\''.format(ckpt_path_old))
    print()
    # weight = torch.load(ckpt_path)######################
    # load_partial_weight(v, weight)
    # v.cuda()
    # build optimizer
    max_epoch = args.max_epoch
    val_per_epoch = args.val_per
    save_per_epoch = args.save_per
    lr = args.lr

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = None 
    if(args.optimizer.lower() == "sgd"):
        optimizer = torch.optim.SGD(v.parameters(), args.lr, momentum=0.9)
    elif(args.optimizer.lower() == "adadelta"):
        optimizer = torch.optim.Adadelta(v.parameters(), args.lr)
    elif(args.optimizer.lower() == "adagrad"):
        optimizer = torch.optim.Adagrad(v.parameters(), args.lr)
    elif(args.optimizer.lower() == "adam"):
        optimizer = torch.optim.Adam(v.parameters(), args.lr)
    elif(args.optimizer.lower() == "rmsprop"):
        optimizer = torch.optim.RMSprop(v.parameters(), args.lr)
    elif(args.optimizer.lower() == "radam"):
        optimizer = optim.RAdam(v.parameters(), args.lr)
    elif(args.optimizer.lower() == "ranger"):
        optimizer = optim.Ranger(v.parameters(), args.lr)
    else:
        raise ValueError("Optimizer type: ", args.optimizer, " is not supported or known")
    
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max_epoch/4,eta_min=0.0003)

    [activation_integer_bits, activation_fraction_bits] = args.activation_bits
    [activation_integer_bits, activation_fraction_bits] = [int(activation_integer_bits), int(activation_fraction_bits)]
    weight_old = torch.load(ckpt_path_old)
    #ck_weight = torch.load(ckpt_path)
    load_partial_weight(v, weight_old)
    if args.shift_depth > 0:
        v,_ = convert_to_shift(v, args.shift_depth, args.shift_type, convert_all_linear=True, convert_weights=True, use_kernel = False, use_cuda = True, rounding = args.rounding, weight_bits = args.weight_bits, act_integer_bits = activation_integer_bits, act_fraction_bits = activation_fraction_bits)
    
    # for name,parms in v.named_parameters(): 
    #     print("======================================")
    #     print(name,":",parms)

    #Load Checkpoint:
    #load_partial_weight(v, ck_weight)
    v.cuda()
    
#Torch_Summary
    # model_summary, model_params_info = torchsummary.summary_string(v, input_size=(3,224,224))
    # print(model_summary)
    #print(v.state_dict)
    # for name,parms in v.named_parameters(): 
    #         print(name,":",parms)
    # validate before training
    #v = round_shift_weights(v)
    # val_acc_second = validate(val_loader, v, criterion,args)
    if args.val_begin:
        v = round_shift_weights(v)
        val_acc_second = validate(val_loader, v, criterion,args)
        print('Validated Accuracy: {}%'.format(val_acc_second))
        print('Validated Accuracy: {}%'.format(val_acc_second),file=f)
            
    max_acc = 0.0
    val_accs = []
    train_acc = []
    train_loss = []
    for epoch in range(1, args.max_epoch+1):
        loss,acc = train(train_loader,v,criterion,optimizer,epoch,args)
        print('Training Loss: {}, Training Acc: {}%'.format(loss,acc),file = f)
        train_loss.append(loss)
        train_acc.append(acc)
        v = round_shift_weights(v)
        val_acc = validate(val_loader, v, criterion,args)
        print('Validated Accuracy: {}%'.format(val_acc),file=f)
        val_accs.append(val_acc)
        if val_acc>max_acc:
            max_acc = val_acc
            print('Saving:')
            save_file = 'Max' + str(epoch) + 'acc_{:.2f}'.format(val_acc)+'.pth'
            save_file_path = os.path.join(args.save_path, save_file)
            torch.save(v.state_dict(), save_file_path)
            print('Checkpoint Saved to \'{}\''.format(save_file_path))
        if (epoch%2==0):
            print('Saving:')
            save_file = 'epoch_' + str(epoch) + 'acc_{:.2f}'.format(val_acc)+'.pth'
            save_file_path = os.path.join(args.save_path, save_file)
            torch.save(v.state_dict(), save_file_path)
            print('Checkpoint Saved to \'{}\''.format(save_file_path))

        print()

    # save final weight
    print('Training Finished')
    save_file = 'final_epoch_' + str(epoch) + '.pth'
    save_file_path = os.path.join(args.save_path, save_file)
    torch.save(v.state_dict(), save_file_path)
    print('Final Checkpoint Saved to \'{}\''.format(save_file_path))

    # run test on cifar10
    print()
    print('Testing Fine-tuned ViT on Cifar100 Testset')
    v.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(val_loader):
            imgs, labels = data
            imgs, labels = imgs.cuda(), labels.cuda()
            output,head_scores = v(imgs)
            _,predict_labels = torch.max(output.data,1)
            predict_labels = predict_labels.view(-1)
            correct+= torch.sum(torch.eq(predict_labels,labels)).item()
            total+=len(labels)
        print('Tested on {} Images'.format(total))
        print('Final Accuracy: %f%%'%(correct/total*100.0))
        print('Tested on {} Images'.format(total),file=f)
        print('Final Accuracy: %f%%'%(correct/total*100.0),file=f)
        
    f.seek(0,0)
    mail_content = f.read()
    f.close()
    mailreport(mail_content)
    

def mailreport(mail_content):
    host_server = 'smtp.qq.com'  #qq邮箱smtp服务器
    sender_qq = '2514218467@qq.com' #发件人邮箱
    pwd = 'xhsruapgbgfmdjii'
    receiver = ['freyaeasy@163.com']#收件人邮箱
    mail_title = 'Result' #邮件标题
    msg = MIMEMultipart()
    msg["Subject"] = Header(mail_title,'utf-8')
    msg['To'] = ";".join(receiver)
    msg.attach(MIMEText(mail_content,'plain','utf-8'))
    smtp = SMTP_SSL(host_server) # ssl登录
    smtp.login(sender_qq,pwd)
    smtp.sendmail(sender_qq,receiver,msg.as_string())
    smtp.quit()
    print("Report is sent by email\n")
    
    

def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    total_train_loss = 0.0
    total_train_acc = 0.0
    total_data_num = 0
    total_train_correct = 0
    i =0
    print('Epoch{}_start:'.format(epoch),time.asctime( time.localtime(time.time()) ))
    for data in train_loader:
        imgs, labels = data
        imgs, labels = imgs.cuda(), labels.cuda()
        output,head_scores = model(imgs,False)
        loss = criterion(output, labels)
        total_train_loss += loss * imgs.shape[0]
        total_data_num += imgs.shape[0]
        _,predict_labels = torch.max(output.data,1)
        predict_labels = predict_labels.view(-1)
        total_train_correct += torch.sum(torch.eq(predict_labels,labels)).item()
        # if i==0 or i%args.updata_per==0:
        #     weight_tensor = head_scores
        # else:
        #     weight_tensor = [i + j for i, j in zip(weight_tensor, head_scores)]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i = i+ 1
        if i%args.updata_per==0:
            #weight_tensor_n = [i/args.updata_per  for i in weight_tensor]
            update_weight_tensor(model,head_scores,'head_score')

        print('Epoch{}'.format(epoch),'[{}/{}]'.format(i,int(50000/args.batch_size)),'Loss: {}, Acc: {}%'.format(total_train_loss /total_data_num,total_train_correct / total_data_num*100))
    total_train_loss /= total_data_num
    total_train_acc = total_train_correct / total_data_num * 100
    endtime=time.time()
    print('Epoch{}_end:'.format(epoch),time.asctime( time.localtime(time.time()) ))
    print('Training Loss: {}, Training Acc: {}%'.format(total_train_loss, total_train_acc))
    return total_train_loss,total_train_acc



def validate(val_loader, model, criterion,args):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            imgs, labels = data
            imgs, labels = imgs.cuda(), labels.cuda()
            output ,head_scores= model(imgs,True)
            _,predict_labels = torch.max(output.data,1)
            predict_labels = predict_labels.view(-1)
            correct+= torch.sum(torch.eq(predict_labels,labels)).item()
            total+=len(labels)
            print('[finished:{}]'.format(total),'Acc: {}%'.format(correct/total*100.0))
        val_acc = correct/total*100.0
        print('Validated on {} Images, Accuracy: {}%'.format(total, val_acc))
    return val_acc



if __name__ == '__main__':
    main()
