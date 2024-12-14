import argparse
import torch
from h36m.dataloader import H36motion3D
from h36m.model_t import SHAF
import os
from torch import nn, optim
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import random


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--past_length', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--future_length', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=-1, metavar='S',
                    help='random seed (default: -1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--epoch_decay', type=int, default=2, metavar='N',
                    help='number of epochs for the lr decay')
parser.add_argument('--lr_gamma', type=float, default=0.8, metavar='N',
                    help='the lr decay ratio')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='learning rate')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--channels', type=int, default=72, metavar='N',
                    help='number of channels')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--div', type=float, default=1, metavar='N',
                    help='timing experiment')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--model_save_dir', type=str, default='h36m/saved_models',
                    help='Path to save models')
parser.add_argument('--scale', type=float, default=100, metavar='N',
                    help='data scale')
parser.add_argument('--model_name', type=str, default='ckpt_short',
                    help='Name of the model.')
parser.add_argument("--weighted_loss",action='store_true')
parser.add_argument("--apply_decay",default=True,action='store_true')
parser.add_argument("--debug",action='store_true')
parser.add_argument("--add_agent_token",action='store_true')
parser.add_argument('--category_num', type=int, default=4)
parser.add_argument("--test",action='store_true')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#CUDA_VISIBLE_DEVICES={GPU_ID} python main_h36m.py --past_length 10 --future_length 10 --channel 72

time_exp_dic = {'time': 0, 'counter': 0}

args = parser.parse_args()
args.cuda = True
args.add_agent_token = True
if args.future_length == 25:
    args.weighted_loss = True

device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()#均方损失函数

print("args:",args)
try:        #try 有错就跳出，生成SHAF-main/n_body_system/logs
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name) #生成SHAF-main/n_body_system/logs/exp_1
except OSError:
    pass

def setup_seed(seed):#生成随机数种子，让每次生成的随机数相同
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def lr_decay(optimizer, lr_now, gamma):
    lr_new = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new

def main():
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)#如果args中传入了有效seed，调用随机生成种子
    else:
        seed = random.randint(0,1000)#如果如果没有，用随机数随机生成种子
        setup_seed(seed)
    print('The seed is :',seed)

    past_length = args.past_length
    future_length = args.future_length

    if args.debug:
        dataset_train = H36motion3D(actions='walking', input_n=args.past_length, output_n=args.future_length, split=0, scale=args.scale)
    else:
        dataset_train = H36motion3D(actions='all', input_n=args.past_length, output_n=args.future_length, split=0, scale=args.scale)
   
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, #训练数据
                                               num_workers=0)#源代码初始为num_workers=8

    acts = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]
    loaders_test = {}
    for act in acts:
        dataset_test = H36motion3D(actions=act, input_n=args.past_length, output_n=args.future_length, split=1, scale=args.scale)
        loaders_test[act] = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                num_workers=0)#源代码初始为num_workers=8   #测试数据
    #Data.DataLoader(dataset=torch_dataset,  # torch TensorDataset format
        # batch_size=BATCH_SIZE,  # mini batch size
        #shuffle=True,  # 要不要打乱数据 (打乱比较好)
        #num_workers=2,  # 多线程来读数据)


    dim_used = dataset_train.dim_used

    model = SHAF(in_node_nf=args.past_length, in_edge_nf=2, hidden_nf=args.nf, in_channel=args.past_length, hid_channel=args.channels, out_channel=args.future_length,device=device, n_layers=args.n_layers, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh,add_agent_token=args.add_agent_token,category_num=args.category_num)    
    #输入in_node_nf=10，in_edge_nf=2，hidden_nf=64，in_channel=10，hid_channel=72，out_channel=10，device=torch.device("cuda")，
    #   n_layers=4, recurrent=True, norm_diff=False, tanh=False,add_agent_token=Ture,category_num=4
    def get_parameter_number(model):#统计模型参数量
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print(get_parameter_number(model))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=0, last_epoch=-1)

    if args.test:
        model_path = args.model_save_dir + '/' + args.model_name +'.pth.tar'
        print('Loading model from:', model_path)
        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt['state_dict'], strict=False)
        if args.future_length == 25:
            avg_mpjpe = np.zeros((6))
        elif args.future_length == 15:
            avg_mpjpe = np.zeros((2))
        else:
            avg_mpjpe = np.zeros((4))

        for act in acts:
            mpjpe = test(model, optimizer, 0, (act, loaders_test[act]), dim_used, backprop=False)
            avg_mpjpe += mpjpe
        avg_mpjpe = avg_mpjpe / len(acts)
        print('avg mpjpe:',avg_mpjpe)
        return

    results = {'epochs': [], 'losess': []}
    best_test_loss = 1e8
    best_ade = 1e8
    best_epoch = 0
    lr_now = args.lr
    lr = []
    for epoch in range(0, 160):#args.epochs=80
        if args.apply_decay:
            if epoch % 2 == 0 and epoch > 0:
                lr_now = lr_decay(optimizer, lr_now, args.lr_gamma)
        print(lr_now)

        train(model, optimizer, epoch,lr_now, loader_train, dim_used)
        #model=SHAF(in_node_nf=args.past_length, in_edge_nf=2, hidden_nf=args.nf, in_channel=args.past_length, hid_channel=args.channels, out_channel=args.future_length,device=device, n_layers=args.n_layers, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh,add_agent_token=args.add_agent_token,category_num=args.category_num)
        #optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
        #dim_used为all_seqs.shape[1]即列，中不包括dim_ignore号的列表

        if epoch % args.test_interval == 0: #args.test_interval=1
            if args.future_length == 25:
                avg_mpjpe = np.zeros((6))
            elif args.future_length == 15:
                avg_mpjpe = np.zeros((2))
            else:
                avg_mpjpe = np.zeros((4))
            for act in acts:
                mpjpe = test(model, optimizer, epoch, (act, loaders_test[act]), dim_used, backprop=False)
                avg_mpjpe += mpjpe
            avg_mpjpe = avg_mpjpe / len(acts)
            print('avg mpjpe:',avg_mpjpe)
            avg_mpjpe = np.mean(avg_mpjpe)

            if avg_mpjpe < best_test_loss:
                best_test_loss = avg_mpjpe
                best_epoch = epoch
                state = {'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()}
                if args.future_length == 25:
                    file_path = os.path.join(args.model_save_dir, 'ckpt_long_best.pth.tar')
                else:
                    file_path = os.path.join(args.model_save_dir, 'ckpt_best.pth.tar')
                torch.save(state, file_path)

            state = {'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()}
            if args.future_length == 25:
                file_path = os.path.join(args.model_save_dir, 'ckpt_long_'+str(epoch)+'.pth.tar')
            else:
                file_path = os.path.join(args.model_save_dir, 'ckpt_'+str(epoch)+'.pth.tar')
            torch.save(state, file_path)

            print("Best Test Loss: %.5f \t Best epoch %d" % (best_test_loss, best_epoch))
            print('The seed is :',seed)

    return

from torch.nn import functional as F
class AveargeJoint2(nn.Module):

    def __init__(self):
        super().__init__()
        self.torso = [8, 9, 10, 11]
        self.left_leg = [0, 1, 2, 3]
        self.right_leg = [4, 5, 6, 7]
        self.left_arm = [12, 13, 14, 15,16]
        self.right_arm = [17, 18, 19,20,21]

    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 2))  # [N, C, T, V=1]
        x_leftleg = F.avg_pool2d(x[:, :, :, self.left_leg], kernel_size=(1, 2))  # [N, C, T, V=1]
        x_rightleg = F.avg_pool2d(x[:, :, :, self.right_leg], kernel_size=(1, 2))  # [N, C, T, V=1]
        x_leftarm = F.avg_pool2d(x[:, :, :, self.left_arm], kernel_size=(1, 2))  # [N, C, T, V=1]
        x_rightarm = F.avg_pool2d(x[:, :, :, self.right_arm], kernel_size=(1, 2))  # [N, C, T, V=1]
        x_body = torch.cat((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm),
                           dim=-1)  # [N, C, T, V=1]), dim=-1)        # [N, C, T, 10]
        return x_body

def smooth(src, sample_len):
    """
    data:[100,22,10,3]
    """

    smooth_data = src.clone()
    for i in range(sample_len):
        smooth_data[:, :, i] = torch.mean(src[:, :, :i + 1], dim=2)  # 取平滑值，对本帧之后的所有帧取平均值作为本帧值
    return smooth_data
# model=SHAF(in_node_nf=args.past_length, in_edge_nf=2, hidden_nf=args.nf, in_channel=args.past_length, hid_channel=args.channels, out_channel=args.future_length,device=device, n_layers=args.n_layers, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh,add_agent_token=args.add_agent_token,category_num=args.category_num)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
# dim_used为all_seqs.shape[1]即列，中不包括dim_ignore号的列表
#python main_h36m.py --past_length 10 --future_length 10 --channel 72
def train(model, optimizer, epoch,lr_now, loader, dim_used=[], backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0,'loss_0': 0,'loss_10_pre':0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, length, _ = data[0].size() #batch_size=100,n_nodes=22,length=10,3 因为忽略了十个关节
        data = [d.to(device) for d in data]
        loc, vel, loc_end, _, item = data #loc为(100,22, 10, 3)22个关节10帧三维坐标，vel为100组22个关节10帧两针间坐标差，loc_end100组22个关节10到20帧三维坐标
        loc_start = loc[:,:,-1:]

        optimizer.zero_grad()
        nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()#nodes的torch.Size([100, 22, 10])求了坐标差的平方和开根号


        loc_pred_0,loc_pred_1,loc_pred_2 = model(nodes, loc.detach(),loc_end.detach(), vel)#nodes维度为[100, 22, 10]求了三个坐标的平方和开根号，loc为(100，22, 10, 3)100组22个关节10帧三维坐标，vel为100组22个关节10帧两针间坐标差
        #输出为loc_pred=100*22*10*3是预测的坐标

        get_mid = AveargeJoint2()
        loc_end_0 = get_mid(loc_end.transpose(1, 3))
        loc_end_0 = loc_end_0.transpose(1, 3)  # 100*10*10*3
        loc_end_0 = smooth(loc_end_0, 25)

        loc_0 = get_mid(loc.transpose(1, 3))
        loc_0 = loc_0.transpose(1, 3)  # 100*10*10*3
        loc_end_0=torch.cat([loc_0,loc_end_0],dim=-2)

        loc_end_1 = smooth(loc_end, 25)
        loc_end_1 = torch.cat([loc, loc_end_1], dim=-2)

        loc_end=torch.cat([loc,loc_end],dim=-2)

        if args.weighted_loss:
            # weight = np.arange(1,5,(4/args.future_length))
            # weight = args.future_length / weight
            # # weight = weight / np.sum(weight)
            # weight = torch.from_numpy(weight).type_as(loc_end)
            # weight = weight[None,None]
            # loss = torch.mean(weight*torch.norm(loc_pred_2-loc_end,dim=-1))
            loss_0 = torch.mean(torch.norm(loc_pred_0 - loc_end_0, dim=-1))
            loss_1 = torch.mean(torch.norm(loc_pred_1 - loc_end_1, dim=-1))
            loss_2 = torch.mean(torch.norm(loc_pred_2 - loc_end, dim=-1))
            loss_10_pre = torch.mean(torch.norm(loc_pred_2[:, :, 10:35] - loc_end[:, :, 10:35], dim=-1))
            loss_all = loss_0 + loss_1 + loss_2
        else:
            loss_0 = torch.mean(torch.norm(loc_pred_0-loc_end_0,dim=-1))
            loss_1 = torch.mean(torch.norm(loc_pred_1 - loc_end_1, dim=-1))
            loss_2 = torch.mean(torch.norm(loc_pred_2-loc_end, dim=-1))
            loss_10_pre=torch.mean(torch.norm(loc_pred_2[:,:,10:35]-loc_end[:,:,10:35], dim=-1))
            loss_all=loss_0+loss_1+loss_2
        if backprop:
            if (loss_2 < 50 and loss_2 >0):
                loss_all.backward()
            # else:
            #     #print(loss_1)
            #     lr_now = lr_decay(optimizer, lr_now, args.lr_gamma)
            optimizer.step()
        res['loss_0']+= loss_0.item()*batch_size
        res['loss'] += loss_2.item()*batch_size
        res['loss_10_pre'] += loss_10_pre.item() * batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print("loss_0",res['loss_0'] / res['counter'])
    print('%s epoch %d avg loss: %.5f' % (prefix+'train', epoch, res['loss'] / res['counter']))
    print("loss_10_pre", res['loss_10_pre'] / res['counter'])

    return res['loss'] / res['counter']

#python main_h36m.py --past_length 10 --future_length 10 --channel 72 --model_name ckpt_best --test
def test(model, optimizer, epoch, act_loader,dim_used=[],backprop=False):

    act, loader = act_loader[0], act_loader[1]

    model.eval()

    validate_reasoning = False
    if validate_reasoning:
        acc_list = [0]*args.n_layers
    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'ade': 0}

    output_n = args.future_length
    if output_n == 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 15:
        eval_frame = [3, 14]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
    t_3d = np.zeros(len(eval_frame))

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            batch_size, n_nodes, length, _ = data[0].size()
            data = [d.to(device) for d in data]
            loc, vel, loc_end, loc_end_ori,_ = data
            loc_start = loc[:,:,-1:]
            pred_length = loc_end.shape[2]

            optimizer.zero_grad()

            nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
            loc_pred_0,loc_pred_1, loc_pred = model(nodes, loc.detach(),loc_end.detach(), vel)

            loc_pred=loc_pred[:,:,10:35]



            pred_3d = loc_end_ori.clone()
            loc_pred = loc_pred.transpose(1,2)
            loc_pred = loc_pred.contiguous().view(batch_size,loc_end.shape[2],n_nodes*3)
            
            joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
            index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
            joint_equal = np.array([13, 19, 22, 13, 27, 30])
            index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

            pred_3d[:,:,dim_used] = loc_pred
            pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
            pred_p3d = pred_3d.contiguous().view(batch_size, pred_length, -1, 3)#[:, input_n:, :, :]
            targ_p3d = loc_end_ori.contiguous().view(batch_size, pred_length, -1, 3)#[:, input_n:, :, :]

            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                t_3d[k] += torch.mean(torch.norm(targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2, 1)).item() * batch_size
            
            res['counter'] += batch_size
    t_3d *= args.scale
    N = res['counter']
    actname = "{0: <14} |".format(act)
    if args.future_length == 25:
        print('Act: {},  ErrT: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}, TestError {:.4f}'\
            .format(actname, 
                    float(t_3d[0])/N, float(t_3d[1])/N, float(t_3d[2])/N, float(t_3d[3])/N, float(t_3d[4])/N, float(t_3d[5])/N, 
                    float(t_3d.mean())/N))
    elif args.future_length == 15:
        print('Act: {},  ErrT: {:.3f} {:.3f}, TestError {:.4f}'\
            .format(actname, 
                    float(t_3d[0])/N, float(t_3d[1])/N, 
                    float(t_3d.mean())/N))
    else:
        print('Act: {},  ErrT: {:.3f} {:.3f} {:.3f} {:.3f}, TestError {:.4f}'\
            .format(actname, 
                    float(t_3d[0])/N, float(t_3d[1])/N, float(t_3d[2])/N, float(t_3d[3])/N, 
                    float(t_3d.mean())/N))

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    # print('%s epoch %d avg loss: %.5f ade: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter'], res['ade'] / res['counter']))
    return t_3d / N

if __name__ == "__main__":
    main()




