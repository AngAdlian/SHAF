import time

import torch
from torch import nn
from h36m.gcl_t import Feature_learning_layer
import numpy as np
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
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 2))  
        x_leftleg = F.avg_pool2d(x[:, :, :, self.left_leg], kernel_size=(1, 2))  
        x_rightleg = F.avg_pool2d(x[:, :, :, self.right_leg], kernel_size=(1, 2))  
        x_leftarm = F.avg_pool2d(x[:, :, :, self.left_arm], kernel_size=(1, 2))  
        x_rightarm = F.avg_pool2d(x[:, :, :, self.right_arm], kernel_size=(1, 2))  
        x_body = torch.cat((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm),
                           dim=-1)  
        return x_body

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1,silu=nn.SiLU(),device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rel_channels = 9

        
        self.Linear1 = nn.Linear(self.in_channels, self.rel_channels)
        self.Linear2 = nn.Linear(self.in_channels, self.rel_channels)
        self.Linear3 = nn.Linear(self.in_channels, self.rel_channels)
        
        self.Linear4=nn.Sequential(
            nn.Linear(self.rel_channels, 18),
            silu,  
            nn.Linear(18,self.rel_channels),
            silu)
        self.Linear5 = nn.Sequential(
            nn.Linear(self.rel_channels, 6),
            silu,  
            nn.Linear(6,self.out_channels),
            silu)

        self.device = device
        self.to(self.device)

    def forward(self, x, A=None,alpha=1,beta=1):

        
        x1, x2, x3 = self.Linear1(x), self.Linear2(x), self.Linear3(x)
        
        x1=x1.transpose(1,3).mean(-2).cuda()
        x2=x2.transpose(1,3).mean(-2).cuda()
        x3 = x3.transpose(1, 3).cuda()
        x1 = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        x1 = x1.transpose(1, 3)
        x1 = self.Linear4(x1)
        x1 = x1.transpose(1, 3)
        x1 = (x1* alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)).cuda()  
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        
        x1 = x1.transpose(1, 3)
        x1 = self.Linear5(x1)

        return x1

class SHAF(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, in_channel, hid_channel, out_channel, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, recurrent=False, norm_diff=False, tanh=False,add_agent_token=False,n_agent=22,category_num=4):
        
        
        super(SHAF, self).__init__()
        self.hidden_nf = hidden_nf 
        self.device = device    
        self.n_layers = n_layers    
        self.embedding = nn.Linear(in_node_nf*2, int(self.hidden_nf/2))
        self.embedding_v = nn.Linear(in_node_nf*2, int(self.hidden_nf/2))
        self.embedding_a = nn.Linear(in_node_nf*2, int(self.hidden_nf / 2))  
        self.coord_trans = nn.Linear(in_channel*2, int(hid_channel), bias=False)
        self.vel_trans = nn.Linear(in_channel*2, int(hid_channel), bias=False)
        self.predict_head = nn.Linear(hid_channel, out_channel*2, bias=False)
        self.apply_dct = True

        self.validate_reasoning = True
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.category_num = category_num
        self.tao = 1

        self.add_agent_token = add_agent_token 
        if self.add_agent_token:
             self.agent_embed = nn.Parameter(torch.randn(1, 10,96))
             self.embed_MLP = nn.Sequential(
                 
                 
                nn.Dropout(0.4),
                nn.Linear(hidden_nf*3, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn)

        self.given_category = False
        if not self.given_category:
            self.edge_mlp = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(hidden_nf*2+hid_channel*2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn)
            
            self.coord_mlp = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(hid_channel*2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hid_channel*2),
                act_fn)

            self.node_mlp = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(hidden_nf+hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn)

            self.category_mlp = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(hidden_nf*2+hid_channel*2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, category_num),
                act_fn)

        for i in range(0, n_layers):
            if i == n_layers-1:
                self.add_module("gcl_%d" % i, Feature_learning_layer(self.hidden_nf, self.hidden_nf, self.hidden_nf, in_channel, hid_channel, out_channel, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh, apply_reasoning=False,input_reasoning=True, category_num=category_num))
                
                
            else:
                self.add_module("gcl_%d" % i, Feature_learning_layer(self.hidden_nf, self.hidden_nf, self.hidden_nf, in_channel, hid_channel, out_channel, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh, apply_reasoning=False,input_reasoning=True, category_num=category_num))



        self.embedding_1 = nn.Linear(in_node_nf*2, int(self.hidden_nf / 2))  
        self.embedding_v_1 = nn.Linear(in_node_nf*2, int(self.hidden_nf / 2))  
        self.embedding_a_1 = nn.Linear(in_node_nf*2, int(self.hidden_nf / 2))  
        self.coord_trans_1 = nn.Linear(in_channel*2, int(hid_channel), bias=False)  
        self.vel_trans_1 = nn.Linear(in_channel*2, int(hid_channel), bias=False)  
        self.predict_head_1 = nn.Linear(hid_channel, out_channel*2, bias=False)  

        self.tao_1 = 1

        if self.add_agent_token:
            self.agent_embed_1 = nn.Parameter(torch.randn(1, n_agent, 96))  
            self.embed_MLP_1 = nn.Sequential(
                
                
                nn.Dropout(0.1),
                nn.Linear(hidden_nf * 3, hidden_nf),  
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),  
                act_fn)

        if not self.given_category:
            self.edge_mlp_1 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_nf * 2 + hid_channel * 2, hidden_nf),  
                act_fn,  
                nn.Linear(hidden_nf, hidden_nf),  
                act_fn)

            self.coord_mlp_1 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hid_channel * 2, hidden_nf),  
                act_fn,
                nn.Linear(hidden_nf, hid_channel * 2),  
                act_fn)

            self.node_mlp_1 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_nf + hidden_nf, hidden_nf),  
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),  
                act_fn)

            self.category_mlp_1 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_nf * 2 + hid_channel * 2, hidden_nf),  
                act_fn,
                nn.Linear(hidden_nf, category_num),  
                act_fn)

        for i in range(0, n_layers):  
            if i == n_layers - 1:
                self.add_module("gcl1_%d" % i,Feature_learning_layer(self.hidden_nf, self.hidden_nf, self.hidden_nf, in_channel,hid_channel, out_channel, edges_in_d=in_edge_nf, act_fn=act_fn,coords_weight=coords_weight, recurrent=recurrent,norm_diff=norm_diff, tanh=tanh, apply_reasoning=False,input_reasoning=True, category_num=category_num))
                
                
            else:
                self.add_module("gcl1_%d" % i,Feature_learning_layer(self.hidden_nf, self.hidden_nf, self.hidden_nf, in_channel,hid_channel, out_channel, edges_in_d=in_edge_nf, act_fn=act_fn,coords_weight=coords_weight, recurrent=recurrent,norm_diff=norm_diff, tanh=tanh, apply_reasoning=False,input_reasoning=True, category_num=category_num))





        self.embedding_2 = nn.Linear(in_node_nf * 2, int(self.hidden_nf / 2))  
        self.embedding_v_2 = nn.Linear(in_node_nf * 2, int(self.hidden_nf / 2))  
        self.embedding_a_2 = nn.Linear(in_node_nf * 2, int(self.hidden_nf / 2))  
        self.coord_trans_2 = nn.Linear(in_channel * 2, int(hid_channel), bias=False)  
        self.vel_trans_2 = nn.Linear(in_channel * 2, int(hid_channel), bias=False)  
        self.predict_head_2 = nn.Linear(hid_channel, out_channel * 2, bias=False)  

        self.tao_1 = 1

        if self.add_agent_token:
            self.agent_embed_2 = nn.Parameter(torch.randn(1, n_agent, 96))  
            self.embed_MLP_2 = nn.Sequential(
                
                
                nn.Dropout(0.1),
                nn.Linear(hidden_nf * 3, hidden_nf),  
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),  
                act_fn)

        if not self.given_category:
            self.edge_mlp_2 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_nf * 2 + hid_channel * 2, hidden_nf),  
                act_fn,  
                nn.Linear(hidden_nf, hidden_nf),  
                act_fn)

            self.coord_mlp_2 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hid_channel * 2, hidden_nf),  
                act_fn,
                nn.Linear(hidden_nf, hid_channel * 2),  
                act_fn)

            self.node_mlp_2 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_nf + hidden_nf, hidden_nf),  
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),  
                act_fn)

            self.category_mlp_2 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_nf * 2 + hid_channel * 2, hidden_nf),  
                act_fn,
                nn.Linear(hidden_nf, category_num),  
                act_fn)

        for i in range(0, n_layers):  
            if i == n_layers - 1:
                self.add_module("gcl2_%d" % i,
                                Feature_learning_layer(self.hidden_nf, self.hidden_nf, self.hidden_nf, in_channel,
                                                       hid_channel, out_channel, edges_in_d=in_edge_nf, act_fn=act_fn,
                                                       coords_weight=coords_weight, recurrent=recurrent,
                                                       norm_diff=norm_diff, tanh=tanh, apply_reasoning=False,
                                                       input_reasoning=True, category_num=category_num))
                
                
            else:
                self.add_module("gcl2_%d" % i,
                                Feature_learning_layer(self.hidden_nf, self.hidden_nf, self.hidden_nf, in_channel,
                                                       hid_channel, out_channel, edges_in_d=in_edge_nf, act_fn=act_fn,
                                                       coords_weight=coords_weight, recurrent=recurrent,
                                                       norm_diff=norm_diff, tanh=tanh, apply_reasoning=False,
                                                       input_reasoning=True, category_num=category_num))



        self.A = nn.Parameter(torch.randn(3,22,22))
        self.CTR_in_channels=3
        self.CTR_out_channels=3
        self.convs1 = nn.ModuleList()
        self.num_subset=3
        self.alpha = nn.Parameter(torch.zeros(3,1))
        self.beta=nn.Parameter(torch.ones(3,1))
        for i in range(self.num_subset):
            self.convs1.append(CTRGC(self.CTR_in_channels, self.CTR_out_channels))

        self.mid_trans = nn.Sequential(
            
            
            
            nn.Linear(10, hidden_nf),  
            act_fn,
            nn.Linear(hidden_nf, 22),  
            act_fn)

        self.get_mid = AveargeJoint2()

        self.to(self.device)


    def get_dct_matrix(self,N,x):
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N) 

        idct_m = np.linalg.inv(dct_m)
        dct_m = torch.from_numpy(dct_m).type_as(x)
        idct_m = torch.from_numpy(idct_m).type_as(x)
        return dct_m, idct_m
    
    def transform_edge_attr(self,edge_attr):
        edge_attr = (edge_attr / 2) + 1
        interaction_category = F.one_hot(edge_attr.long(),num_classes=self.category_num)
        return interaction_category

    def calc_category(self,h,coord):

        
        
        
        
        
        



        import torch.nn.functional as F
        batch_size, agent_num, channels = coord.shape[0], coord.shape[1], coord.shape[2] 
        h1 = h[:,:,None,:].repeat(1,1,agent_num,1)
        h2 = h[:,None,:,:].repeat(1,agent_num,1,1)
        coord_diff = coord[:,:,None,:,:] - coord[:,None,:,:,:]
        coord_dist = torch.norm(coord_diff,dim=-1)
        coord_dist = self.coord_mlp(coord_dist)
        edge_feat_input = torch.cat([h1,h2,coord_dist],dim=-1)
        
        edge_feat = self.edge_mlp(edge_feat_input)
        mask = (torch.ones((agent_num,agent_num)) - torch.eye(agent_num)).type_as(edge_feat)
        mask = mask[None,:,:,None].repeat(batch_size,1,1,1)
        node_new = self.node_mlp(torch.cat([h,torch.sum(mask*edge_feat,dim=2)],dim=-1))
        
        node_new1 = node_new[:,:,None,:].repeat(1,1,agent_num,1)
        node_new2 = node_new[:,None,:,:].repeat(1,agent_num,1,1)
        edge_feat_input_new = torch.cat([node_new1,node_new2,coord_dist],dim=-1)
        interaction_category = F.softmax(self.category_mlp(edge_feat_input_new)/self.tao,dim=-1)
        return interaction_category

    def calc_category_1(self,h,coord):

        x = coord  
        y = None
        for i in range(self.num_subset):
            z = self.convs1[i](x, self.A[i], self.alpha[i], self.beta[i])
            y = z + y if y is not None else z
        coord = y

        import torch.nn.functional as F
        batch_size, agent_num, channels = coord.shape[0], coord.shape[1], coord.shape[2] 
        h1 = h[:,:,None,:].repeat(1,1,agent_num,1)
        h2 = h[:,None,:,:].repeat(1,agent_num,1,1)
        coord_diff = coord[:,:,None,:,:] - coord[:,None,:,:,:]
        coord_dist = torch.norm(coord_diff,dim=-1)
        coord_dist = self.coord_mlp_1(coord_dist)
        edge_feat_input = torch.cat([h1,h2,coord_dist],dim=-1)
        
        edge_feat = self.edge_mlp_1(edge_feat_input)
        mask = (torch.ones((agent_num,agent_num)) - torch.eye(agent_num)).type_as(edge_feat)
        mask = mask[None,:,:,None].repeat(batch_size,1,1,1)
        node_new = self.node_mlp_1(torch.cat([h,torch.sum(mask*edge_feat,dim=2)],dim=-1))
        
        node_new1 = node_new[:,:,None,:].repeat(1,1,agent_num,1)
        node_new2 = node_new[:,None,:,:].repeat(1,agent_num,1,1)
        edge_feat_input_new = torch.cat([node_new1,node_new2,coord_dist],dim=-1)
        interaction_category = F.softmax(self.category_mlp_1(edge_feat_input_new)/self.tao_1,dim=-1)
        return interaction_category


    def calc_category_2(self,h,coord):

        x = coord  
        y = None
        for i in range(self.num_subset):
            z = self.convs1[i](x, self.A[i], self.alpha[i], self.beta[i])
            y = z + y if y is not None else z
        coord = y

        import torch.nn.functional as F
        batch_size, agent_num, channels = coord.shape[0], coord.shape[1], coord.shape[2] 
        h1 = h[:,:,None,:].repeat(1,1,agent_num,1)
        h2 = h[:,None,:,:].repeat(1,agent_num,1,1)
        coord_diff = coord[:,:,None,:,:] - coord[:,None,:,:,:]
        coord_dist = torch.norm(coord_diff,dim=-1)
        coord_dist = self.coord_mlp_2(coord_dist)
        edge_feat_input = torch.cat([h1,h2,coord_dist],dim=-1)
        
        edge_feat = self.edge_mlp_2(edge_feat_input)
        mask = (torch.ones((agent_num,agent_num)) - torch.eye(agent_num)).type_as(edge_feat)
        mask = mask[None,:,:,None].repeat(batch_size,1,1,1)
        node_new = self.node_mlp_2(torch.cat([h,torch.sum(mask*edge_feat,dim=2)],dim=-1))
        
        node_new1 = node_new[:,:,None,:].repeat(1,1,agent_num,1)
        node_new2 = node_new[:,None,:,:].repeat(1,agent_num,1,1)
        edge_feat_input_new = torch.cat([node_new1,node_new2,coord_dist],dim=-1)
        interaction_category = F.softmax(self.category_mlp_2(edge_feat_input_new)/self.tao_1,dim=-1)
        return interaction_category




    def forward(self, h, x,x_end, vel, edge_attr=None):

        coord = x

        loc_0 = self.get_mid(x.transpose(1, 3))
        loc_0 = loc_0.transpose(1, 3)  


        for i in range(10):
            loc_0 = torch.cat([loc_0, loc_0[:, :, 9].unsqueeze(-2)], dim=-2)
        x_0=loc_0

        all_seqs_vel_0 = torch.zeros_like(x_0)  
        all_seqs_vel_0[:, :, 1:] = x_0[:, :, 1:] - x_0[:, :, :-1]  
        all_seqs_vel_0[:, :, 0] = x_0[:, :, 1]  
        vel_0=all_seqs_vel_0  

        h_0 = torch.sqrt(torch.sum(vel_0 ** 2, dim=-1)).detach()  



        accelerate_0 = torch.zeros_like(h_0)
        accelerate_0[:, :, 1:] = h_0[:, :, :-1]  
        accelerate_0[:, :, 0] = h_0[:, :, 0]  
        accelerate_0 = accelerate_0 - h_0

        vel_pre_0 = torch.zeros_like(vel_0)
        vel_pre_0[:,:,1:] = vel_0[:,:,:-1] 
        vel_pre_0[:,:,0] = vel_0[:,:,0]  
        EPS = 1e-6 
        vel_cosangle_0 = torch.sum(vel_pre_0*vel_0,dim=-1)/((torch.norm(vel_pre_0,dim=-1)+EPS)*(torch.norm(vel_0,dim=-1)+EPS))

        vel_angle_0 = torch.acos(torch.clamp(vel_cosangle_0,-1,1))

        batch_size, agent_num, length = x_0.shape[0], x_0.shape[1], x_0.shape[2] 



        if self.apply_dct:
            x_center_0 = torch.mean(x_0,dim=(1,2),keepdim=True)
            x_0 = x_0 - x_center_0 
            dct_m,_ = self.get_dct_matrix(self.in_channel*2,x_0) 
            _,idct_m = self.get_dct_matrix(self.out_channel*2,x_0)
            dct_m = dct_m[None,None,:,:].repeat(batch_size,agent_num,1,1)
            idct_m = idct_m[None,None,:,:].repeat(batch_size,agent_num,1,1)
            x_0 = torch.matmul(dct_m,x_0)
            vel_0 = torch.matmul(dct_m,vel_0)


        h_0 = self.embedding(h_0) 
        vel_angle_embedding_0 = self.embedding_v(vel_angle_0) 
        accelerate_0 = self.embedding_a(accelerate_0)  
        h_0 = torch.cat([h_0, vel_angle_embedding_0, accelerate_0], dim=-1)  

        if self.add_agent_token:
            batch_ind = torch.arange(batch_size)[:,None].cuda()
            agent_ind = torch.arange(agent_num)[None,:].cuda()
            
            h_0 = torch.cat([h_0,self.agent_embed.repeat(batch_size,1,1)],dim=-1) 
            h_0 = self.embed_MLP(h_0)

        x_mean_0 = torch.mean(torch.mean(x_0,dim=-2,keepdim=True),dim=-3,keepdim=True)
        x_0 = self.coord_trans((x_0-x_mean_0).transpose(2,3)).transpose(2,3) + x_mean_0
        vel_0 = self.vel_trans(vel_0.transpose(2,3)).transpose(2,3)
        x_cat_0 = torch.cat([x_0,vel_0],dim=-2)
        cagegory_per_layer = []
        if self.given_category:
            category = self.transform_edge_attr(edge_attr)
        else:
            category = self.calc_category(h_0,x_cat_0)

        for i in range(0, 4):
            h_0, x_0, category = self._modules["gcl_%d" % i](h_0, x_0, vel_0, edge_attr=edge_attr, category=category) 
            
            
            cagegory_per_layer.append(category) 

        x_mean = torch.mean(torch.mean(x_0,dim=-2,keepdim=True),dim=-3,keepdim=True) 
        x_0 = self.predict_head((x_0-x_mean).transpose(2,3)).transpose(2,3) + x_mean 
        if self.apply_dct: 
            x_0 = torch.matmul(idct_m,x_0)
            x_0 = x_0 + x_center_0

        x_mid = x_0.transpose(1, 3)
        x_mid = self.mid_trans(x_mid)  
        x_mid = x_mid.transpose(1, 3)


        x_1=torch.cat([coord,x_mid[:,:,10:20]], dim=-2)

        all_seqs_vel_1 = torch.zeros_like(x_1)  
        all_seqs_vel_1[:, :, 1:] = x_1[:, :, 1:] - x_1[:, :, :-1]  
        all_seqs_vel_1[:, :, 0] = x_1[:, :, 1]  
        vel_1 = all_seqs_vel_1  

        h_1 = torch.sqrt(torch.sum(vel_1 ** 2, dim=-1)).detach()  

        accelerate_1 = torch.zeros_like(h_1)
        accelerate_1[:, :, 1:] = h_1[:, :, :-1]  
        accelerate_1[:, :, 0] = h_1[:, :, 0]  
        accelerate_1 = accelerate_1 - h_1

        vel_pre_1 = torch.zeros_like(vel_1)
        vel_pre_1[:, :, 1:] = vel_1[:, :, :-1]  
        vel_pre_1[:, :, 0] = vel_1[:, :, 0]  
        EPS = 1e-6  
        vel_cosangle_1 = torch.sum(vel_pre_1 * vel_1, dim=-1) / ((torch.norm(vel_pre_1, dim=-1) + EPS) * (
                torch.norm(vel_1, dim=-1) + EPS))  

        vel_angle_1 = torch.acos(torch.clamp(vel_cosangle_1, -1,
                                           1))  

        batch_size, agent_num, length = x_1.shape[0], x_1.shape[1], x_1.shape[2]  


        if self.apply_dct:
            x_center_1 = torch.mean(x_1,dim=(1,2),keepdim=True)
            x_1 = x_1 - x_center_1 
            dct_m,_ = self.get_dct_matrix(self.in_channel*2,x_1) 
            _,idct_m = self.get_dct_matrix(self.out_channel*2,x_1)
            dct_m = dct_m[None,None,:,:].repeat(batch_size,agent_num,1,1)
            idct_m = idct_m[None,None,:,:].repeat(batch_size,agent_num,1,1)
            x_1 = torch.matmul(dct_m,x_1)


        h_1 = self.embedding_1(h_1) 
        vel_angle_embedding_1 = self.embedding_v_1(vel_angle_1)
        accelerate_1 = self.embedding_a_1(accelerate_1)  
        h_1 = torch.cat([h_1, vel_angle_embedding_1, accelerate_1], dim=-1)  
        if self.add_agent_token:
            batch_ind = torch.arange(batch_size)[:,None].cuda()
            agent_ind = torch.arange(agent_num)[None,:].cuda()
            
            h_1 = torch.cat([h_1,self.agent_embed_1.repeat(batch_size,1,1)],dim=-1) 
            h_1 = self.embed_MLP_1(h_1)


        x_mean_1 = torch.mean(torch.mean(x_1, dim=-2, keepdim=True), dim=-3, keepdim=True)  
        x_1 = self.coord_trans_1((x_1 - x_mean_1).transpose(2, 3)).transpose(2, 3) + x_mean_1  
        vel_1 = self.vel_trans_1(vel_1.transpose(2, 3)).transpose(2,3)  
        x_cat_1 = torch.cat([x_1, vel_1], dim=-2)  
        cagegory_per_layer_1 = []
        if self.given_category:
            category_1 = self.transform_edge_attr(edge_attr)
        else:
            category_1 = self.calc_category_1(h_1,x_cat_1)  

        for i in range(0, 4):  
            h_1, x_1, category_1 = self._modules["gcl1_%d" % i](h_1, x_1, vel_1, edge_attr=edge_attr, category=category_1)  
            
            
            cagegory_per_layer_1.append(category_1)  

        x_mean_1 = torch.mean(torch.mean(x_1, dim=-2, keepdim=True), dim=-3, keepdim=True)  
        x_1 = self.predict_head_1((x_1 - x_mean_1).transpose(2, 3)).transpose(2, 3) + x_mean_1  
        if self.apply_dct:  
            x_1 = torch.matmul(idct_m, x_1)
            x_1 = x_1 + x_center_1






        x_2=torch.cat([coord,x_1[:,:,10:20]], dim=-2)

        all_seqs_vel_2 = torch.zeros_like(x_2)  
        all_seqs_vel_2[:, :, 1:] = x_2[:, :, 1:] - x_2[:, :, :-1]  
        all_seqs_vel_2[:, :, 0] = x_2[:, :, 1]  
        vel_2 = all_seqs_vel_2  

        h_2 = torch.sqrt(torch.sum(vel_2 ** 2, dim=-1)).detach()  

        accelerate_2 = torch.zeros_like(h_2)
        accelerate_2[:, :, 1:] = h_2[:, :, :-1]  
        accelerate_2[:, :, 0] = h_2[:, :, 0]  
        accelerate_2 = accelerate_2 - h_2

        vel_pre_2 = torch.zeros_like(vel_2)
        vel_pre_2[:, :, 1:] = vel_2[:, :, :-1]  
        vel_pre_2[:, :, 0] = vel_2[:, :, 0]  
        EPS = 1e-6  
        vel_cosangle_2 = torch.sum(vel_pre_2 * vel_2, dim=-1) / ((torch.norm(vel_pre_2, dim=-1) + EPS) * (
                torch.norm(vel_2, dim=-1) + EPS))  

        vel_angle_2 = torch.acos(torch.clamp(vel_cosangle_2, -1,
                                           1))  

        batch_size, agent_num, length = x_2.shape[0], x_2.shape[1], x_2.shape[2]  


        if self.apply_dct:
            x_center_2 = torch.mean(x_2,dim=(1,2),keepdim=True)
            x_2 = x_2 - x_center_2 
            dct_m,_ = self.get_dct_matrix(self.in_channel*2,x_2) 
            _,idct_m = self.get_dct_matrix(self.out_channel*2,x_2)
            dct_m = dct_m[None,None,:,:].repeat(batch_size,agent_num,1,1)
            idct_m = idct_m[None,None,:,:].repeat(batch_size,agent_num,1,1)
            x_2 = torch.matmul(dct_m,x_2)


        h_2 = self.embedding_2(h_2) 
        vel_angle_embedding_2 = self.embedding_v_2(vel_angle_2)
        accelerate_2 = self.embedding_a_2(accelerate_2)  
        h_2 = torch.cat([h_2, vel_angle_embedding_2, accelerate_2], dim=-1)  
        if self.add_agent_token:
            batch_ind = torch.arange(batch_size)[:,None].cuda()
            agent_ind = torch.arange(agent_num)[None,:].cuda()
            
            h_2 = torch.cat([h_2,self.agent_embed_2.repeat(batch_size,1,1)],dim=-1) 
            h_2 = self.embed_MLP_2(h_2)

        x_mean_2 = torch.mean(torch.mean(x_2, dim=-2, keepdim=True), dim=-3, keepdim=True)  
        x_2 = self.coord_trans_2((x_2 - x_mean_2).transpose(2, 3)).transpose(2, 3) + x_mean_2  
        vel_2 = self.vel_trans_2(vel_2.transpose(2, 3)).transpose(2,3)  
        x_cat_2 = torch.cat([x_2, vel_2], dim=-2)  
        cagegory_per_layer_2 = []
        if self.given_category:
            category_2 = self.transform_edge_attr(edge_attr)
        else:
            category_2 = self.calc_category_2(h_2,x_cat_2)  

        for i in range(0, 4):  
            h_2, x_2, category_2 = self._modules["gcl2_%d" % i](h_2, x_2, vel_2, edge_attr=edge_attr, category=category_2)  
            
            
            cagegory_per_layer_2.append(category_2)  

        x_mean_2 = torch.mean(torch.mean(x_2, dim=-2, keepdim=True), dim=-3, keepdim=True)  
        x_2 = self.predict_head_2((x_2 - x_mean_2).transpose(2, 3)).transpose(2, 3) + x_mean_2  
        if self.apply_dct:  
            x_2 = torch.matmul(idct_m, x_2)
            x_2 = x_2 + x_center_2


        return x_0,x_1,x_2