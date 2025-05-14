import torch
import numpy as np
import torch.nn as nn


# NeRFW的basemodel,input:xyz、dir、appearance、transient; output:static_rgb, static_sigma, t_sigma, t_rgb, beta
# 由θ1、θ2和θ3组成
# θ1: 建模σ和scene隐式语义表示z(t)
# θ2: 输入appearance、dir、z(t)直接建模rgb
# θ3: 输入transient、z(t)建模瞬态对象σ、rgb和beta

class NeRFW(nn.Module):
    def __init__(self, layers=8, W=256, in_xyz_dim=63, in_dir_dim=27, encode_a=True, encode_t=True, a_dim=48, t_dim=16,
                 res_layer=[4], device='cpu'):
        super().__init__()
        # init
        self.layers = layers
        self.W = W
        self.in_xyz_dim = in_xyz_dim
        self.in_dir_dim = in_dir_dim
        self.encode_a = encode_a
        self.encode_t = encode_t
        self.a_dim = a_dim if encode_a else 0
        self.t_dim = t_dim if encode_t else 0
        self.res_layer = res_layer
        self.device = device

        # init nerf-w structures
        # θ1: obtain the voxel density, i.e σ.
        # input: [xyz_embedding]
        setattr(self, "theta1_encode_layer1", nn.Sequential(nn.Linear(in_xyz_dim, W), nn.ReLU(inplace=True)))
        for i in range(1, layers):
            layer = nn.Sequential(
                nn.Linear((in_xyz_dim if i in res_layer else 0) + W, W), nn.ReLU(True))
            setattr(self, "theta1_encode_layer%d" % (i + 1), layer)
        self.theta1_final_encode = nn.Linear(W, W)  # 再次encode static scene的隐式语义特征以输入到θ2和θ3中。
        self.theta1_decode = nn.Sequential(nn.Linear(W, 1), nn.Softplus())

        # Lighting-aware components
        self.lighting_mlp = nn.Sequential(
            nn.Linear(7, 32),  # 7 = 6 one-hot time + 1 cloudiness
            nn.ReLU(True),
            nn.Linear(32, 16)
        )

        # θ2: obtain the final rgb combining diffuse and specular
        self.theta2_diffuse = nn.Sequential(
            nn.Linear(self.a_dim + 16 + W, W // 2),  # appearance + lighting + scene features
            nn.ReLU(True),
            nn.Linear(W // 2, 3),
            nn.Sigmoid()
        )

        self.theta2_specular = nn.Sequential(
            nn.Linear(self.a_dim + in_dir_dim + 16 + W, W // 2),  # appearance + direction + lighting + scene features
            nn.ReLU(True),
            nn.Linear(W // 2, 3),
            nn.Sigmoid()
        )

        if encode_t:
            # encode: t_dim+W => W//2
            self.theta3_encode = nn.Sequential(nn.Linear(self.t_dim + W, W // 2), nn.ReLU(True),
                                               nn.Linear(W // 2, W // 2), nn.ReLU(True),
                                               nn.Linear(W // 2, W // 2), nn.ReLU(True),
                                               nn.Linear(W // 2, W // 2), nn.ReLU(True))
            # decode: transient sigma, transient rgb, transient beta(std)
            self.theta3_decode_sigma = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())
            self.theta3_decode_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())
            self.theta3_decode_beta = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())

    def forward(self, x, flag):
        """
        nerf-w model with lighting, input [xyz_emb, dir_emb, appearance_emb, transient_emb, lighting], output sigma、rgb
        :param x: shape: (N_points, in_xyz_dim+in_dir_dim+a_dim+t_dim+7)
        :param flag: 0,1,2

        Outputs:
            if flag==0:
                only output static_sigma from θ1
            elif flag==1:
                output static_sigma, rgb from θ2
            elif flag==2:
                output static_sigma, static_rgb, transient_sigma, transient_rgb, transient beta

        """
        xyz_emb = x[:, :self.in_xyz_dim]
        encode_xyz = xyz_emb
        for i in range(self.layers):
            layer = getattr(self, "theta1_encode_layer%d" % (i + 1))
            if i in self.res_layer:
                encode_xyz = layer(torch.hstack([xyz_emb, encode_xyz]))
            else:
                encode_xyz = layer(encode_xyz)

        # flag==0, only static sigma
        static_sigma = self.theta1_decode(encode_xyz)
        if flag == 0:
            return static_sigma

        # extract embeddings from x
        curr_idx = self.in_xyz_dim
        dir_emb = x[:, curr_idx:curr_idx + self.in_dir_dim]
        curr_idx += self.in_dir_dim

        appearance_emb = x[:, curr_idx:curr_idx + self.a_dim] if self.encode_a else None
        curr_idx += self.a_dim if self.encode_a else 0

        transient_emb = x[:, curr_idx:curr_idx + self.t_dim] if self.encode_t else None
        curr_idx += self.t_dim if self.encode_t else 0

        lighting_emb = x[:, curr_idx:curr_idx + 7]  # 7 = 6 one-hot time + 1 cloudiness
        lighting_features = self.lighting_mlp(lighting_emb)

        static_scene_emb = self.theta1_final_encode(encode_xyz)

        # Calculate diffuse and specular components
        if self.encode_a:
            diffuse_input = torch.hstack([appearance_emb, lighting_features, static_scene_emb])
            specular_input = torch.hstack([appearance_emb, dir_emb, lighting_features, static_scene_emb])
        else:
            diffuse_input = torch.hstack([lighting_features, static_scene_emb])
            specular_input = torch.hstack([dir_emb, lighting_features, static_scene_emb])

        diffuse_rgb = self.theta2_diffuse(diffuse_input)
        specular_rgb = self.theta2_specular(specular_input)

        # Combine diffuse and specular based on lighting
        cloudiness = lighting_emb[:, -1:]  # Get cloudiness factor
        static_rgb = diffuse_rgb + (1.0 - cloudiness) * specular_rgb  # Less specular when cloudy

        # flag==1, static sigma and static rgb
        if flag == 1:
            return static_sigma, static_rgb

        # flag==2 static sigma, static rgb, transient sigma rgb and beta
        if self.encode_t:
            transient_encode = self.theta3_encode(torch.hstack([transient_emb, static_scene_emb]))
            transient_sigma = self.theta3_decode_sigma(transient_encode)
            transient_rgb = self.theta3_decode_rgb(transient_encode)
            transient_beta = self.theta3_decode_beta(transient_encode)
            return static_sigma, static_rgb, transient_sigma, transient_rgb, transient_beta
        else:
            return static_sigma, static_rgb
