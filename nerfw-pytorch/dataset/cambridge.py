import os.path
import sys
import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import *
import pickle


class CambridgeDataset(Dataset):
    def __init__(self, root_dir, scene='StMarysChurch', split='train', img_downscale=1, use_cache=False,
                 if_save_cache=True):
        self.scene = scene
        self.root_dir = root_dir
        self.split = split
        self.img_downscale = img_downscale
        self.img_size = (1920, 1080)  # fixed size of Cambridge nerf
        self.downscale_size = (self.img_size[0] // img_downscale, self.img_size[1] // img_downscale)
        self.true_downscale = self.img_size[0] / self.downscale_size[0]
        self.use_cache = use_cache
        self.if_save_cache = if_save_cache
        self.view_lighting = {}  # Store lighting conditions for each view
        self.load_data()
        self.view_dims = {}
        for vid, fname in self.view_filename.items():
            img_path = os.path.join(self.root_dir, self.scene, fname)
            with Image.open(img_path) as img:
                w0, h0 = img.size
            # record the *downscaled* dims for this view
            self.view_dims[vid] = (w0 // self.img_downscale,
                                   h0 // self.img_downscale)

    def load_data(self):
        """
        nvm format
        <Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
        <Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>

        dataset format
        ImageFile, Camera Position [X Y Z W P Q R], Lighting [Dawn Morning Noon Afternoon Dusk Night Cloudy]
        """
        print('load reconstruction data of scene "{}", split: {}'.format(self.scene, self.split))
        base_dir = os.path.join(self.root_dir, self.scene)
        # read Bundler .out
        with open(os.path.join(base_dir, 'notredame.out'), 'r') as f:
            bundle = f.readlines()
        with open(os.path.join(base_dir, 'dataset_train.txt'), 'r') as f:
            train_split = f.readlines()  # train split
        with open(os.path.join(base_dir, 'dataset_test.txt'), 'r') as f:
            test_split = f.readlines()  # test split


        with open(os.path.join(base_dir,'list.txt')) as f:
            all_images = [l.strip() for l in f if l.strip()]

        # 1. Parse cameras and lighting conditions
        # header: "# Bundle file v0.3"
        # next line: num_cams, num_points
        _, header = bundle[0], bundle[1].split()
        self.N_views, self.N_points = int(header[0]), int(header[1])
        if self.use_cache:
            self.load_cache()
            return
        cam_start = 2
        self.file2id = {}  # img name->id
        self.view_filename = {}  # id->img file name
        self.view_K = {}  # id->K
        self.view_c2w = {}  # id->c2w
        self.view_rectifymap = {}  # id->rectifyMap, the distortion correction
        self.view_w2c = {}
        self.view_lighting = {}  # id->lighting conditions [dawn, morning, noon, afternoon, dusk, night, cloudy]

        # for i, data in enumerate(nvm_data[3:3 + self.N_views]):
        #     data = data.split()
        #     # data = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
        #     dir_and_file=data[0].split('/')
        #     dir_and_file[-1]=dir_and_file[-1].split('.')[0]+'.png'
        #     data[0] = os.path.join(dir_and_file[0],dir_and_file[1])
        #     self.file2id[data[0]] = i
        #     self.view_filename[i] = data[0]
        #     params = np.array(data[1:-1], dtype=float)
        #     # img_downscale: fx、fy、cx、cy、img_w、img_h ↓
        #     K = np.zeros((3, 3), dtype=float)
        #     K[0, 0] = K[1, 1] = params[0] / self.img_downscale
        #     K[0, 2] = self.downscale_size[0] / 2
        #     K[1, 2] = self.downscale_size[1] / 2
        #     K[2, 2] = 1
        #     self.view_K[i] = K
        #     bottom = np.zeros(shape=(1, 4))
        #     bottom[0, -1] = 1
        #     rotmat = quat2rotmat(params[1:5]/np.linalg.norm(params[1:5]))
        #     t_vec = -rotmat @ params[5:8].reshape(3, 1)  # visualSFM存储的c2w,即world系下的camera center的位置。rotmat是w2c...
        #     w2c = np.vstack([np.hstack([rotmat, t_vec]), bottom])
        #     self.view_w2c[i] = w2c[:3]  # w2c
        #     c2w = np.linalg.inv(w2c)
        #     self.view_c2w[i] = c2w[:3]  # c2w

        #     self.view_rectifymap[i] = get_rectify_map((self.downscale_size[0], self.downscale_size[1]), params[-1], K)
        for i,img in enumerate(all_images):
            # each camera block is 5 lines:
            #   line 0: f, k1, k2
            #   lines 1–3: R rows
            #   line 4: t
            block = bundle[cam_start + 5*i : cam_start + 5*i + 5]
            # we assume filenames stored separately in list.txt → use view_filename from that

            self.file2id[img] = i
            self.view_filename[i] = img
            # parse intrinsics
            f, k1, k2 = map(float, block[0].split())
            K = np.array([[f/self.img_downscale, 0, self.downscale_size[0]/2],
                          [0, f/self.img_downscale, self.downscale_size[1]/2],
                          [0, 0, 1]],dtype=float)
            self.view_K[i] = K
            # parse R and t
            R = np.vstack([list(map(float, block[j].split())) for j in [1,2,3]])
            t = np.array(list(map(float, block[4].split())))
            # compute w2c and c2w
            bottom = np.array([[0,0,0,1]])
            t_vec = -R @ t[:,None]
            w2c = np.vstack([np.hstack([R, t_vec]), bottom])
            self.view_w2c[i] = w2c[:3]
            self.view_c2w[i] = np.linalg.inv(w2c)[:3]
            # rectify map
            self.view_rectifymap[i] = get_rectify_map(self.downscale_size, k1, K)



        # Load lighting conditions from train/test splits
        for data in train_split + test_split:
            data = data.split()
            dir_and_file = data[0].split('/')
            data[0] = os.path.join(dir_and_file[0], dir_and_file[1])
            id = self.file2id[data[0]]
            # Extract lighting conditions if available (7 values)
            if len(data) > 15:  # Original data has 8 values (filename + 7 camera params)
                lighting = np.array(data[15:22], dtype=float)  # [Dawn Morning Noon Afternoon Dusk Night Cloudy]
            else:
                # Default to noon, sunny if not specified
                lighting = np.zeros(7)
                lighting[2] = 1.0  # Noon
            self.view_lighting[id] = lighting

        # # 2. 解析points
        self.view_near = {}  # camera坐标系下最小深度, 用以nerf采样
        self.view_far = {}  # 最大深度
        # points_str = nvm_data[5 + self.N_views:5 + self.N_points + self.N_views]
        # self.points = np.vstack([np.array(x.split()[:6], dtype=float) for x in points_str])

        # 2. parse Bundler points (we only need XYZ for near/far)
        pts_start = cam_start + 5*self.N_views
        pts = []
        for j in range(self.N_points):
            pos = list(map(float, bundle[pts_start + 3*j].split()[:3]))
            pts.append(pos)
        self.points = np.array(pts)

        points_h = np.hstack([self.points[:, :3], np.ones(shape=(self.N_points, 1))])  # 齐次坐标
        for i in range(self.N_views):
            w2c = self.view_w2c[i]
            xyz_cam_i = points_h @ w2c.T
            xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2] > 0]
            self.view_near[i] = np.percentile(xyz_cam_i[:, 2], 0.1)  # 注意这里range[0,100]
            # 画出z的分布图,z∈(0,100)
            self.view_far[i] = np.percentile(xyz_cam_i[:, 2], 99.9)
        # 尺度放缩, 影响w2c、c2w的tvec, points以及near、far
        max_dep = np.max([v for v in self.view_far.values()])
        scale_factor = max_dep / 5
        self.scale_factor = scale_factor
        for i in range(self.N_views):
            self.view_c2w[i] = np.hstack([self.view_c2w[i][:, :3], self.view_c2w[i][:, 3:] / scale_factor])
            self.view_w2c[i] = np.hstack([self.view_w2c[i][:, :3], self.view_w2c[i][:, 3:] / scale_factor])
            self.view_near[i] = self.view_near[i] / scale_factor
            self.view_far[i] = self.view_far[i] / scale_factor
        self.points[:, :3] = self.points[:, :3] / scale_factor
        # 3. 解析train_split和test_split
        self.train_set = []  # train
        self.test_set = []  # test
        for data in train_split:
            data = data.split()
            dir_and_file=data[0].split('/')
            data[0] = os.path.join(dir_and_file[0],dir_and_file[1])
            id = self.file2id[data[0]]
            self.train_set.append(id)
        for data in test_split:
            data = data.split()
            dir_and_file=data[0].split('/')
            data[0] = os.path.join(dir_and_file[0],dir_and_file[1])
            id = self.file2id[data[0]]
            self.test_set.append(id)

        # 4. 生成all train view的rays
        if self.split == 'train':
            self.all_rays = []
            for i, id in tqdm.tqdm(enumerate(self.train_set), total=len(self.train_set), file=sys.stdout,
                                   desc="acquiring rays "):
                # distortion rectify
                rays = self.get_rays(id)
                self.all_rays += [rays]
                # if i==29: # test usage
                #     break
            self.all_rays = torch.vstack(self.all_rays)
            print('all rays: ', self.all_rays.shape)

        # save cache
        if self.if_save_cache:
            self.save_cache()

    def get_rays(self, i):
        """
        rays:(N_rays, 19), position(3)、direction(3)、near(1)、far(1)、id(1)、rgb(3)、lighting(7);
        pos、dir are in world coordinates, near/far for ray sampling,
        id matches appearance/transient embedding, lighting is one-hot + cloudy
        """
        base_dir = os.path.join(self.root_dir, self.scene)
        img = Image.open(os.path.join(base_dir, self.view_filename[i])).convert(mode='RGB')
        img_w, img_h = img.size
        img_w, img_h = img_w // self.img_downscale, img_h // self.img_downscale
        img = img.resize((img_w, img_h), Image.LANCZOS)
        rect_img = np.array(img)

        c2w = self.view_c2w[i]
        rays_o, rays_d = get_rays_o_d(img_w, img_h, self.view_K[i], c2w)
        nears = self.view_near[i] * torch.ones((len(rays_o), 1))
        fars = self.view_far[i] * torch.ones((len(rays_o), 1))
        ids = i * torch.ones((len(rays_o), 1))

        # Add lighting conditions to rays
        lighting = self.view_lighting.get(i, torch.zeros(7))  # Default to noon, sunny if not found
        lighting = torch.tensor(lighting).float()
        lighting = lighting.repeat(len(rays_o), 1)  # Repeat for each ray

        # Combine all components
        rays = torch.hstack([
            rays_o,                                                    # position (3)
            rays_d,                                                    # direction (3)
            nears,                                                     # near (1)
            fars,                                                      # far (1)
            ids,                                                       # view id (1)
            torch.FloatTensor(rect_img.reshape(-1, 3)) / 255,         # RGB (3)
            lighting                                                   # lighting (7)
        ])
        return rays

    def load_cache(self):
        # 1. load dicts
        dict_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'dicts.pkl')
        with open(dict_path, 'rb') as f:
            ds = pickle.load(f)
            self.train_set, self.test_set, self.file2id, self.view_filename, self.view_K, self.view_w2c, \
                self.view_c2w, _, self.view_near, self.view_far, self.scale_factor, self.view_lighting = ds
        # 2. load points
        points_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'sfm_points.npy')
        self.points = np.load(points_path)
        # 3. load all rays
        if self.split == 'train':
            ray_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'all_rays.npy')
            self.all_rays = torch.from_numpy(np.load(ray_path))
            print('all rays: ', self.all_rays.shape)
        # if self.split == 'train':
        #     self.all_rays=torch.zeros(size=(341913600,12),dtype=torch.float)
        print('cache load done...')

    def save_cache(self):
        if not os.path.exists(os.path.join(self.root_dir, self.scene, 'cache')):
            os.makedirs(os.path.join(self.root_dir, self.scene, 'cache'))
        # 1. Save dicts including lighting
        dict_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'dicts.pkl')
        with open(dict_path, 'wb') as f:
            pickle.dump([self.train_set, self.test_set, self.file2id, self.view_filename, self.view_K, self.view_w2c,
                         self.view_c2w, self.view_rectifymap, self.view_near, self.view_far, self.scale_factor,
                         self.view_lighting], f)
        # 2. 转numpy存all_rays
        rays = self.all_rays.numpy()
        ray_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'all_rays.npy')
        np.save(ray_path, rays)

        # 3. 存points
        points_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'sfm_points.npy')
        np.save(points_path, self.points)
        print('cache save done...')

        # 4. 单独存scale_factor(供后续pose regressor缩放尺度用, 保证与nerf-w建模尺度一致)
        with open(os.path.join(self.root_dir, self.scene, 'scale_factor.txt'), 'w') as f:
            np.savetxt(f, np.c_[self.scale_factor], fmt='%.6f')

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        elif self.split == 'valid':  # 固定训练集某张图像进行validate, 有效不会过拟合因为train unit是ray而不是一张img
            return len(self.train_set)

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.all_rays[idx]
        elif self.split in ['valid', 'test']:
            return self.get_rays(self.train_set[idx])


if __name__ == '__main__':
    server_dir = '/root/autodl-tmp/dataset/Cambridge'
    scene = 'StMarysChurch'
    local_dir = 'E:\\dataset\\Cambridge'
    train_dataset = CambridgeDataset(root_dir=local_dir, scene=scene,
                                     split='train', img_downscale=3, use_cache=False, if_save_cache=True)
