from src.eagle import EagleDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from matplotlib.tri import Triangulation
import os
from src import config as cfg
import matplotlib.animation as animation
import torch
from model_v2 import GNN

if __name__ == '__main__':
    model = GNN().to(cfg.DEVICE)
    model.load_state_dict(torch.load(os.path.join(cfg.PROJECT_DIR, 'models', f'v4.pt')))
    model.eval()

    d = EagleDataset(data_path=cfg.DATA_DIR, window_length=51, mode='test', with_cells=True)
    dataloader = DataLoader(d, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


    model.eval()
    for i in tqdm(range(10)):
        data = d[i]
        mesh_pos_model = data["mesh_pos"].index_select(0, torch.arange(start=0, end=50))
        edges_model = data['edges'].index_select(0, torch.arange(start=0, end=50))
        velocity_model = data["velocity"].index_select(0, torch.arange(start=0, end=50))
        pressure_model = data["pressure"].index_select(0, torch.arange(start=0, end=50))
        node_type_model = data["node_type"].index_select(0, torch.arange(start=0, end=50))

        mesh_pos = mesh_pos_model.to(cfg.DEVICE).unsqueeze(0)
        edges = edges_model.to(cfg.DEVICE).long().unsqueeze(0)
        velocity = velocity_model.to(cfg.DEVICE).unsqueeze(0)
        pressure = pressure_model.to(cfg.DEVICE).unsqueeze(0)
        node_type = node_type_model.to(cfg.DEVICE).unsqueeze(0)
        state = torch.cat([velocity, pressure], dim=-1)
        mask = torch.ones_like(mesh_pos)[..., 0]

        state_hat, output_hat, target = model(mesh_pos, edges, state, node_type)

        m = mesh_pos.squeeze(0)
        cells = data['cells'].numpy()[0]
        v = (state_hat[:, 1:, :, :2].squeeze(0) ** 2).sum(-1).detach().numpy()

        # print(v.shape, v_pred.shape)

        fig, ax = plt.subplots(figsize=(10, 5))
        tri = Triangulation(m[0, :, 0], m[0, :, 1], cells)
        vmin = v.min()
        vmax = v.max()
        print(v[0].shape)
        r = ax.tripcolor(tri, v[0], cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        ax.set_aspect('equal')
        fig.suptitle("Timestep 0")

        def animate(i):
            maskedTris = tri.get_masked_triangles()
            r.set_array(v[i][maskedTris].mean(axis=1))
            fig.suptitle(f"Timestep {i}")
            return [r]

        plt.tight_layout()
        fig.subplots_adjust(right=1, left=0)
        anim = animation.FuncAnimation(fig, animate, frames=1, interval=1)
        writer = animation.writers["ffmpeg"](fps=30)
        anim.save(os.path.join(cfg.VIDEOS_DIR, f'{i}_model.mp4'), writer=writer)

        #--------------------------

        m = data["mesh_pos"]
        cells = data['cells'].numpy()[0]
        v = (data['velocity'] ** 2).sum(-1)

        # print(v.shape, v_pred.shape)

        fig, ax = plt.subplots(figsize=(10, 5))
        tri = Triangulation(m[0, :, 0], m[0, :, 1], cells)
        vmin = v.min()
        vmax = v.max()
        print(v[0].shape)
        r = ax.tripcolor(tri, v[0], cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        ax.set_aspect('equal')
        fig.suptitle("Timestep 0")


        def animate(i):
            maskedTris = tri.get_masked_triangles()
            r.set_array(v[i][maskedTris].mean(axis=1))
            fig.suptitle(f"Timestep {i}")
            return [r]


        plt.tight_layout()
        fig.subplots_adjust(right=1, left=0)
        anim = animation.FuncAnimation(fig, animate, frames=1, interval=1)
        writer = animation.writers["ffmpeg"](fps=30)
        anim.save(os.path.join(cfg.VIDEOS_DIR, f'{i}_model_true.mp4'), writer=writer)