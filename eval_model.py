import os
import model_v2 as model
from src.eagle import EagleDataset
from src import config as cfg
import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import adabound

def load_dataset(data_path: str = cfg.DATA_DIR,
                 mode: str = 'train',
                 window_length: int = 50,
                 apply_onehot: bool = True,
                 with_cells: bool = False,
                 with_cluster: bool = False,
                 normalize: bool = False,
                 batch_size: int = 1) -> torch.utils.data.DataLoader:

    dataset = EagleDataset(data_path=data_path,
                           mode=mode,
                           window_length=window_length,
                           apply_onehot=apply_onehot,
                           with_cells=with_cells,
                           with_cluster=with_cluster,
                           normalize=normalize)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=1,
                                             pin_memory=True)
    return dataloader

def get_loss_validation(velocity, pressure, output, state_hat, target, mask):
    MSE = nn.MSELoss()
    velocity = velocity[:, 1:]
    pressure = pressure[:, 1:]
    velocity_hat = state_hat[:, 1:, :, :2]
    mask = mask[:, 1:].unsqueeze(-1)

    rmse_velocity = torch.sqrt(((velocity * mask - velocity_hat * mask) ** 2).mean(dim=(-1)))
    loss_velocity = torch.mean(rmse_velocity)

    velocity_range = (velocity.max(dim=1).values - velocity.min(dim=1).values).unsqueeze(-1)
    normalized_rmse_velocity = rmse_velocity / velocity_range.mean()

    loss = MSE(target[..., :2] * mask, output[..., :2] * mask)

    pressure_hat = state_hat[:, 1:, :, 2:]
    rmse_pressure = torch.sqrt(((pressure * mask - pressure_hat * mask) ** 2).mean(dim=(-1)))
    loss_pressure = torch.mean(rmse_pressure)

    pressure_range = (pressure.max(dim=1).values - pressure.min(dim=1).values).unsqueeze(-1)
    normalized_rmse_pressure = rmse_pressure / pressure_range.mean()

    losses = {}
    losses['MSE_pressure'] = loss_pressure
    losses['loss'] = loss
    losses['MSE_velocity'] = loss_velocity
    losses['normalized_RMSE_velocity'] = torch.mean(normalized_rmse_velocity)
    losses['normalized_RMSE_pressure'] = torch.mean(normalized_rmse_pressure)

    return losses

def evaluate_model(test_dataloader: torch.utils.data.DataLoader, model: nn.Module, criterion: nn.Module) -> None:
    model.eval()
    total_loss = 0
    MSE_velocity = 0
    MSE_pressure = 0
    normalized_RMSE_velocity = 0
    normalized_RMSE_pressure = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            mesh_pos = data["mesh_pos"].to(cfg.DEVICE)
            edges = data['edges'].to(cfg.DEVICE).long()
            velocity = data["velocity"].to(cfg.DEVICE)
            pressure = data["pressure"].to(cfg.DEVICE)
            node_type = data["node_type"].to(cfg.DEVICE)

            state = torch.cat([velocity, pressure], dim=-1)
            mask = torch.ones_like(mesh_pos)[..., 0]

            state_hat, output_hat, target = model(mesh_pos, edges, state, node_type)

            losses = get_loss_validation(velocity, pressure, output_hat, state_hat, target, mask)
            total_loss += losses['loss'].item()
            MSE_velocity += losses['MSE_velocity'].item()
            MSE_pressure += losses['MSE_pressure'].item()
            normalized_RMSE_velocity += losses['normalized_RMSE_velocity'].item()
            normalized_RMSE_pressure += losses['normalized_RMSE_pressure'].item()

        print(f'Loss: {total_loss / len(test_dataloader)}')
        print(f'MSE_velocity: {MSE_velocity / len(test_dataloader)}')
        print(f'MSE_pressure: {MSE_pressure / len(test_dataloader)}')
        print(f'Normalized_RMSE_velocity: {normalized_RMSE_velocity / len(test_dataloader)}')
        print(f'Normalized_RMSE_pressure: {normalized_RMSE_pressure / len(test_dataloader)}')

if __name__ == '__main__':
    model = model.GNN().to(cfg.DEVICE)
    model.load_state_dict(torch.load(os.path.join(cfg.PROJECT_DIR, 'models', f'v4.pt')))

    test_dataloader = load_dataset(data_path=cfg.DATA_DIR, mode='test', window_length=100, with_cells=True)
    evaluate_model(test_dataloader, model, nn.MSELoss())