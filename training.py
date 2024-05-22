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


def compute_navier_stokes_residuals(velocity: torch.Tensor,
                                    pressure: torch.Tensor,
                                    density: float,
                                    viscosity: float,
                                    mesh_pos: torch.Tensor) -> tuple:
    # Assuming velocity: [batch_size, sequence_length, num_nodes, 2]
    # Assuming pressure: [batch_size, sequence_length, num_nodes, 1]
    # Assuming mesh_pos: [batch_size, sequence_length, num_nodes, 2]

    velocity = velocity.requires_grad_(True)
    pressure = pressure.requires_grad_(True)

    # Compute spatial gradients using autograd
    grad_velocity_x = \
    torch.autograd.grad(velocity[..., 0], velocity, grad_outputs=torch.ones_like(velocity[..., 0]), create_graph=True,
                        retain_graph=True)[0]
    grad_velocity_y = \
    torch.autograd.grad(velocity[..., 1], velocity, grad_outputs=torch.ones_like(velocity[..., 1]), create_graph=True,
                        retain_graph=True)[0]
    grad_pressure = \
    torch.autograd.grad(pressure[..., 0], pressure, grad_outputs=torch.ones_like(pressure[..., 0]), create_graph=True,
                        retain_graph=True)[0]

    # Compute divergence of velocity (continuity equation residual)
    divergence = grad_velocity_x[..., 0] + grad_velocity_y[..., 1]

    # Compute momentum equation residuals
    velocity_dot_grad_velocity_x = velocity[..., 0] * grad_velocity_x[..., 0] + velocity[..., 1] * grad_velocity_x[
        ..., 1]
    velocity_dot_grad_velocity_y = velocity[..., 0] * grad_velocity_y[..., 0] + velocity[..., 1] * grad_velocity_y[
        ..., 1]

    residual_momentum_x = (
            grad_velocity_x[..., 0] + velocity_dot_grad_velocity_x + grad_pressure[..., 0] / density - viscosity * (
                grad_velocity_x[..., 0] + grad_velocity_y[..., 1])
    )
    residual_momentum_y = (
            grad_velocity_y[..., 1] + velocity_dot_grad_velocity_y + grad_pressure[..., 1] / density - viscosity * (
                grad_velocity_x[..., 0] + grad_velocity_y[..., 1])
    )

    return divergence, residual_momentum_x, residual_momentum_y

def get_loss_train(velocity: torch.Tensor,
             pressure: torch.Tensor,
             output: torch.Tensor,
             state_hat: torch.Tensor,
             target: torch.Tensor,
             mask: torch.Tensor,
             mesh_pos: torch.Tensor,
             density: float=1.225,
             viscosity: float=1.48e-5) -> dict:

    MSE = nn.MSELoss()
    velocity = velocity[:, 1:]
    pressure = pressure[:, 1:]
    velocity_hat = state_hat[:, 1:, :, :2].requires_grad_(True)
    mask = mask[:, 1:].unsqueeze(-1)

    rmse_velocity = torch.sqrt(((velocity * mask - velocity_hat * mask) ** 2).mean(dim=(-1)))
    loss_velocity = torch.mean(rmse_velocity)

    velocity_range = (velocity.max(dim=1).values - velocity.min(dim=1).values).unsqueeze(-1)
    normalized_rmse_velocity = rmse_velocity / velocity_range.mean()

    loss = MSE(target[..., :2] * mask, output[..., :2] * mask)

    pressure_hat = state_hat[:, 1:, :, 2:].requires_grad_(True)
    rmse_pressure = torch.sqrt(((pressure * mask - pressure_hat * mask) ** 2).mean(dim=(-1)))
    loss_pressure = torch.mean(rmse_pressure)

    pressure_range = (pressure.max(dim=1).values - pressure.min(dim=1).values).unsqueeze(-1)
    normalized_rmse_pressure = rmse_pressure / pressure_range.mean()

    divergence, residual_momentum_x, residual_momentum_y = compute_navier_stokes_residuals(velocity_hat, pressure_hat,
                                                                                           density, viscosity,
                                                                                           mesh_pos[:, 1:])
    loss_continuity = torch.mean(divergence ** 2)
    loss_momentum_x = torch.mean(residual_momentum_x ** 2)
    loss_momentum_y = torch.mean(residual_momentum_y ** 2)

    loss = loss + loss_continuity + loss_momentum_x + loss_momentum_y

    losses = {}
    losses['MSE_pressure'] = loss_pressure
    losses['loss'] = loss
    losses['MSE_velocity'] = loss_velocity
    losses['normalized_RMSE_velocity'] = torch.mean(normalized_rmse_velocity)
    losses['normalized_RMSE_pressure'] = torch.mean(normalized_rmse_pressure)
    losses['loss_continuity'] = loss_continuity
    losses['loss_momentum_x'] = loss_momentum_x
    losses['loss_momentum_y'] = loss_momentum_y

    return losses

def get_loss_validation(velocity: torch.Tensor,
                        pressure: torch.Tensor,
                        output: torch.Tensor,
                        state_hat: torch.Tensor,
                        target: torch.Tensor,
                        mask: torch.Tensor) -> dict:

    """
    Changes the loss so physical equations are not considered in validation
    """
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

def train(model: nn.Module,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          writer: SummaryWriter,
          num_epochs: int = 10) -> None:

    for epoch in range(num_epochs):

        current_time = datetime.now().strftime('%H:%M')
        print(f"Epoch {epoch}/{num_epochs}, time started: {current_time}")
        model.train()

        for batch_idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            mesh_pos = data["mesh_pos"].to(cfg.DEVICE)
            edges = data['edges'].to(cfg.DEVICE).long()
            velocity = data["velocity"].to(cfg.DEVICE)
            pressure = data["pressure"].to(cfg.DEVICE)
            node_type = data["node_type"].to(cfg.DEVICE)

            state = torch.cat([velocity, pressure], dim=-1)
            mask = torch.ones_like(mesh_pos)[..., 0]

            state_hat, output_hat, target = model(mesh_pos, edges, state, node_type)

            losses = get_loss_train(velocity, pressure, output_hat, state_hat, target, mask, mesh_pos)

            writer.add_scalar('Loss/train', losses['loss'].item(), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('MSE_velocity/train', losses['MSE_velocity'].item(), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('MSE_pressure/train', losses['MSE_pressure'].item(), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Normalized_RMSE_velocity/train', losses['normalized_RMSE_velocity'].item(), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Normalized_RMSE_pressure/train', losses['normalized_RMSE_pressure'].item(), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Loss_continuity/train', losses['loss_continuity'].item(), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Loss_momentum_x/train', losses['loss_momentum_x'].item(), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Loss_momentum_y/train', losses['loss_momentum_y'].item(), epoch * len(train_dataloader) + batch_idx)

            losses['loss'].backward()
            optimizer.step()

        #print(f"Epoch {epoch}/{num_epochs} training finished, time ended: {datetime.now().strftime('%H:%M')}")

        model.eval()
        total_loss = 0
        MSE_velocity = 0
        MSE_pressure = 0
        normalized_RMSE_velocity = 0
        normalized_RMSE_pressure = 0


        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):
                mesh_pos = data["mesh_pos"].to(cfg.DEVICE)
                edges = data['edges'].to(cfg.DEVICE).long()
                velocity = data["velocity"].to(cfg.DEVICE)
                pressure = data["pressure"].to(cfg.DEVICE)
                node_type = data["node_type"].to(cfg.DEVICE)

                state = torch.cat([velocity, pressure], dim=-1)
                mask = torch.ones_like(mesh_pos)[..., 0]

                state_hat, output_hat, target = model(mesh_pos, edges, state, node_type)

                # Changes the loss so physical equations are not considered in validation
                losses = get_loss_validation(velocity, pressure, output_hat, state_hat, target, mask)
                total_loss += losses['loss'].item()
                MSE_velocity += losses['MSE_velocity'].item()
                MSE_pressure += losses['MSE_pressure'].item()
                normalized_RMSE_velocity += losses['normalized_RMSE_velocity'].item()
                normalized_RMSE_pressure += losses['normalized_RMSE_pressure'].item()


            writer.add_scalar('Loss/valid', total_loss / len(val_dataloader), epoch)
            writer.add_scalar('MSE_velocity/valid', MSE_velocity / len(val_dataloader), epoch)
            writer.add_scalar('MSE_pressure/valid', MSE_pressure / len(val_dataloader), epoch)
            writer.add_scalar('Normalized_RMSE_velocity/valid', normalized_RMSE_velocity / len(val_dataloader), epoch)
            writer.add_scalar('Normalized_RMSE_pressure/valid', normalized_RMSE_pressure / len(val_dataloader), epoch)


if __name__ == '__main__':
    train_dataloader = load_dataset(mode='train')
    val_dataloader = load_dataset(mode='valid')
    model = model.GNN().to(cfg.DEVICE)
    optimizer = adabound.AdaBound(model.parameters(), lr=cfg.LR, final_lr=0.1)
    criterion = nn.MSELoss()
    writer = SummaryWriter()

    train(model, optimizer, criterion, train_dataloader, val_dataloader, writer, num_epochs=25)

    comment = input("Enter a name for this experiment: ")
    writer.add_text('Experiment', comment)
    writer.close()

    save_model = input("Do you want to save the model parameters? (yes/no): ")
    if save_model.lower() == 'yes':
        model_path = os.path.join(cfg.PROJECT_DIR, 'models', f'{comment}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")
