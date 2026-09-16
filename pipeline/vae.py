import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from networks import ConvVAE, flattenFuncVision
from buffers import ReplayBuffer, DictReplayBuffer, load_replay_buffer, save_replay_buffer
from artifact_names import replay_buffer_path, vae_path, vae_preview_path
from tqdm.rich import tqdm, trange
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# --- 1. CONFIGURATION ---
IMAGE_CHANNELS = 3
BETA = 1.0                   # KL factor; keep 0.0 when reconstruction quality matters most.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

parser = argparse.ArgumentParser(description="Train ConvVAE from a replay buffer")
parser.add_argument("--buffer", type=str, default=None, help="Path to replay buffer pickle file")
parser.add_argument("--noisy", action="store_true", help="Use the noisy state buffer for the selected reward mode")
parser.add_argument(
    "--reward-mode",
    choices=["conservative", "aggressive"],
    default=None,
    help="Reward preset used in default replay-buffer names. Defaults to conservative.",
)
parser.add_argument("--epochs", type=int, default=10000, help="Number of VAE training epochs")
parser.add_argument("--batch-size", type=int, default=16, help="Replay-buffer batch size")
parser.add_argument("--beta", type=float, default=BETA, help="KL weight. Keep 0.0 for best reconstruction quality.")
parser.add_argument("--kl-anneal-epochs", type=int, default=700, help="Epochs used to ramp KL weight from 0 to beta")
parser.add_argument("--viz-dpi", type=int, default=200, help="DPI for saved reconstruction preview")
parser.add_argument("--save-path", type=str, default=None, help="Output VAE model path")
parser.add_argument("--viz-path", type=str, default=None, help="Output reconstruction image path")
parser.add_argument(
    "--include-last-action",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use the last-action replay-buffer naming variant.",
)
args = parser.parse_args()

reward_mode = args.reward_mode or "conservative"
if args.buffer:
    rb_path = args.buffer
else:
    rb_path = replay_buffer_path(reward_mode, args.noisy, args.include_last_action)

if not os.path.exists(rb_path):
    raise FileNotFoundError(f"Expected replay buffer at exactly: {rb_path}")

rb_basename = os.path.splitext(os.path.basename(rb_path))[0]
save_path = args.save_path or vae_path(reward_mode, args.noisy, args.include_last_action)
viz_path = args.viz_path or vae_preview_path(reward_mode, args.noisy, args.include_last_action)


# --- Example Usage ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_size = 256
kl_tolerance = 0.5
device = DEVICE

# Replay-buffer camera frames are kept at their stored resolution.
# PyTorch expects channels first: (N, C, H, W).
def prepare_vision_batch(observations, device):
    vision, sensors = flattenFuncVision(observations)
    vision = vision.contiguous().reshape(-1, IMAGE_CHANNELS, vision.shape[-2], vision.shape[-1])
    vision = vision / 255.0
    return vision.to(device)


writer = SummaryWriter(f"runs/vae_training_{rb_basename}")
writer.add_text("info", f"replay_buffer: {rb_path}")

rb = load_replay_buffer(rb_path)
batch_size = args.batch_size
sample_observations = rb.sample(1)[0]
input_spatial = tuple(sample_observations["cameraFront"].shape[-2:])
print(f"Using VAE input/output spatial resolution: {input_spatial}")

model = ConvVAE(z_size=z_size, kl_tolerance=kl_tolerance, input_spatial=input_spatial).to(device)

# 2. Define optimizer
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()
batch_idx = 0

MAX_BETA = args.beta
BETA_ANNEAL_EPOCHS = max(args.kl_anneal_epochs, 1)


print(f"Loaded replay buffer: {rb_path}")
print(rb.size())

bufferSamples = None
for epoch in tqdm(range(args.epochs)):
    bufferSamples = rb.sample(batch_size)
    observations = bufferSamples[0]
    # obsflat = flattenFunc(observations)
    vision = prepare_vision_batch(observations, device)
    # print(vision.shape)
    data = vision

    # data = data.to(DEVICE)
    
    # Forward pass
    recon_batch, mu, logvar = model(data)


    # Calculate loss
    if epoch < BETA_ANNEAL_EPOCHS:
        # Linearly ramp up beta from 0 to MAX_BETA
        current_beta = MAX_BETA * (epoch / BETA_ANNEAL_EPOCHS)
    else:
        # After annealing, keep beta at its max value
        current_beta = MAX_BETA

    recon_loss = F.mse_loss(recon_batch, data)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    kl_loss_per_pixel = kl_loss / data[0].numel()
    loss = recon_loss + current_beta * kl_loss_per_pixel

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    writer.add_scalar("Loss/train", loss.item(), epoch)
    writer.add_scalar("Loss/reconstruction_mse", recon_loss.item(), epoch)
    writer.add_scalar("Loss/kl_per_pixel", kl_loss_per_pixel.item(), epoch)
    writer.add_scalar("Loss/beta", current_beta, epoch)
    
    if batch_idx % 2 == 0:
        # print(f"Epoch {epoch} [Batch {batch_idx}]: "
        #         f"Total Loss: {loss.item():.4f}, "
        #         f"Recon Loss: {recon_loss.item():.4f}, "
        #         f"KL Loss: {kl_loss.item():.4f}")
        print(f"Epoch {epoch} [Batch {batch_idx}]: "
                f"Total Loss: {loss.item():.6f}, "
                f"Recon MSE: {recon_loss.item():.6f}, "
                f"KL/pixel: {kl_loss_per_pixel.item():.6f}, "
                f"Beta: {current_beta:.6f}")

    batch_idx += 1

writer.close()

# 4. GET THE ENCODER FOR YOUR RL POLICY
# After training, save the state dict:
# torch.save(model.state_dict(), 'spatial_vae.pt')
os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
torch.save(model, save_path)
print(f"Saved VAE to: {save_path}")

def visualize_reconstruction(model, image_samples, device, num_images=8):
    """
    Visualizes original vs. reconstructed images from the VAE.
    
    Args:
        model (nn.Module): The trained VAE model.
        data_loader (DataLoader): DataLoader to fetch a batch from.
        device (torch.device): The device the model and data are on.
        num_images (int): Number of image pairs to display.
    """
    
    print("Generating reconstructions...")
    # Set model to evaluation mode
    model.eval()
    
    # # Get one batch of test data
    # try:
    #     originals = next(iter(data_loader))
    # except StopIteration:
    #     print("DataLoader is empty.")
    #     return
    originals=image_samples

    # Move data to the correct device
    originals = originals.to(device)
    
    # We only need num_images
    originals = originals[:num_images]
    
    # Get deterministic reconstructions; stochastic samples can make a trained
    # VAE preview look worse than the latent used by the policy.
    with torch.no_grad():
        if hasattr(model, "reconstruct"):
            recons, _, _ = model.reconstruct(originals, deterministic=True)
        else:
            recons, _, _ = model(originals)
        # recons = model(originals)
        # recons, mu, logvar = model(data)

    # Move images back to CPU for plotting
    originals = originals.cpu()
    recons = recons.cpu()
    
    # --- Create a comparison grid ---
    # We'll interleave the images: [original_1, recon_1, original_2, recon_2, ...]
    
    # 1. Stack them: (num_images, 2, C, H, W)
    #    (The '2' dimension holds the [original, recon] pair)
    comparison = torch.stack([originals, recons], dim=1)
    
    # 2. Flatten them into a single list: (num_images * 2, C, H, W)
    comparison = comparison.view(-1, *originals.shape[1:])

    # 3. Create the grid
    #    nrow=2 makes it display in [original, recon] columns
    grid = vutils.make_grid(
        comparison,
        nrow=2,             # Display as [Original, Recon]
        padding=2,          # Padding between images
        normalize=False
    )
    
    # --- Plot the grid (convert colors and rotate 90 deg right) ---
    plt.figure(figsize=(num_images * 2, 8))
    # Convert (C, H, W) -> (H, W, C), move to CPU and to numpy
    img = grid.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    # Replay-buffer frames display correctly with the same BGR->RGB flip and
    # clockwise grid rotation used by the original preview code.
    if img.shape[-1] == 3:
        img = img[..., ::-1]
    img_rot = np.rot90(img, k=-1)
    plt.imshow(img_rot)
    plt.title('Originals (Top Row) vs. Reconstructions (Bottom Row)')
    plt.axis('off')
    plt.savefig(viz_path, dpi=args.viz_dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved VAE reconstruction preview to: {viz_path}")



if bufferSamples is None:
    bufferSamples = rb.sample(batch_size)
observations = bufferSamples[0]
# obsflat = flattenFunc(observations)
vision = prepare_vision_batch(observations, device)

visualize_reconstruction(model, vision, device)
