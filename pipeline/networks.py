import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import gymnasium as gym

from pacsimEnv import pacsimEnv

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A simple residual block with two Conv layers.
    It takes an input with 'in_channels' and outputs with 'out_channels'.
    It handles the downsampling (stride) and channel changes.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # The main convolutional path
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels)
        )
        
        # The "skip connection" path
        # If we change dimensions (stride != 1) or channels (in != out),
        # we need a 1x1 conv to make the 'x' tensor match the 'conv_path' output.
        self.skip_path = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_path = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )
            
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # Add the original 'x' (after passing through skip_path)
        # to the output of the convolutional path
        out = self.conv_path(x) + self.skip_path(x)
        out = self.relu(out)
        return out


class StructuredHierarchicalActor(nn.Module):
    def __init__(self, vision_input_shape, sensor_dim, action_dim=5):
        """
        vision_input_shape: tuple (T, V, C, H, W) -> e.g., (3, 3, 3, 224, 224)
        """
        super(StructuredHierarchicalActor, self).__init__()
        
        # Unpack input shape explicitly
        # T=Time, V=Views(Cameras), C=Channels
        self.T, self.V, self.C, self.H, self.W = vision_input_shape
        
        # --- 1. SHARED VISION ENCODER (Image -> Feature) ---
        # Input: (C, H, W) -> Output: 32 x 20 x 16 feature map.
        #
        # This is deliberately small: driving from cone images does not need
        # a wide ResNet-style tower.  The native 32-channel, 20 x 16 output retains
        # useful spatial continuity before the fully connected fusion layers.
        # The stem and first two residual stages downsample to 20 x 16 for
        # native 306 x 256 images; the third stage preserves it.
        # The resulting feature map is passed directly to
        # the fully connected fusion layers as 10,240 features per camera.
        self.vision_stem = nn.Sequential(
            nn.Conv2d(self.C, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = ResidualBlock(32, 32, stride=2)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 64, stride=1)
        self.conv_down = nn.Conv2d(64, 32, kernel_size=1) # Compression
        
        self.spatial_pool = nn.Identity()
        
        # Calculate CNN output size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, self.C, self.H, self.W)
            out = self.vision_stem(dummy)
            out = self.layer3(self.layer2(self.layer1(out)))
            out = self.spatial_pool(self.conv_down(out))
            self.cnn_feat_dim = out.numel()

        # --- 2. MULTI-VIEW FUSION (Fuse Cameras) ---
        # "What is happening at this specific timestep?"
        # Input: V * cnn_feat_dim 
        # Output: timestep_embed_dim
        self.timestep_embed_dim = 512
        self.view_fusion = nn.Sequential(
            nn.Linear(self.V * self.cnn_feat_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.timestep_embed_dim),
            nn.LayerNorm(self.timestep_embed_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- 3. TEMPORAL FUSION (Fuse visual history and proprioception) ---
        # "How did the scene change over time?"
        # Input: T * timestep_embed_dim + flattened proprioception
        # Output: context_dim
        self.sensor_dim = sensor_dim
        self.context_dim = 512
        self.temporal_fusion = nn.Sequential(
            nn.Linear(self.T * self.timestep_embed_dim + self.sensor_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.context_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- 4. POLICY HEAD ---
        self.joint_mlp = nn.Sequential(
            nn.Linear(self.context_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output heads
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        # Orthogonal Initialization
        torch.nn.init.orthogonal_(self.fc_mean.weight, gain=1.0)
        torch.nn.init.constant_(self.fc_mean.bias, 0)
        torch.nn.init.orthogonal_(self.fc_logstd.weight, gain=0.01)
        torch.nn.init.constant_(self.fc_logstd.bias, 0)
        
        self.action_scale = 1.0
        self.action_bias = 0.0

    def forward(self, vision_input, sensor_input):
        """
        vision_input: [B, T, V, C, H, W] 
        sensor_input: [B, S_dim]
        """
        # --- Sanity Checks ---
        if torch.isnan(vision_input).any():
            raise ValueError("Vision input contains NaNs!")
        if torch.isnan(sensor_input).any():
            raise ValueError("Sensor input contains NaNs!")
        if torch.isinf(vision_input).any() or torch.isinf(sensor_input).any():
            raise ValueError("Input contains Infinity!")
        # ---------------------
        
        # Convert and normalize on the active device. Camera frames stay uint8
        # through replay sampling and host-to-device transfer.
        vision_input = vision_input.to(dtype=torch.float32) / 255.0

        B, T, V, C, H, W = vision_input.shape
        if T != self.T:
            raise ValueError(f"Expected {self.T} stacked frames, got {T}")
        expected_sensor_dim = self.sensor_dim
        if sensor_input.ndim != 2 or sensor_input.shape != (B, expected_sensor_dim):
            raise ValueError(
                f"Expected sensor input shape ({B}, {expected_sensor_dim}), got {tuple(sensor_input.shape)}"
            )
        
        # === STEP 1: BATCH FOLDING (Parallel CNN) ===
        # Merge Batch, Time, and View dimensions to treat them all as independent images
        # Shape: [B*T*V, C, H, W]
        x = vision_input.view(B * T * V, C, H, W)
        
        # Apply Shared CNN
        x = self.vision_stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv_down(x)
        x = self.spatial_pool(x)
        
        # Flatten feature maps: [B*T*V, FeatureDim]
        cnn_feats = x.view(B * T * V, -1)
        
        # === STEP 2: MULTI-VIEW FUSION ===
        # Unfold: [B, T, V, FeatureDim]
        cnn_feats = cnn_feats.view(B, T, V, -1)
        
        # Flatten Cameras into the feature dimension
        # Shape: [B, T, (V * FeatureDim)]
        view_input = cnn_feats.view(B, T, -1)
        
        # Apply MLP. Note: PyTorch Linear applies to the last dimension, 
        # so this effectively runs the MLP on every timestep independently in parallel.
        # Shape: [B, T, timestep_embed_dim]
        time_embeddings = self.view_fusion(view_input)
        
        # === STEP 3: TEMPORAL FUSION ===
        # Flatten Time into the feature dimension
        # Shape: [B, (T * timestep_embed_dim) + sensor_dim]
        temporal_input = torch.cat([time_embeddings.reshape(B, -1), sensor_input], dim=1)
        
        # Shape: [B, context_dim]
        visual_context = self.temporal_fusion(temporal_input)
        
        # === STEP 4: POLICY OUTPUT ===
        joint_state = self.joint_mlp(visual_context)
        
        # Heads
        mean = self.fc_mean(joint_state)
        log_std = self.fc_logstd(joint_state)
        
        # Stable Log Std Squashing
        log_std = torch.tanh(log_std)
        log_std = -20 + 0.5 * (2 - -20) * (log_std + 1)

        return mean, log_std

    def get_action(self, vision_input, sensor_input):
        """
        Samples an action from the policy, applies Tanh squashing,
        and calculates the log_prob.
        (This method is unchanged from your original)
        """
        mean, log_std = self(vision_input, sensor_input)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Calculate log_prob, correcting for the Tanh squashing
        log_prob = normal.log_prob(x_t)
        # The log-prob calculation for Tanh squashing
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # 3. Calculate Entropy (required for loss calculation)
        # Approximation: Use the entropy of the Gaussian base
        entropy = normal.entropy().sum(1, keepdim=True)

        # Calculate the deterministic (mean) action, also squashed
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, entropy, mean

    def get_imitation_loss(self, vision_input, sensor_input, expert_actions):
        """
        Calculates NLL loss for Behavior Cloning on a Tanh-Squashed Policy.
        """
        # 1. Get distribution parameters from the network
        #    Note: We do NOT sample here. We just need the parameters.
        mu_raw, log_std = self(vision_input, sensor_input)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu_raw, std)

        # 2. Un-scale and Un-squash the expert action
        #    We need to convert the expert action (e.g., -1 to 1) back 
        #    to the raw Gaussian space to evaluate the Normal distribution.
        
        # Remove the scale and bias
        y_expert = (expert_actions - self.action_bias) / self.action_scale
        
        # CRITICAL: Clamp inputs to atanh to avoid NaNs. 
        # atanh(1.0) or atanh(-1.0) is infinity. 
        # We clip to 1 - epsilon (e.g., 0.999999)
        epsilon = 1e-6
        y_expert = torch.clamp(y_expert, -1.0 + epsilon, 1.0 - epsilon)
        
        # Inverse Tanh (atanh)
        x_expert = torch.atanh(y_expert)
        # Alternatively in newer PyTorch: torch.atanh(y_expert)

        # 3. Calculate Log Probability of these specific expert values
        log_prob = normal.log_prob(x_expert)

        # 4. Apply the Tanh Correction (Jacobian)
        #    This is the same math as in your get_action, but using y_expert
        log_prob -= torch.log(self.action_scale * (1 - y_expert.pow(2)) + 1e-6)
        
        # Sum over action dimensions
        log_prob = log_prob.sum(1, keepdim=True)

        # 5. Return Negative Log Likelihood (Maximize probability = Minimize NLL)
        loss = -log_prob.mean()
        
        return loss



class ConvVAE(nn.Module):
    def __init__(self, z_size=1024, kl_tolerance=0.5, input_spatial=(306, 256)):
        super(ConvVAE, self).__init__()
        
        self.z_size = z_size
        self.kl_tolerance = kl_tolerance
        self.input_spatial = tuple(input_spatial)
        encoder_sizes = self._encoder_spatial_sizes(self.input_spatial)
        self.pooled_spatial = encoder_sizes[-1]
        decoder_output_padding = self._decoder_output_padding(encoder_sizes)
        
        def norm(channels):
            return nn.GroupNorm(8, channels)

        # Encoder layers. GroupNorm keeps train/eval behavior consistent for
        # replay-buffer batches, where BatchNorm running stats can drift badly.
        self.encoder_conv = nn.Sequential(
            # Input: [N, 3, H, W]
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), 
            norm(64),
            nn.LeakyReLU(0.2),
            # [N, 64, H/2, W/2]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            norm(128),
            nn.LeakyReLU(0.2),
            # [N, 128, H/4, W/4]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            norm(256),
            nn.LeakyReLU(0.2),
            # [N, 256, H/8, W/8]
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            norm(512),
            nn.LeakyReLU(0.2),
            # [N, 512, H/16, W/16]
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            norm(1024),
            nn.LeakyReLU(0.2)
            # [N, 1024, H/32, W/32]
        )
        
        # Pool to the configured encoder resolution. For the native 306x256
        # camera frames this is 9x8 after five stride-2 encoder stages.
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.pooled_spatial)
        self.flattened_size = self.pooled_spatial[0] * self.pooled_spatial[1] * 1024
        
        # VAE linear layers
        self.fc_mu = nn.Linear(self.flattened_size, z_size)
        self.fc_logvar = nn.Linear(self.flattened_size, z_size)
        
        # Decoder layers 
        self.decoder_fc = nn.Linear(z_size, self.flattened_size)
        
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, output_padding=decoder_output_padding[0]),
            norm(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=decoder_output_padding[1]),
            norm(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=decoder_output_padding[2]),
            norm(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=decoder_output_padding[3]),
            norm(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, output_padding=decoder_output_padding[4]),
            nn.Sigmoid() 
        )

    @staticmethod
    def _encoder_spatial_sizes(input_spatial):
        sizes = [tuple(input_spatial)]
        h, w = sizes[0]
        for _ in range(5):
            h = h // 2
            w = w // 2
            sizes.append((h, w))
        return sizes

    @staticmethod
    def _decoder_output_padding(encoder_sizes):
        output_padding = []
        for source, target in zip(reversed(encoder_sizes[1:]), reversed(encoder_sizes[:-1])):
            op = (target[0] - 2 * source[0], target[1] - 2 * source[1])
            if op[0] not in (0, 1) or op[1] not in (0, 1):
                raise ValueError(f"Cannot invert encoder spatial transition {source} -> {target}")
            output_padding.append(op)
        return tuple(output_padding)

    def reparameterize(self, mu, logvar):
        """
        Implements the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std

    def encode(self, x):
        if x.shape[-2:] != self.input_spatial:
            raise ValueError(f"ConvVAE expected input spatial {self.input_spatial}, got {tuple(x.shape[-2:])}")

        h = self.encoder_conv(x)
        h = self.adaptive_pool(h)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.decoder_fc(z)
        ph, pw = self.pooled_spatial
        h = h.view(h.size(0), 1024, ph, pw)
        return self.decoder_deconv(h)

    def reconstruct(self, x, deterministic=True):
        mu, logvar = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def forward(self, x):
        """
        Defines the forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            recon_x (torch.Tensor): Reconstructed input
            mu (torch.Tensor): Latent mean
            logvar (torch.Tensor): Latent log variance
        """
        mu, logvar = self.encode(x)
        
        # --- Reparameterize ---
        z = self.reparameterize(mu, logvar)
        
        # --- Decode ---
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Calculates the VAE loss (Reconstruction + KL divergence).
        """
        # --- Reconstruction Loss (MSE) ---
        r_loss_per_item = torch.sum(F.mse_loss(recon_x, x, reduction='none'), dim=[1, 2, 3])
        r_loss = torch.mean(r_loss_per_item)

        # --- KL Divergence Loss ---
        kl_loss_per_item = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        kl_tolerance_tensor = torch.tensor(
            self.kl_tolerance * self.z_size, 
            device=mu.device, 
            dtype=mu.dtype
        )
        kl_loss_per_item = torch.maximum(kl_loss_per_item, kl_tolerance_tensor)
        
        kl_loss = torch.mean(kl_loss_per_item)
        
        total_loss = r_loss + kl_loss
        
        return total_loss, r_loss, kl_loss


class LatentHierarchicalActor(nn.Module):
    def __init__(self, vae, vision_input_shape, sensor_dim, action_dim=5, freeze_vae=True):
        """
        vae: A pretrained ConvVAE instance
        vision_input_shape: tuple (T, V, C, H, W) -> e.g., (3, 3, 3, 224, 224)
        """
        super(LatentHierarchicalActor, self).__init__()
        
        self.vae = vae
        # Freeze VAE weights
        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
            
        self.T, self.V, self.C, self.H, self.W = vision_input_shape
        self.z_size = vae.z_size
        
        # --- 2. MULTI-VIEW FUSION (Fuse VAE Latents) ---
        # Input: V * z_size 
        self.timestep_embed_dim = 512
        self.view_fusion = nn.Sequential(
            nn.Linear(self.V * self.z_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, self.timestep_embed_dim),
            nn.LayerNorm(self.timestep_embed_dim),
            nn.ReLU()
        )

        # --- 3. TEMPORAL FUSION (Fuse visual history and proprioception) ---
        self.sensor_dim = sensor_dim
        self.context_dim = 512
        self.temporal_fusion = nn.Sequential(
            nn.Linear(self.T * self.timestep_embed_dim + self.sensor_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.context_dim),
            nn.ReLU()
        )

        # --- 4. POLICY HEAD ---
        self.joint_mlp = nn.Sequential(
            nn.Linear(self.context_dim, 256),
            nn.ReLU(),
        )
        
        # Output heads
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        
        self.action_scale = 1.0
        self.action_bias = 0.0

    def forward(self, vision_input, sensor_input):
        """
        vision_input: [B, T, V, C, H, W] 
        sensor_input: [B, S_dim]
        """
        # --- Sanity Checks ---
        if torch.isnan(vision_input).any():
            raise ValueError("Vision input contains NaNs!")
        if torch.isnan(sensor_input).any():
            raise ValueError("Sensor input contains NaNs!")
        if torch.isinf(vision_input).any() or torch.isinf(sensor_input).any():
            raise ValueError("Input contains Infinity!")
            
        # Match the structured actor: keep camera frames uint8 until they have
        # reached the active device, then normalize before VAE encoding.
        vision_input = vision_input.to(dtype=torch.float32) / 255.0

        B, T, V, C, H, W = vision_input.shape
        if T != self.T:
            raise ValueError(f"Expected {self.T} stacked frames, got {T}")
        expected_sensor_dim = self.sensor_dim
        if sensor_input.ndim != 2 or sensor_input.shape != (B, expected_sensor_dim):
            raise ValueError(
                f"Expected sensor input shape ({B}, {expected_sensor_dim}), got {tuple(sensor_input.shape)}"
            )
        
        # === STEP 1: BATCH FOLDING & VAE ENCODING ===
        # Merge Batch, Time, and View dimensions
        x = vision_input.reshape(B * T * V, C, H, W)
        
        if any(param.requires_grad for param in self.vae.parameters()):
            # Pass through VAE encoder parts. Keep this path differentiable when
            # callers explicitly requested VAE fine-tuning.
            h = self.vae.encoder_conv(x)
            h = self.vae.adaptive_pool(h)
            h = h.view(h.size(0), -1)
            mu = self.vae.fc_mu(h)
        else:
            with torch.no_grad():
                h = self.vae.encoder_conv(x)
                h = self.vae.adaptive_pool(h)
                h = h.view(h.size(0), -1)
                mu = self.vae.fc_mu(h)
        # We use mu as the representative visual feature.
        
        # Unfold back to [B, T, V, z_size]
        z_feats = mu.view(B, T, V, -1)
        
        # === STEP 2: MULTI-VIEW FUSION ===
        # Flatten Cameras into the feature dimension: [B, T, (V * z_size)]
        view_input = z_feats.view(B, T, -1)
        
        # Shape: [B, T, timestep_embed_dim]
        time_embeddings = self.view_fusion(view_input)
        
        # === STEP 3: TEMPORAL FUSION ===
        # Flatten Time: [B, (T * timestep_embed_dim)]
        temporal_input = torch.cat([time_embeddings.view(B, -1), sensor_input], dim=1)
        
        # Shape: [B, context_dim]
        visual_context = self.temporal_fusion(temporal_input)
        
        # === STEP 4: POLICY OUTPUT ===
        joint_state = self.joint_mlp(visual_context)
        
        # Heads
        mean = self.fc_mean(joint_state)
        log_std = self.fc_logstd(joint_state)
        
        # Stable Log Std Squashing
        log_std = torch.tanh(log_std)
        log_std = -20 + 0.5 * (2 - -20) * (log_std + 1)

        return mean, log_std

    def get_action(self, vision_input, sensor_input):
        mean, log_std = self(vision_input, sensor_input)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        entropy = normal.entropy().sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, entropy, mean

    def get_imitation_loss(self, vision_input, sensor_input, expert_actions):
        mu_raw, log_std = self(vision_input, sensor_input)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu_raw, std)

        y_expert = (expert_actions - self.action_bias) / self.action_scale
        epsilon = 1e-6
        y_expert = torch.clamp(y_expert, -1.0 + epsilon, 1.0 - epsilon)
        x_expert = torch.atanh(y_expert)

        log_prob = normal.log_prob(x_expert)
        log_prob -= torch.log(self.action_scale * (1 - y_expert.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return -log_prob.mean()





# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        obs_size_flattened = 0
        for i in obs_space:
            obs_size_flattened += np.array(obs_space[i].shape).prod()

        self.fc1 = nn.Linear(
            # np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            obs_size_flattened + 5,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Critic(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        obs_size_flattened = 0
        for i in obs_space:
            obs_size_flattened += np.array(obs_space[i].shape).prod()

        self.fc1 = nn.Linear(
            # np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            obs_size_flattened + 0,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        torch.nn.init.orthogonal_(self.fc3.weight, 1.0)

    def forward(self, x):
        # x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5

STATE_OBSERVATION_KEYS = ("ranges", "velocity", "rpm", "imu", "steer", "last_action")
VISION_SENSOR_KEYS = ("rpm", "imu", "steer", "last_action")


def _has_obs_key(obs, key):
    if isinstance(obs, gym.spaces.Dict):
        return key in obs.spaces
    return key in obs


def _obs_value(obs, key):
    if isinstance(obs, gym.spaces.Dict):
        return obs.spaces[key]
    return obs[key]


def _flatten_obs_values(obs, keys, batched):
    flattened = []
    any_tensor = False
    for key in keys:
        if not _has_obs_key(obs, key):
            continue
        value = _obs_value(obs, key)
        any_tensor = any_tensor or torch.is_tensor(value)
        if batched:
            flattened.append(value.reshape(value.shape[0], -1))
        else:
            flattened.append(value.flatten())

    if len(flattened) == 0:
        raise KeyError("None of the requested observation keys are present: {0}".format(keys))

    axis = 1 if batched else 0
    if any_tensor:
        tensors = [
            value if torch.is_tensor(value) else torch.from_numpy(value)
            for value in flattened
        ]
        return torch.cat(tensors, dim=axis).to(dtype=torch.float32)

    return torch.from_numpy(np.concatenate(flattened, axis=axis)).to(dtype=torch.float32)


def _flatten_vision_sensor_timesteps(obs, batched):
    """Flatten vision sensors in timestep-major order for aligned fusion."""
    per_timestep = []
    frame_count = None
    batch_size = None
    for key in VISION_SENSOR_KEYS:
        if not _has_obs_key(obs, key):
            continue
        value = _obs_value(obs, key)
        value = value if torch.is_tensor(value) else torch.from_numpy(value)

        min_dims = 2 if batched else 1
        if value.ndim < min_dims:
            raise ValueError(
                f"Vision sensor {key!r} must include a frame-stack dimension; got {tuple(value.shape)}"
            )

        if batched:
            current_batch_size, current_frame_count = value.shape[:2]
            reshaped = value.reshape(current_batch_size, current_frame_count, -1)
        else:
            current_frame_count = value.shape[0]
            current_batch_size = None
            reshaped = value.reshape(current_frame_count, -1)

        if frame_count is None:
            frame_count = current_frame_count
            batch_size = current_batch_size
        elif frame_count != current_frame_count or (batched and batch_size != current_batch_size):
            raise ValueError("All vision sensors must have matching batch and frame-stack dimensions.")
        per_timestep.append(reshaped)

    if not per_timestep:
        raise KeyError("None of the vision sensor keys are present in the observation.")

    fused = torch.cat(per_timestep, dim=-1).to(dtype=torch.float32)
    return fused.reshape(batch_size, -1) if batched else fused.reshape(-1)


def flattened_dim_from_space(obs_space, keys):
    return int(sum(
        np.array(obs_space[key].shape).prod()
        for key in keys
        if _has_obs_key(obs_space, key)
    ))


def state_input_dim(obs_space):
    return flattened_dim_from_space(obs_space, STATE_OBSERVATION_KEYS)


def vision_sensor_dim(obs_space):
    return flattened_dim_from_space(obs_space, VISION_SENSOR_KEYS)


class Actor(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        obs_size_flattened = 0
        for i in obs_space:
            obs_size_flattened += np.array(obs_space[i].shape).prod()
        # print(obs_size_flattened)
        self.fc1 = nn.Linear(obs_size_flattened, 256)
        self.fc2 = nn.Linear(256, 256)
        # self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        # self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_mean = nn.Linear(256, 5)
        self.fc_logstd = nn.Linear(256, 5)
        self.action_scale = 1.0
        self.action_bias = 0.0


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        mean = torch.tanh(mean)
        return action, log_prob, mean


def flattenFuncVision(obs):
    concat = _flatten_vision_sensor_timesteps(obs, batched=True)
    cam = obs["cameraFront"]
    camL = obs["cameraLeft"]
    camR = obs["cameraRight"]
    if isinstance(cam, np.ndarray):
        cam = torch.from_numpy(cam)
        camL = torch.from_numpy(camL)
        camR = torch.from_numpy(camR)

    camL = camL.unsqueeze(2)
    cam = cam.unsqueeze(2)
    camR = camR.unsqueeze(2)
    cams = torch.cat((camL, cam, camR), dim=2)

    return cams, concat



def flattenFuncVisionSingle(obs):

    concat = _flatten_vision_sensor_timesteps(obs, batched=False)
    concat = torch.unsqueeze(concat, dim=0)

    cam = torch.from_numpy(obs["cameraFront"])
    cam = torch.unsqueeze(cam, dim=0)

    camL = torch.from_numpy(obs["cameraLeft"])
    camL = torch.unsqueeze(camL, dim=0)

    camR = torch.from_numpy(obs["cameraRight"])
    camR = torch.unsqueeze(camR, dim=0)

    camL = camL.unsqueeze(2)
    cam = cam.unsqueeze(2)
    camR = camR.unsqueeze(2)
    cams = torch.cat((camL, cam, camR), dim=2)

    return cams, concat

def flattenFuncState(obs):
    return _flatten_obs_values(obs, STATE_OBSERVATION_KEYS, batched=True)

def flattenFuncStateSingle(obs):
    concat = _flatten_obs_values(obs, STATE_OBSERVATION_KEYS, batched=False)
    concat = torch.unsqueeze(concat, dim=0)
    return concat

def filterObservationForState(obs):
    keysState = [key for key in STATE_OBSERVATION_KEYS if _has_obs_key(obs, key)]
    obsNew = {key: _obs_value(obs, key) for key in keysState}
    return obsNew

def filterObservationForVision(obs):
    keysState = ["cameraFront"] + [key for key in VISION_SENSOR_KEYS if _has_obs_key(obs, key)]
    obsNew = {key: _obs_value(obs, key) for key in keysState}
    return obsNew
