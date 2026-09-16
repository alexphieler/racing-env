import os

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def pipeline_path(*parts):
    return os.path.join(PIPELINE_DIR, *parts)


def artifact_prefix(reward_mode, noisy=False, last_action=True):
    noise_part = "noisy_" if noisy else ""
    last_action_part = "last_action_" if last_action else ""
    return f"{reward_mode}_{noise_part}{last_action_part}"


def state_actor_path(reward_mode, noisy=False, last_action=True, seed=None):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    return pipeline_path("networks", f"{artifact_prefix(reward_mode, noisy, last_action)}state_sac_actor{seed_part}.pt")


def state_ppo_actor_path(reward_mode, noisy=False, last_action=True, seed=None):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    return pipeline_path("networks", f"{artifact_prefix(reward_mode, noisy, last_action)}state_ppo_actor{seed_part}.pt")


def state_ppo_critic_path(reward_mode, noisy=False, last_action=True, seed=None):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    return pipeline_path("networks", f"{artifact_prefix(reward_mode, noisy, last_action)}state_ppo_critic{seed_part}.pt")


def state_qf1_path(reward_mode, noisy=False, last_action=True, seed=None):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    return pipeline_path("networks", f"{artifact_prefix(reward_mode, noisy, last_action)}state_sac_qf1{seed_part}.pt")


def state_qf2_path(reward_mode, noisy=False, last_action=True, seed=None):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    return pipeline_path("networks", f"{artifact_prefix(reward_mode, noisy, last_action)}state_sac_qf2{seed_part}.pt")


def replay_buffer_path(reward_mode, noisy=False, last_action=True, seed=None):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    return pipeline_path("data", f"{artifact_prefix(reward_mode, noisy, last_action)}state_sac_replay_buffer{seed_part}.pkl")


def vae_path(reward_mode, noisy=False, last_action=True):
    return pipeline_path("networks", f"{artifact_prefix(reward_mode, noisy, last_action)}vision_vae.pt")


def vae_preview_path(reward_mode, noisy=False, last_action=True):
    return pipeline_path("data", f"{artifact_prefix(reward_mode, noisy, last_action)}vision_vae.png")


def vision_actor_kind(latent=False):
    return "encoder" if latent else "resnet"


def bc_actor_path(reward_mode, noisy=False, latent=False, last_action=True, seed=None):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    return pipeline_path(
        "networks",
        f"{artifact_prefix(reward_mode, noisy, last_action)}vision_bc_{vision_actor_kind(latent)}{seed_part}.pt",
    )


def dagger_actor_path(reward_mode, noisy=False, latent=False, last_action=True, seed=None):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    return pipeline_path(
        "networks",
        f"{artifact_prefix(reward_mode, noisy, last_action)}vision_dagger_{vision_actor_kind(latent)}{seed_part}.pt",
    )


def ppo_actor_path(reward_mode, noisy=False, latent=False, last_action=True, seed=None):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    return pipeline_path(
        "networks",
        f"{artifact_prefix(reward_mode, noisy, last_action)}vision_ppo_{vision_actor_kind(latent)}{seed_part}.pt",
    )


def ppo_critic_path(reward_mode, noisy=False, latent=False, last_action=True, seed=None):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    return pipeline_path(
        "networks",
        f"{artifact_prefix(reward_mode, noisy, last_action)}vision_ppo_{vision_actor_kind(latent)}_critic{seed_part}.pt",
    )


def vision_sac_actor_path(reward_mode, noisy=False, latent=False, last_action=True, seed=None, asymmetric=False):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    critic_part = "_asymmetric" if asymmetric else ""
    return pipeline_path(
        "networks",
        f"{artifact_prefix(reward_mode, noisy, last_action)}vision_sac_{vision_actor_kind(latent)}{critic_part}{seed_part}.pt",
    )


def vision_sac_qf1_path(reward_mode, noisy=False, latent=False, last_action=True, seed=None, asymmetric=False):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    critic_part = "_asymmetric" if asymmetric else ""
    return pipeline_path(
        "networks",
        f"{artifact_prefix(reward_mode, noisy, last_action)}vision_sac_{vision_actor_kind(latent)}{critic_part}_qf1{seed_part}.pt",
    )


def vision_sac_qf2_path(reward_mode, noisy=False, latent=False, last_action=True, seed=None, asymmetric=False):
    seed_part = f"_seed_{seed}" if seed is not None else ""
    critic_part = "_asymmetric" if asymmetric else ""
    return pipeline_path(
        "networks", f"{artifact_prefix(reward_mode, noisy, last_action)}vision_sac_{vision_actor_kind(latent)}{critic_part}_qf2{seed_part}.pt",
    )
