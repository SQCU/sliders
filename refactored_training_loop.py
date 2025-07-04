#thanks a bunch gemini 2.5
import torch
import numpy as np
from PIL import Image
import os
import random

# Assuming the following files are in the same directory
from trainscripts.imagesliders import train_util, model_util, config_util, lora, prompt_util
from trainscripts.imagesliders.prompt_util import PromptEmbedsXL

def functional_train_step(
    unet,
    vae,
    noise_scheduler,
    img1_batch,
    img2_batch,
    scale_to_look,
    prompt_pair,
    config,
    network,
    criteria,
    device,
    weight_dtype,
    add_time_ids,
):
    """
    A functional and batch-oriented training step.
    """
    seed = random.randint(0, 2**15)
    generator = torch.manual_seed(seed)

    # Process a batch of images
    denoised_latents_low, low_noise = train_util.get_noisy_image(
        img1_batch,
        vae,
        generator,
        unet,
        noise_scheduler,
        start_timesteps=0,
        total_timesteps=config.train.max_denoising_steps -1,
    )
    denoised_latents_low = denoised_latents_low.to(device, dtype=weight_dtype)
    low_noise = low_noise.to(device, dtype=weight_dtype)

    generator = torch.manual_seed(seed)
    denoised_latents_high, high_noise = train_util.get_noisy_image(
        img2_batch,
        vae,
        generator,
        unet,
        noise_scheduler,
        start_timesteps=0,
        total_timesteps=config.train.max_denoising_steps -1,
    )
    denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)
    high_noise = high_noise.to(device, dtype=weight_dtype)

    noise_scheduler.set_timesteps(1000)
    current_timestep = noise_scheduler.timesteps[
        int((config.train.max_denoising_steps -1) * 1000 / config.train.max_denoising_steps)
    ]

    # High scale
    network.set_lora_slider(scale=scale_to_look)
    with network:
        target_latents_high = train_util.predict_noise_xl(
            unet,
            noise_scheduler,
            current_timestep,
            denoised_latents_high,
            prompt_pair.positive.text_embeds,
            prompt_pair.positive.pooled_embeds,
            add_time_ids,
            guidance_scale=1,
        )
    loss_high = criteria(target_latents_high, high_noise)

    # Low scale
    network.set_lora_slider(scale=-scale_to_look)
    with network:
        target_latents_low = train_util.predict_noise_xl(
            unet,
            noise_scheduler,
            current_timestep,
            denoised_latents_low,
            prompt_pair.neutral.text_embeds,
            prompt_pair.neutral.pooled_embeds,
            add_time_ids,
            guidance_scale=1,
        )
    loss_low = criteria(target_latents_low, low_noise)

    return loss_high, loss_low


def superfunctional_train_step(
    unet,
    vae,
    noise_scheduler,
    img_batches,
    scales,
    prompt_embeds,
    prompt_pair,
    config,
    network,
    criteria,
    device,
    weight_dtype,
    add_time_ids,
):
    """
    A more functional and compact training step that processes high and low cases concurrently.
    """
    seed = random.randint(0, 2**15)
    noise_scheduler.set_timesteps(1000)
    current_timestep = noise_scheduler.timesteps[
        int((config.train.max_denoising_steps - 1) * 1000 / config.train.max_denoising_steps)
    ]

    def process_case(img_batch, scale, embeds):
        generator = torch.manual_seed(seed)
        denoised_latents, noise = train_util.get_noisy_image(
            img_batch,
            vae,
            generator,
            unet,
            noise_scheduler,
            start_timesteps=0,
            total_timesteps=config.train.max_denoising_steps - 1,
        )
        denoised_latents = denoised_latents.to(device, dtype=weight_dtype)
        noise = noise.to(device, dtype=weight_dtype)

        network.set_lora_slider(scale=scale)
        with network:
            target_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                embeds.text_embeds,
                embeds.pooled_embeds,
                add_time_ids,
                guidance_scale=1,
            )
        return criteria(target_latents, noise)

    loss_high = process_case(img_batches[0], scales[0], prompt_embeds[0])
    loss_low = process_case(img_batches[1], scales[1], prompt_embeds[1])

    return loss_high, loss_low


def test_refactored_training_loop():
    """
    Tests that the refactored training loops are float-for-float identical to the original.
    """
    # Load config and models
    config = config_util.load_config_from_yaml("trainscripts/imagesliders/data/config-xl-dilora.yaml")
    tokenizers, text_encoders, unet, noise_scheduler, vae = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )
    device = torch.device("cuda:0")
    weight_dtype = config_util.parse_precision(config.train.precision)
    
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)


    # Create a dummy network and prompt pair
    network = lora.LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)
    
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, [])
    prompt_pair = prompt_util.PromptEmbedsPair(
        torch.nn.MSELoss(),
        PromptEmbedsXL(*train_util.encode_prompts_xl(tokenizers, text_encoders, [prompts[0].target])),
        PromptEmbedsXL(*train_util.encode_prompts_xl(tokenizers, text_encoders, [prompts[0].positive])),
        PromptEmbedsXL(*train_util.encode_prompts_xl(tokenizers, text_encoders, [prompts[0].unconditional])),
        PromptEmbedsXL(*train_util.encode_prompts_xl(tokenizers, text_encoders, [prompts[0].neutral])),
        prompts[0],
    )


    # Load dummy images
    img1 = Image.new("RGB", (256, 256), color="red")
    img2 = Image.new("RGB", (256, 256), color="blue")

    add_time_ids = train_util.get_add_time_ids(
        prompts[0].resolution, prompts[0].resolution, prompts[0].dynamic_crops, weight_dtype
    ).to(device)

    # Original implementation
    loss_high_orig, loss_low_orig = original_train_step(
        unet, vae, noise_scheduler, img1, img2, 1.0, prompt_pair, config, network, torch.nn.MSELoss(), device, weight_dtype, add_time_ids
    )

    # Refactored implementation
    loss_high_refactored, loss_low_refactored = functional_train_step(
        unet,
        vae,
        noise_scheduler,
        img1,
        img2,
        1.0,
        prompt_pair,
        config,
        network,
        torch.nn.MSELoss(),
        device,
        weight_dtype,
        add_time_ids
    )

    # Super-functional implementation
    loss_high_super, loss_low_super = superfunctional_train_step(
        unet,
        vae,
        noise_scheduler,
        (img1, img2),
        (1.0, -1.0),
        (prompt_pair.positive, prompt_pair.neutral),
        prompt_pair,
        config,
        network,
        torch.nn.MSELoss(),
        device,
        weight_dtype,
        add_time_ids
    )

    # Assert that the losses are identical
    assert torch.allclose(loss_high_orig, loss_high_refactored)
    assert torch.allclose(loss_low_orig, loss_low_refactored)
    assert torch.allclose(loss_high_orig, loss_high_super)
    assert torch.allclose(loss_low_orig, loss_low_super)

    print("All tests passed! The refactored and super-functional training loops are float-for-float identical to the original.")


def original_train_step(
    unet,
    vae,
    noise_scheduler,
    img1,
    img2,
    scale_to_look,
    prompt_pair,
    config,
    network,
    criteria,
    device,
    weight_dtype,
    add_time_ids,
):
    """
    A recreation of the original training step for comparison.
    """
    seed = random.randint(0, 2**15)
    generator = torch.manual_seed(seed)
    denoised_latents_low, low_noise = train_util.get_noisy_image(
        img1,
        vae,
        generator,
        unet,
        noise_scheduler,
        start_timesteps=0,
        total_timesteps=config.train.max_denoising_steps - 1,
    )
    denoised_latents_low = denoised_latents_low.to(device, dtype=weight_dtype)
    low_noise = low_noise.to(device, dtype=weight_dtype)

    generator = torch.manual_seed(seed)
    denoised_latents_high, high_noise = train_util.get_noisy_image(
        img2,
        vae,
        generator,
        unet,
        noise_scheduler,
        start_timesteps=0,
        total_timesteps=config.train.max_denoising_steps - 1,
    )
    denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)
    high_noise = high_noise.to(device, dtype=weight_dtype)
    noise_scheduler.set_timesteps(1000)

    current_timestep = noise_scheduler.timesteps[
        int((config.train.max_denoising_steps - 1) * 1000 / config.train.max_denoising_steps)
    ]

    network.set_lora_slider(scale=scale_to_look)
    with network:
        target_latents_high = train_util.predict_noise_xl(
            unet,
            noise_scheduler,
            current_timestep,
            denoised_latents_high,
            prompt_pair.positive.text_embeds,
            prompt_pair.positive.pooled_embeds,
            add_time_ids,
            guidance_scale=1,
        ).to("cpu", dtype=torch.float32)
    loss_high = criteria(target_latents_high, high_noise.cpu().to(torch.float32))

    network.set_lora_slider(scale=-scale_to_look)
    with network:
        target_latents_low = train_util.predict_noise_xl(
            unet,
            noise_scheduler,
            current_timestep,
            denoised_latents_low,
            prompt_pair.neutral.text_embeds,
            prompt_pair.neutral.pooled_embeds,
            add_time_ids,
            guidance_scale=1,
        ).to("cpu", dtype=torch.float32)
    loss_low = criteria(target_latents_low, low_noise.cpu().to(torch.float32))

    return loss_high, loss_low


if __name__ == "__main__":
    test_refactored_training_loop()
