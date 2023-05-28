import inspect
import os

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler, PNDMScheduler, DDIMScheduler
from omegaconf import DictConfig
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel

from fid_utils import calculate_fid_given_features
from models.blip_override.blip import blip_feature_extractor, init_tokenizer
from models.diffusers_override.unet_2d_condition import UNet2DConditionModel
from models.inception import InceptionV3


class LightningDataset(pl.LightningDataModule):
    def __init__(self, args: DictConfig):
        super(LightningDataset, self).__init__()
        self.kwargs = {"num_workers": args.num_workers, "persistent_workers": True if args.num_workers > 0 else False,
                       "pin_memory": True}
        self.args = args

    def setup(self, stage="fit"):
        if self.args.dataset == "pororo":
            import datasets.pororo as data
        elif self.args.dataset == 'flintstones':
            import datasets.flintstones as data
        elif self.args.dataset == 'vistsis':
            import datasets.vistsis as data
        elif self.args.dataset == 'vistdii':
            import datasets.vistdii as data
        else:
            raise ValueError("Unknown dataset: {}".format(self.args.dataset))
        if stage == "fit":
            self.train_data = data.ImageDataset("train", self.args)
            self.val_data = data.ImageDataset("val", self.args)
        if stage == "test":
            self.test_data = data.ImageDataset("test", self.args)

    def train_dataloader(self):
        if not hasattr(self, 'trainloader'):
            self.trainloader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        return self.trainloader

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def get_length_of_train_dataloader(self):
        if not hasattr(self, 'trainloader'):
            self.trainloader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        return len(self.trainloader)


class ARLDM(pl.LightningModule):
    def __init__(self, args: DictConfig, steps_per_epoch=1):
        super(ARLDM, self).__init__()
        self.args = args
        self.steps_per_epoch = steps_per_epoch
        """
            Configurations
        """
        self.task = args.task

        if args.mode == 'sample':
            if args.scheduler == "pndm":
                self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               skip_prk_steps=True)
            elif args.scheduler == "ddim":
                self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               clip_sample=False, set_alpha_to_one=True)
            else:
                raise ValueError("Scheduler not supported")
            self.fid_augment = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.inception = InceptionV3([block_idx])

        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        
        self.max_length = args.get(args.dataset).max_length

        clip_text_null_token = self.clip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids
        

        self.register_buffer('clip_text_null_token', clip_text_null_token)
        
        self.text_encoder = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                          subfolder="text_encoder")
        self.text_encoder.resize_token_embeddings(args.get(args.dataset).clip_embedding_tokens)
        # resize_position_embeddings
        old_embeddings = self.text_encoder.text_model.embeddings.position_embedding
        new_embeddings = self.text_encoder._get_resized_embeddings(old_embeddings, self.max_length)
        self.text_encoder.text_model.embeddings.position_embedding = new_embeddings
        self.text_encoder.config.max_position_embeddings = self.max_length
        self.text_encoder.max_position_embeddings = self.max_length
        self.text_encoder.text_model.embeddings.position_ids = torch.arange(self.max_length).expand((1, -1))

        # self.modal_type_embeddings = nn.Embedding(2, 768)
        # self.time_embeddings = nn.Embedding(5, 768)
        
        self.vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet")
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                             num_train_timesteps=1000)

        # Freeze vae and unet
        self.freeze_params(self.vae.parameters())
        if args.freeze_resnet:
            self.freeze_params([p for n, p in self.unet.named_parameters() if "attentions" not in n])

        if args.freeze_clip and hasattr(self, "text_encoder"):
            self.freeze_params(self.text_encoder.parameters())
            self.unfreeze_params(self.text_encoder.text_model.embeddings.token_embedding.parameters())

    @staticmethod
    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    @staticmethod
    def unfreeze_params(params):
        for param in params:
            param.requires_grad = True

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=self.args.warmup_epochs * self.steps_per_epoch,
                                                  max_epochs=self.args.max_epochs * self.steps_per_epoch)
        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,  # The LR scheduler instance (required)
                'interval': 'step',  # The unit of the scheduler's step size
            }
        }
        return optim_dict

    def forward(self, batch):
        if self.args.freeze_clip and hasattr(self, "text_encoder"):
            self.text_encoder.eval()
        
        images, captions, attention_mask  = batch
        B = captions.shape[0]
        # S = captions.shape[-1]
        captions = torch.flatten(captions, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)
        # 1 is not masked, 0 is masked

        classifier_free_idx = np.random.rand(B) < 0.1

        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state  # B * V, S, D
        
        caption_embeddings[classifier_free_idx] = self.text_encoder(self.clip_text_null_token).last_hidden_state[0]
        
        encoder_hidden_states = caption_embeddings
        
        attention_mask = ~(attention_mask.bool())
        attention_mask[classifier_free_idx] = False

        
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215

        noise = torch.randn(latents.shape, device=self.device)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, attention_mask).sample
        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
        return loss

    def sample(self, batch):
        original_images, captions, attention_mask = batch
        B, S = captions.shape[0],captions.shape[-1]
        
        captions = torch.flatten(captions, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)
        
        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state
        encoder_hidden_states = caption_embeddings #B, S, D

        attention_mask = torch.cat([attention_mask], dim=1)
        attention_mask = ~(attention_mask.bool()) # B, 2 * S
        # attention_mask[:, -S:] = 1

        uncond_caption_embeddings = self.text_encoder(self.clip_text_null_token).last_hidden_state
        uncond_embeddings = uncond_caption_embeddings #B, S, D
        # uncond_embeddings = uncond_embeddings.expand(B * V, -1, -1)

        encoder_hidden_states = torch.cat([uncond_embeddings]*B + [encoder_hidden_states])
        uncond_attention_mask = torch.zeros((B, S), device=self.device).bool()
        attention_mask = torch.cat([uncond_attention_mask, attention_mask], dim=0)

        # attention_mask = attention_mask.reshape(2, B, 2 * S)
        
        # encoder_hidden_states = encoder_hidden_states.reshape(2, B, 2 * S, -1)

        new_image = self.diffusion(encoder_hidden_states,
                                    attention_mask,
                                    512, 512, self.args.num_inference_steps, self.args.guidance_scale, 0.0)

        return original_images, new_image

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('loss/train_loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('loss/val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        original_images, images = self.sample(batch)
        if self.args.calculate_fid:
            original_images = original_images.cpu().numpy().astype('uint8')
            original_images = [Image.fromarray(im, 'RGB') for im in original_images]
            ori = self.inception_feature(original_images).cpu().numpy()
            gen = self.inception_feature(images).cpu().numpy()
        else:
            ori = None
            gen = None
        return images, ori, gen

    def diffusion(self, encoder_hidden_states, attention_mask, height, width, num_inference_steps, guidance_scale, eta):
        latents = torch.randn((encoder_hidden_states.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                              device=self.device)

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states).sample
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states, attention_mask).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return self.numpy_to_pil(image)

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image, 'RGB') for image in images]

        return pil_images

    def inception_feature(self, images):
        images = torch.stack([self.fid_augment(image) for image in images])
        images = images.type(torch.FloatTensor).to(self.device)
        images = (images + 1) / 2
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        pred = self.inception(images)[0]

        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred.reshape(-1, 2048)


def train(args: DictConfig) -> None:
    dataloader = LightningDataset(args)
    dataloader.setup('fit')
    model = ARLDM(args, steps_per_epoch=dataloader.get_length_of_train_dataloader())

    logger = TensorBoardLogger(save_dir=os.path.join(args.ckpt_dir, args.run_name), name='log', default_hp_metric=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.ckpt_dir, args.run_name),
        save_top_k=0,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callback_list = [lr_monitor, checkpoint_callback]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu_ids,
        max_epochs=args.max_epochs,
        benchmark=True,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callback_list,
        strategy=DDPStrategy(find_unused_parameters=False)
    )
    trainer.fit(model, dataloader, ckpt_path=args.train_model_file)


def sample(args: DictConfig) -> None:
    assert args.test_model_file is not None, "test_model_file cannot be None"
    assert args.gpu_ids == 1 or len(args.gpu_ids) == 1, "Only one GPU is supported in test mode"
    dataloader = LightningDataset(args)
    dataloader.setup('test')
    model = ARLDM.load_from_checkpoint(args.test_model_file, args=args, strict=False)

    predictor = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu_ids,
        max_epochs=-1,
        benchmark=True
    )
    predictions = predictor.predict(model, dataloader)
    images = [elem for sublist in predictions for elem in sublist[0]]
    if not os.path.exists(args.sample_output_dir):
        try:
            os.mkdir(args.sample_output_dir)
        except:
            pass
    for i, image in enumerate(images):
        image.save(os.path.join(args.sample_output_dir, '{:04d}.png'.format(i)))

    if args.calculate_fid:
        ori = np.array([elem for sublist in predictions for elem in sublist[1]])
        gen = np.array([elem for sublist in predictions for elem in sublist[2]])
        fid = calculate_fid_given_features(ori, gen)
        print('FID: {}'.format(fid))


@hydra.main(config_path=".", config_name="config")
def main(args: DictConfig) -> None:
    pl.seed_everything(args.seed)
    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)


if __name__ == '__main__':
    main()
