"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor
from torch import nn
from torchmetrics.multimodal import CLIPScore
import torchvision
import clip
import matplotlib.pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/16", device=device)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)


def crop_image(img, bbox):
    
    x,y,h,w = bbox[0], bbox[1], bbox[2], bbox[3]
    cropped_region = img[:,:, y:y+h, x:x+w]
    return torchvision.transforms.Resize((512,512))(cropped_region)

def get_tensor_clip(normalize=False, toTensor=False):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
     
    return torchvision.transforms.Compose(transform_list)

def get_clip_metric(torch_image,bbox, target_text="A graceful white cat gracefully stretches, showing off its fluffy, pristine fur."):

    target_region = crop_image(torch_image, bbox)
    print(target_region.shape)
    score = clip_metric(target_region.to(torch.uint8), target_text)
    return (1-(score/100))

def cosine_loss(image_tensor,bbox=None, text="A graceful and elegant white cat stretches leisurely, showcasing its pristine and fluffy fur."
            ):
    text = clip.tokenize([text]).to(device)
    x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
    cropped_region = image_tensor[:,:, y:y+h, x:x+w]

    image_features = model_clip.encode_image(torchvision.transforms.Resize((224,224))(cropped_region))
    text_features = model_clip.encode_text(text)
    cos_sim = nn.CosineSimilarity()(image_features,text_features)
    return (1-cos_sim)

# def cosine_loss_masked(image_tensor,bbox=None, text="A graceful and elegant white cat stretches leisurely, showcasing its pristine and fluffy fur."
#             ):
#     text = clip.tokenize([text]).to(device)
#     # with torch.no_grad():
#     image_features = model_clip.encode_image(torchvision.transforms.Resize((224,224))(cropped_region))
#     text_features = model_clip.encode_text(text)
#     cos_sim = nn.CosineSimilarity()(image_features,text_features)
#     return (1-cos_sim)


def cosine_imgtoimg_loss(src_tensor, ref_tensor,bbox=None
            ):
    
    x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
    cropped_region = src_tensor[:,:, y:y+h, x:x+w]

    crop_features = model_clip.encode_image(torchvision.transforms.Resize((224,224))(cropped_region))
    ref_features = model_clip.encode_image(torchvision.transforms.Resize((224,224))(ref_tensor))
    cos_sim = nn.CosineSimilarity()(crop_features,ref_features)
    return (1-cos_sim)

def bg_loss(t_1,t_2 
            ):    
    t_1 = torch.clamp(t_1, min=-1.0, max=1.0)
    t_2 = torch.clamp(t_2, min=-1.0, max=1.0)
    return (nn.MSELoss()(t_1,t_2) + lpips(t_1,t_2))/2.

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def sample_separate(self,
               S,
               batch_size,
               shape,
               gt_image=None,
               bbox=None,
               text_desc=None,
               src_img=None,
               binary_mask=None,
               conditioning=None,
               guidance=True,
               optim_steps=1,
               guidance_wt=200,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs):
        
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        
#         for param in self.model.first_stage_model.parameters():
#             param.requires_grad = False

        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        device = self.model.betas.device
        b = shape[0]

        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T
        #img = img.requires_grad_(True)
        #img = img.detach().requires_grad_(True)
        timesteps=None
        ddim_use_original_steps=False
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        c=conditioning
        for i, step in enumerate(iterator):

            

            #torch.set_grad_enabled(True)
            index = total_steps - i - 1
            ts = torch.full((size[0],), step, device=device, dtype=torch.long)

            
            for j in range(optim_steps):
                #mask=None
                torch.set_grad_enabled(True)
                if mask is not None:
                    assert x0 is not None
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                    # img_orig=img_orig.requires_grad_(True)
                    img = img_orig * mask + (1. - mask) * img
                b, *_, device = *img.shape, img.device
                use_original_steps=ddim_use_original_steps
                alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
                alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
                sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
                sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
                # select parameters corresponding to the currently considered timestep
                a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
                a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
                sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
                sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
                beta_t = a_t / a_prev

                if 'test_model_kwargs' in kwargs:
                    kwargs1=kwargs['test_model_kwargs']
                    x = torch.cat([img, kwargs1['inpaint_image'], kwargs1['inpaint_mask']],dim=1)
                elif 'rest' in kwargs:
                    x = torch.cat((img, kwargs['rest']), dim=1)
                else:
                    raise Exception("kwargs must contain either 'test_model_kwargs' or 'rest' key")

                if guidance:
                    with torch.enable_grad():

                        img_in = x.detach().requires_grad_(True)

                        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                            e_t = self.model.apply_model(img_in, ts, c)
                        else:
                            x_in = torch.cat([img_in] * 2)
                            t_in = torch.cat([ts] * 2)
                            c_in = torch.cat([unconditional_conditioning, c])
                            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

                        if score_corrector is not None:
                            assert self.model.parameterization == "eps"
                            e_t = score_corrector.modify_score(self.model, e_t, img_in, ts, c, **corrector_kwargs)


                        quantize_denoised=False
                        # current prediction for x_0
                        if img_in.shape[1]!=4:
                            pred_x0 = (img_in[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
                            #pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
                        else:
                            pred_x0 = (img_in - sqrt_one_minus_at * e_t) / a_t.sqrt()
                        if quantize_denoised:
                            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

                        #EXTERNAL GUIDANCE

                        recons_image = self.model.differentiable_decode_first_stage(pred_x0)
                        optim_guidance_3_wt = guidance_wt #200

                        loss = (cosine_loss(recons_image*binary_mask.cuda(),bbox=bbox, text=text_desc)) + 0.5*cosine_imgtoimg_loss(recons_image*binary_mask.cuda(),gt_image.cuda(),bbox=bbox) 
                        #optionally you can add the background preservation loss below
                        #+ 10*bg_loss(recons_image*(1-binary_mask.cuda()),src_img.cuda()*(1-binary_mask.cuda()))
                        loss= -1 * loss
                        grad = torch.autograd.grad(loss.sum(), img_in)[0]
                        grad = grad * optim_guidance_3_wt
                        e_t = e_t - sqrt_one_minus_at * grad[:,:4,:,:].detach()

                        repeat_noise=False
                    with torch.no_grad():
                        # direction pointing to x_t
                        pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()

                        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
                        noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
                        if noise_dropout > 0.:
                            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
                        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    #                     img = beta_t.sqrt() * x_prev * kwargs1['inpaint_mask'] + (1 - beta_t).sqrt() * noise_like(img.shape, device, False) * (1-kwargs1['inpaint_mask'])
                        img = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(img.shape, device, False)

                else:
                    with torch.no_grad():
                            # direction pointing to x_t
                        img_in = x

                        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                            e_t = self.model.apply_model(img_in, ts, c)
                        else:
                            x_in = torch.cat([img_in] * 2)
                            t_in = torch.cat([ts] * 2)
                            c_in = torch.cat([unconditional_conditioning, c])
                            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

                        if score_corrector is not None:
                            assert self.model.parameterization == "eps"
                            e_t = score_corrector.modify_score(self.model, e_t, img_in, ts, c, **corrector_kwargs)

                        repeat_noise=False
                        quantize_denoised=False
                        # current prediction for x_0
                        if img_in.shape[1]!=4:
                            pred_x0 = (img_in[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
                            #pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
                        else:
                            pred_x0 = (img_in - sqrt_one_minus_at * e_t) / a_t.sqrt()
                        if quantize_denoised:
                            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
                        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
                        noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
                        if noise_dropout > 0.:
                            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
                        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                        img = x_prev
    
    
            img, pred_x0 = x_prev, pred_x0
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

        

    #@torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    **kwargs
                                                    )
        return samples, intermediates

    #@torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,**kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,**kwargs)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    #@torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,**kwargs):
        b, *_, device = *x.shape, x.device
        if 'test_model_kwargs' in kwargs:
            kwargs=kwargs['test_model_kwargs']
            x = torch.cat([x, kwargs['inpaint_image'], kwargs['inpaint_mask']],dim=1)
        elif 'rest' in kwargs:
            x = torch.cat((x, kwargs['rest']), dim=1)
        else:
            raise Exception("kwargs must contain either 'test_model_kwargs' or 'rest' key")
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if x.shape[1]!=4:
            pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

#    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

#    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print("printing this one . . . . . ")
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec