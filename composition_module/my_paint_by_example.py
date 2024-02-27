import argparse, os, sys, glob
sys.path.append(os.path.join(os.getcwd(),"composition_module"))
#sys.path.append("composition_module")
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from tqdm import tqdm, trange
#from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torchmetrics.multimodal import CLIPScore
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import clip
from torch import optim
from torchvision.transforms import Resize
from helper import OptimizerDetails
wm = "Paint-by-Example"
# wm_encoder = WatermarkEncoder()
# wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def crop_image(img, bbox):
    x,y,h,w = bbox[0], bbox[1], bbox[2], bbox[3]
    cropped_region = img[:, y:y+h, x:x+w]
    return cropped_region



def get_optimization_details():
    
    clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").cuda()
    operation_func=clip_metric
    

    operation = OptimizerDetails()

    operation.num_steps = [1]
    operation.operation_func = crop_image

    operation.optimizer = 'Adam'
    operation.lr = 1e-2
    operation.loss_func = nn.L1Loss()

    operation.max_iters = 1
    operation.loss_cutoff = 0.00001

    operation.guidance_3 = True
    operation.optim_guidance_3_wt = 5.0
    operation.guidance_2 = True
    operation.original_guidance = False

    operation.warm_start = False
    operation.print = True
    operation.print_every = 5

    return operation

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)
  
    


@torch.enable_grad()
def main_paint_by_example(img_p=None, ref_p=None, mask=None,gt_image=None, bbox=None,text_desc=None,add_guidance=True,optim_steps=1,guidance_weight=200, ckpt_path=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=100,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/share/data/drive_2/Hanan/llmblueprint/composition_module/configs/v1.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/share/data/drive_1/hanan/llm_blueprint/paint_by_example/checkpoints/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="evaluate at this precision",
        default="/home/ganimh/llm-grounded-diffusion/paint_by_example/examples/image/example_1.png"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        help="evaluate at this precision",
        default="/home/ganimh/llm-grounded-diffusion/paint_by_example/examples/mask/example_1.png"
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        help="evaluate at this precision",
        default="/home/ganimh/llm-grounded-diffusion/paint_by_example/examples/reference/example_1.jpg"
    )
    opt = parser.parse_args("")


    seed_everything(opt.seed)
    config_path = os.path.join(os.getcwd(),"composition_module","configs/v1.yaml")
    config = OmegaConf.load(config_path)   #f"{opt.config}")
    if ckpt_path:
    	model = load_model_from_config(config, ckpt_path)
    else:
    	model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.requires_grad_(True)
    model.eval()

    if opt.plms:
        print("plms sampler ....")
        sampler = PLMSSampler(model)
    else:
        print("ddim sampler ....")

        sampler = DDIMSampler(model)

#     os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    operation = get_optimization_details()

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
  

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    
    precision_scope = autocast if opt.precision=="autocast" else nullcontext

    image_tensor = get_tensor()(img_p)
    image_tensor = image_tensor.unsqueeze(0)
#                 ref_p = Image.open(opt.reference_path).convert("RGB").resize((224,224))
    
    ref_pp = ref_p.convert("RGB").resize((224,224))
    ref_tensor=get_tensor_clip()(ref_pp)
    ref_tensor = ref_tensor.unsqueeze(0)
    gt_tensor = get_tensor_clip()(ref_p.convert("RGB").resize((512,512))).unsqueeze(0)
    #mask=Image.open(opt.mask_path).convert("L")
    mask = mask.convert("L")
    mask = np.array(mask)[None,None]
    mask = 1 - mask.astype(np.float32)/255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask_tensor = torch.from_numpy(mask)
    inpaint_image = image_tensor*mask_tensor
    test_model_kwargs={}
    test_model_kwargs['inpaint_mask']=mask_tensor.to(device)
    test_model_kwargs['inpaint_image']=inpaint_image.to(device)
    ref_tensor=ref_tensor.to(device)
    uc = None
    if opt.scale != 1.0:
        uc = model.learnable_vector

    #c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
    c = model.get_learned_conditioning(ref_tensor.float())
    c = model.proj_out(c)
    inpaint_mask=test_model_kwargs['inpaint_mask']
    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
    z_inpaint = model.get_first_stage_encoding(z_inpaint) #.detach()

    test_model_kwargs['inpaint_image']=z_inpaint
    test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-2],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    if add_guidance:

        samples_ddim, _ = sampler.sample_separate(S=opt.ddim_steps,gt_image=gt_tensor,bbox=bbox,text_desc=text_desc,src_img=image_tensor,
                                                binary_mask=mask_tensor,
                                            conditioning=c,
                                            guidance=True,
                                            optim_steps=optim_steps,
                                            guidance_wt=guidance_weight,
                                            batch_size=opt.n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=opt.scale,
                                            unconditional_conditioning=uc,
                                            eta=opt.ddim_eta,
                                            x_T=start_code,
                                            test_model_kwargs=test_model_kwargs)
    else:

                    
        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        test_model_kwargs=test_model_kwargs)


    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    for i,x_sample in enumerate(x_samples_ddim):
        x_torch_to_return = x_sample


    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).detach().numpy()


    #x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
    x_checked_image=x_samples_ddim
    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

    def un_norm(x):
        return (x+1.0)/2.0
    def un_norm_clip(x):
        x[0,:,:] = x[0,:,:] * 0.26862954 + 0.48145466
        x[1,:,:] = x[1,:,:] * 0.26130258 + 0.4578275
        x[2,:,:] = x[2,:,:] * 0.27577711 + 0.40821073
        return x

    if not opt.skip_save:
        for i,x_sample in enumerate(x_checked_image_torch):


            all_img=[]
            all_img.append(un_norm(image_tensor[i]).cpu())
            all_img.append(un_norm(inpaint_image[i]).cpu())
            ref_img=ref_tensor
            ref_img=Resize([opt.H, opt.W])(ref_img)
            all_img.append(un_norm_clip(ref_img[i]).cpu())
            all_img.append(x_sample)
            grid = torch.stack(all_img, 0)
            grid = make_grid(grid)
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            img = Image.fromarray(grid.astype(np.uint8))

            x_sample_ = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample_.astype(np.uint8))
            return img, x_torch_to_return