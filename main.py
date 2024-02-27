import numpy as np
import argparse
import ast
import operator
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from utils.parse import filter_boxes
from generation import run as run_layout_to_image
from baseline import run as run_baseline
import torch
from shared import DEFAULT_SO_NEGATIVE_PROMPT, DEFAULT_OVERALL_NEGATIVE_PROMPT
#from segment_anything.segment_anything import build_sam, SamPredictor 
from PIL import Image
print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
import openai
# from diffusers import PaintByExamplePipeline
from transformers import GPT2TokenizerFast
# from paint_by_example.my_paint_by_example import main_paint_by_example
import torch.nn.functional as F
from torchmetrics.multimodal import CLIPScore
from helper_functions import *
from composition_module.my_paint_by_example import *
import backoff 
import re
import time
from diffusers import StableDiffusionImg2ImgPipeline
from prompt_templates import *

device = "cuda"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=False
).to(device)


box_scale = (512, 512)
size = box_scale
clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").cuda()
repo_id = "stabilityai/stable-diffusion-2-base"
sd_pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")

sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
sd_pipe = sd_pipe.to("cuda")
bg_prompt_text = "Background prompt: "

openai.api_key = ""

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# GPT-3 Type
gpt_name = {
    'gpt3.5': 'text-davinci-003',
    'gpt3.5-chat': 'gpt-3.5-turbo',
    'gpt4': 'gpt-4',
}

# Caption: In the quiet countryside, a red farmhouse stands with an old-fashioned charm. Nearby, a weathered picket fence surrounds a garden of wildflowers. An antique tractor, though worn, rests as a reminder of hard work. A scarecrow watches over fields of swaying crops. The air carries the scent of earth and hay. Set against rolling hills, this farmhouse tells a story of connection to the land and its traditions
# Objects: [('a red farmhouse', [105, 228, 302, 245]), ('a weathered picket fence', [4, 385, 504, 112]), ('an antique tractor', [28, 382, 157, 72]), ('a scarecrow', [368, 271, 66, 156]) ]
# Background prompt: A realistic image of a a quiet countryside with rolling hills





def get_lmd_prompt(prompt, template=default_template):
    if prompt == "":
        prompt = prompt_placeholder
    if template == "":
        template = default_template
    return simplified_prompt.format(template=template, prompt=prompt)

def get_layout_image(response):
    if response == "":
        response = layout_placeholder
    gen_boxes, bg_prompt = parse_input(response)
    fig = plt.figure(figsize=(8, 8))
    # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
    show_boxes(gen_boxes, bg_prompt)
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    return data

def get_layout_image_gallery(response):
    return [get_layout_image(response)]

def get_layout_to_image(response, overall_prompt_override="", seed=0, num_inference_steps=20, dpm_scheduler=True, use_autocast=False, fg_seed_start=20, fg_blending_ratio=0.1, frozen_step_ratio=0.4, gligen_scheduled_sampling_beta=0.3, so_negative_prompt=DEFAULT_SO_NEGATIVE_PROMPT, overall_negative_prompt=DEFAULT_OVERALL_NEGATIVE_PROMPT, show_so_imgs=False, scale_boxes=False):
    if response == "":
        response = layout_placeholder
    gen_boxes, bg_prompt = parse_input(response)
    gen_boxes = filter_boxes(gen_boxes, scale_boxes=scale_boxes)
    spec = {
        # prompt is unused
        'prompt': '',
        'gen_boxes': gen_boxes,
        'bg_prompt': bg_prompt
    }
    print(spec)
    if dpm_scheduler:
        scheduler_key = "dpm_scheduler"
    else:
        scheduler_key = "scheduler"
        
    image_np, so_img_list = run_layout_to_image(
        spec, bg_seed=seed, overall_prompt_override=overall_prompt_override, fg_seed_start=fg_seed_start, 
        fg_blending_ratio=fg_blending_ratio,frozen_step_ratio=frozen_step_ratio, use_autocast=use_autocast,
        gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta, num_inference_steps=num_inference_steps, scheduler_key=scheduler_key,
        so_negative_prompt=so_negative_prompt, overall_negative_prompt=overall_negative_prompt, so_batch_size=1
    )
    images = [image_np]
    if show_so_imgs:
        images.extend([np.asarray(so_img) for so_img in so_img_list])
    return images


def parse_input(text=None):
    try:
        if "Objects: " in text:
            text = text.split("Objects: ")[1]

        text_split = text.split(bg_prompt_text)
        if len(text_split) == 2:
            gen_boxes, bg_prompt = text_split
        elif len(text_split) >= 2:
            gen_boxes, _, bg_prompt = text_split
        gen_boxes = ast.literal_eval(gen_boxes)    
        bg_prompt = bg_prompt.strip()
    except Exception as e:
        raise gr.Error(f"response format invalid: {e} (text: {text})")
    
    return gen_boxes, bg_prompt

def draw_boxes(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4)
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h],
                [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)

        # print(ann)
        name = ann['name'] if 'name' in ann else str(ann['category_id'])
        ax.text(bbox_x, bbox_y, name, style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

    p = PatchCollection(polygons, facecolor='none',
                        edgecolors=color, linewidths=2)
    ax.add_collection(p)


def show_boxes(gen_boxes, bg_prompt=None):
    anns = [{'name': gen_box[0], 'bbox': gen_box[1]}
            for gen_box in gen_boxes]

    # White background (to allow line to show on the edge)
    I = np.ones((size[0]+4, size[1]+4, 3), dtype=np.uint8) * 255

    plt.imshow(I)
    plt.axis('off')

    if bg_prompt is not None:
        ax = plt.gca()
        ax.text(0, 0, bg_prompt, style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

        c = np.zeros((1, 3))
        [bbox_x, bbox_y, bbox_w, bbox_h] = (0, 0, size[1], size[0])
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h],
                [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons = [Polygon(np_poly)]
        color = [c]
        p = PatchCollection(polygons, facecolor='none',
                            edgecolors=color, linewidths=2)
        ax.add_collection(p)

    draw_boxes(anns)
    
    
def segment(image, sam_model, boxes=None,point_coords=None,point_labels=None):
  sam_model.set_image(image)
  H, W, _ = image.shape
#   boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

#   transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
  masks, _, _ = sam_model.predict(
      point_coords = point_coords,
      point_labels = point_labels,
      box = boxes,
      multimask_output = False,
      )
  return masks
  

def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    
def generate_image(image, mask, prompt, negative_prompt, pipe, seed):
  # resize for inpainting 
  w, h = image.size
  in_image = image.resize((512, 512))
  in_mask = mask.resize((512, 512))

  generator = torch.Generator(device).manual_seed(seed) 

  result = pipe(image=in_image, mask_image=in_mask, prompt=prompt, negative_prompt=negative_prompt, generator=generator)
  result = result.images[0]

  return result.resize((w, h))


def gen_hq_image_sd(prompt):

    image = sd_pipe(prompt, num_inference_steps=50).images[0]
    return image

def gen_paint_by_example(init_image=None,mask_image=None,example_image=None):

    paint_by_ex_pipe = PaintByExamplePipeline.from_pretrained(
        "Fantasy-Studio/Paint-by-Example",
        torch_dtype=torch.float16,
    )
    paint_by_ex_pipe = paint_by_ex_pipe.to("cuda")

    image = paint_by_ex_pipe(image=init_image, mask_image=mask_image, example_image=example_image).images[0]
    return image  

def get_clip_metric(torch_image, pil_image,bbox, target_text=None):
    #mask = create_square_mask(pil_image,bbox)
    #target_region = crop_with_mask_torch(torch.from_numpy(np.asarray(pil_image)), torch.from_numpy(np.asarray(mask)))
    print("target region shape: ", target_region.shape)
#     plt.imshow(target_region)
#     plt.show()
    score = clip_metric(target_region.cuda(), target_text)
    return score/100.


def remove_newlines(input_string):
    # Define a regular expression pattern to match the object-bbox pairs
    pattern = r"\('[^']*'\s*,\s*\[[^\]]*\]\)"

    # Find all matches of the pattern in the input string
    matches = re.findall(pattern, input_string)

    # Replace "\n" after each match with an empty string
    for match in matches:
        input_string = input_string.replace(match + '\n', match)

    return input_string




# num_layouts = 1
# layout_seed=42
# layout_inference_steps=20


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_first_stage_generation(text_prompt,num_layouts=1,num_inference_steps=20,seed=42):
    img_list = []
    response_list = []
    count=0
    num_layouts=1
    if type(text_prompt)==str:
        while count < num_layouts:
                try:
                    form_prompt = f'\nCaption: {text_prompt} \n Objects:'
                    response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                            {"role": "system", "content": default_template},
                            {"role": "user", "content": form_prompt}
                        ]
                    )
                    count+=1
                except:
                    continue
        response = response_list
    else:
        #use default response 
        keypoint_sample = [('a Golden Retriever', [97, 280, 198, 132]), ('a white cat', [60, 240, 125, 76]), ('a wooden table', [184, 163, 144, 86]), ('a vase of vibrant flowers', [199, 67, 55, 110]), ('a sleek modern television', [391, 21, 101, 63])]
        response = [str(keypoint_sample) + "\n" + "Background prompt: In a cozy living room filled with a sense of companionship and relaxation"]
            
    gen_im = get_layout_to_image(response[0], seed=seed,
                          )

    return gen_im,response[0]


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def first_stage_gen_with_interpolation(text_prompt,num_layouts=3,num_inference_steps=20,seed=42):

    img_list = []
    response_list = []
    count=0
    num_layouts=3
    if type(text_prompt)==str:
        while count < num_layouts:
                try:
                    form_prompt = f'\nCaption: {text_prompt} \n Objects:'
                    response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                            {"role": "system", "content": default_template},
                            {"role": "user", "content": form_prompt}
                        ]
                    )
                    # print(response['choices'])
                    time.sleep(3)
                    if "Background prompt" in response['choices'][0]['message']['content']:
                        response_list.append(remove_newlines(response['choices'][0]['message']['content']))
                        #print(remove_newlines(response['choices'][0]['message']['content']))
                        count+=1
                except:
                    continue
        response = get_avg_boxes_with_bg(response_list)
    else:
        #use default response 
        response =  text_prompt
        # keypoint_sample = [('a Golden Retriever', [97, 280, 198, 132]), ('a white cat', [60, 240, 125, 76]), ('a wooden table', [184, 163, 144, 86]), ('a vase of vibrant flowers', [199, 67, 55, 110]), ('a sleek modern television', [391, 21, 101, 63])]
        # response = [str(keypoint_sample) + "\n" + "Background prompt: In a cozy living room filled with a sense of companionship and relaxation"]
            
    gen_im = get_layout_to_image(response[0], seed=seed,
                          )

    
    return gen_im,response[0]

    
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def gen_obj_descriptions(response, default_description=None):
    
    fail_count=0
    while fail_count<2:
        try:
            list_obj=list(dict(ast.literal_eval(response.split("\n")[0])).keys())
            form_prompt = f'\nlist of the objects: {list_obj} \n text prompt:{text_prompt}'
            response_objs = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                                {"role": "system", "content": extract_obj_prompt},
                                {"role": "user", "content": form_prompt}
                                                    ]
                                                    )
            return response_objs
           
        except:
            fail_count+=1
            continue



def get_clip_metric(torch_image,bbox, target_text="A graceful white cat gracefully stretches, showing off its fluffy, pristine fur."):
        
    target_region = crop_image(torch_image, bbox)
    if target_region.max()<100:
        target_region = (target_region*(255)).to(torch.uint8)
    score = clip_metric(target_region, target_text)
    return score/100

def crop_image(img, bbox):
    if img.shape[0]!=3:
        img=img.permute(2,0,1)
    x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
    cropped_region = img[:,y:y+h,x:x+w] #img[:,x:x+w,y:y+h]
    return cropped_region #torchvision.transforms.Resize((512,512))(cropped_region)

def iterative_refinement(first_stage_gen, objects_dict_bboxes,objects_dict_desc, add_guidance=True, optim_steps=1,guidance_weight=200,skip_small=False, ckpt_path=None):
    shift_flag=False
    
    for obj, bbox in objects_dict_bboxes.items():
        print(obj, objects_dict_desc.keys())
        if obj in objects_dict_desc.keys():
            print("entering ------")
            object_desc = objects_dict_desc[obj]
            if not shift_flag:
                torch_image = first_stage_gen
                pil_image = torchvision.transforms.ToPILImage()(first_stage_gen.permute(2,0,1))

            
            clip_score = get_clip_metric(torch_image.cuda(),tuple(bbox), target_text=object_desc)
            print(clip_score)
            if clip_score < 0.25:
                # if skip_small:
                    # if (bbox[-1]*bbox[-2])/(512*512)>0.01:
                        shift_flag=True
                        ref_image = gen_hq_image_sd(object_desc)
                        image_mask_pil = create_square_mask(pil_image,tuple(bbox))
                        p_by_ex, torch_ex = main_paint_by_example(img_p=pil_image, ref_p=ref_image, mask=image_mask_pil,
                                                                    bbox=bbox,text_desc=object_desc,
                                                                    add_guidance=add_guidance, optim_steps=optim_steps, guidance_weight=guidance_weight, ckpt_path=ckpt_path)
                        torch_image=torch_ex
                        pil_image=p_by_ex
    return pil_image,torch_image



if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="./configs/livingroom_1.yaml")
    args = parser.parse_args()


    conf = OmegaConf.load(args.config)
    os.makedirs(conf.out_dir, exist_ok=True)

    ############################ first stage generation ###########################################
    text_prompt = conf.prompt_info.text_prompt
    bg_prompt = conf.prompt_info.bg_prompt
    layout =  ast.literal_eval(conf.prompt_info.keypoints)
    object_descs = ast.literal_eval(conf.prompt_info.obj_descs)
    object_descs  = {key: value[0] for key, value in object_descs}
    print(object_descs)
    first_stage_image, response = first_stage_gen_with_interpolation([str(layout) + "\n" + bg_prompt_text + bg_prompt],num_layouts=conf.first_stage_gen_config.num_layouts,num_inference_steps=conf.first_stage_gen_config.diffusion_steps,seed=conf.first_stage_gen_config.seed)
    
    if not conf.second_stage_gen_config.enable:
        Image.fromarray(first_stage_image[0]).save(os.path.join(conf.out_dir, conf.exp_name + "_first_stage.png"))
        print("generation finished . . . check {} directory for outputs".format(conf.out_dir))
        exit()


    ########################### second stage generation ###########################################

    
    objects_dict_bboxes = dict(ast.literal_eval(response.split("\n")[0]))
    sorted_bbox_dict = dict(sorted(objects_dict_bboxes.items(), key=lambda item: item[1][2] * item[1][3], reverse=True))

    #optional 
    #generator = torch.Generator(device=device).manual_seed(conf.second_stage_gen_config.obj_descs)
    #first_phase_pil = Image.fromarray(first_stage_gen_dict[0])
    #first_stage_image = pipe(prompt=text, image=first_phase_pil, strength=0.75, guidance_scale=7.5, generator=generator).images[0]

    first_phase_pil = Image.fromarray(first_stage_image[0])

    pil_img, _ = iterative_refinement(torchvision.transforms.ToTensor()(first_phase_pil).permute(1,2,0), sorted_bbox_dict,object_descs, add_guidance=conf.second_stage_gen_config.add_guidance,
                    optim_steps=conf.second_stage_gen_config.optim_steps,guidance_weight=conf.second_stage_gen_config.guidance_weight,skip_small=conf.second_stage_gen_config.skip_small,ckpt_path=conf.composition_model_path )

    pil_img.save(os.path.join(conf.out_dir,conf.exp_name +  "_second_stage.png"))
    first_phase_pil.save(os.path.join(conf.out_dir, conf.exp_name + "_first_stage.png"))
    print("generation finished . . . check {} directory for outputs".format(conf.out_dir))