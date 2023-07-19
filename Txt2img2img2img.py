from modules.shared import opts, cmd_opts, state
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images
from modules import paths, shared, modelloader, sd_models
from modules import sd_samplers
from PIL import Image, ImageDraw
import gradio as gr
import modules.scripts as scripts
from random import randint
from skimage.util import random_noise
from gradio.processing_utils import encode_pil_to_base64
import numpy as np
import os.path
from copy import deepcopy
def remap_range(value, minIn, MaxIn, minOut, maxOut):
            if value > MaxIn: value = MaxIn;
            if value < minIn: value = minIn;
            if (MaxIn - minIn) == 0 : return maxOut
            finalValue = ((value - minIn) / (MaxIn - minIn)) * (maxOut - minOut) + minOut;
            return finalValue;

class Script(scripts.Script):
    def title(self):
        return "Txt2img2img2img"

    def ui(self, is_img2img):
        if is_img2img: return

        # samplers list
        img2img_samplers_names = [s.name for s in sd_samplers.samplers_for_img2img]

        # models list
        model_dir = "Stable-diffusion"
        model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
        model_list = modelloader.load_models(model_path=model_path, command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name="v1-5-pruned-emaonly.safetensors", ext_blacklist=[".vae.ckpt", ".vae.safetensors"])
        model_list = [m.split("\\")[-1] for m in model_list]
        model_list.append("Same")

        t2iii_reprocess = gr.Slider(minimum=1, maximum=128, step=1, label='Number of img2img', value=1)
        t2iii_steps = gr.Slider(minimum=1, maximum=120, step=1, label='img2img steps', value=24)
        with gr.Row():
            t2iii_cfg_scale = gr.Slider(minimum=1, maximum=30, step=0.1, label='img2img cfg scale ', value=8.3)
            t2iii_cfg_scale_end = gr.Slider(minimum=0, maximum=30, step=0.1, label='img2img cfg end scale (0=disabled) ', value=0)
        with gr.Row():
            t2iii_denoising_strength = gr.Slider(minimum=0, maximum=1, step=0.01, label='img2img denoising strength ', value=0.42)
            t2iii_patch_end_denoising = gr.Slider(minimum=0, maximum=1,  step=0.01, label='Last img denoising (0=disabled)', value=0)
        t2iii_seed_shift = gr.Slider(minimum=-1, maximum=1000000, step=1, label='img2img new seed+ (-1 for random)', value=1)
        with gr.Row():
            t2iii_patch_upscale   = gr.Checkbox(label='Patch upscale', value=False)
            t2iii_patch_shift     = gr.Checkbox(label='Patch upscale shift grid', value=True)
            t2iii_save_first      = gr.Checkbox(label='Save first image', value=False)
            t2iii_only_last       = gr.Checkbox(label='Only save last img2img', value=True)
            t2iii_face_correction = gr.Checkbox(label='Face correction on all', value=False)
            t2iii_face_correction_last = gr.Checkbox(label='Face correction on last', value=False)
        with gr.Row():
            t2iii_model   = gr.Dropdown(label="Model",   choices=model_list, value="Same")
            t2iii_sampler = gr.Dropdown(label="Sampler", choices=img2img_samplers_names, value="DPM++ 2M")
        # with gr.Row():
        #     t2iii_override_s_noise = gr.Checkbox(label='Override sampler\'s noise for last pass', value=False)
        #     t2iii_sampler_noise    = gr.Slider(minimum=0, maximum=1,  step=0.5, label='Sampler\'s noise for last pass', value=0)
        with gr.Row():
            t2iii_clip    = gr.Slider(minimum=0, maximum=12, step=1, label='change clip for img2img (0 = disabled)', value=0)
            t2iii_noise   = gr.Slider(minimum=0, maximum=0.005,  step=0.0001, label='Add noise before img2img ', value=0)
        with gr.Row():
            t2iii_patch_square_size   = gr.Slider(minimum=64, maximum=2048,  step=64, label='Patch upscale square size', value=512)
            t2iii_patch_padding       = gr.Slider(minimum=0, maximum=512,  step=8, label='Patch upscale padding', value=128)
        with gr.Row():
            t2iii_patch_border        = gr.Slider(minimum=0, maximum=64,   step=1, label='Patch upscale mask inner border', value=8)
            t2iii_patch_mask_blur     = gr.Slider(minimum=0, maximum=64 ,  step=1, label='Patch upscale mask blur', value=4)
        with gr.Row():
            t2iii_upscale_x = gr.Slider(minimum=64, maximum=16384, step=64, label='img2img width (64 = no rescale) ', value=768)
            t2iii_upscale_y = gr.Slider(minimum=64, maximum=16384, step=64, label='img2img height (64 = no rescale) ', value=960)
        t2iii_2x_last = gr.Slider(minimum=1, maximum=4, step=0.1, label='resize time x size for last pass', value=1)
        with gr.Row():
            t2iii_replace_prompt  = gr.Checkbox(label='Replace the prompt', value=False)
            t2iii_replace_negative_prompt  = gr.Checkbox(label='Replace the negative prompt', value=False)
        add_to_prompt   = gr.Textbox(label="Add to prompt", lines=2, max_lines=2000)
        add_to_negative_prompt   = gr.Textbox(label="Add to prompt", lines=2, max_lines=2000)

        return    [t2iii_reprocess,t2iii_steps,t2iii_cfg_scale,t2iii_seed_shift,t2iii_denoising_strength,t2iii_patch_upscale,t2iii_patch_shift,t2iii_2x_last,t2iii_save_first,t2iii_only_last,t2iii_face_correction,t2iii_face_correction_last, t2iii_model, t2iii_sampler,t2iii_clip,t2iii_noise,t2iii_patch_padding,t2iii_patch_square_size,t2iii_patch_border,t2iii_patch_mask_blur,t2iii_patch_end_denoising,t2iii_upscale_x,t2iii_upscale_y,add_to_prompt,add_to_negative_prompt,t2iii_replace_prompt,t2iii_replace_negative_prompt,t2iii_cfg_scale_end]
    def run(self,p,t2iii_reprocess,t2iii_steps,t2iii_cfg_scale,t2iii_seed_shift,t2iii_denoising_strength,t2iii_patch_upscale,t2iii_patch_shift,t2iii_2x_last,t2iii_save_first,t2iii_only_last,t2iii_face_correction,t2iii_face_correction_last, t2iii_model, t2iii_sampler,t2iii_clip,t2iii_noise,t2iii_patch_padding,t2iii_patch_square_size,t2iii_patch_border,t2iii_patch_mask_blur,t2iii_patch_end_denoising,t2iii_upscale_x,t2iii_upscale_y,add_to_prompt,add_to_negative_prompt,t2iii_replace_prompt,t2iii_replace_negative_prompt,t2iii_cfg_scale_end):
        def add_noise_to_image(img,seed,t2iii_noise):
            img = np.array(img)
            img = random_noise(img, mode='gaussian', seed=proc.seed, clip=True, var=t2iii_noise)
            img = np.array(255*img, dtype = 'uint8')
            img = Image.fromarray(np.array(img))
            return img
        def create_mask(size, border_width):
            im = Image.new('RGB', (size, size), color='white')
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, size, size), outline='black', width=border_width)
            return im
        def clean_model_name(model_name):
            if "[" in model_name:
                model_name = model_name.split(" ")[-2]
            return model_name
        def get_current_model_name():
            return clean_model_name(shared.opts.sd_model_checkpoint)
        def is_model_loaded(wanted):
            return wanted == get_current_model_name()

        img2img_samplers_names = [s.name for s in sd_samplers.samplers_for_img2img]
        img2img_sampler_index = [i for i in range(len(img2img_samplers_names)) if img2img_samplers_names[i] == t2iii_sampler][0]



        # models paths
        model_dir = "Stable-diffusion"
        model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
        initial_model = get_current_model_name()
        initial_model_path = os.path.abspath(os.path.join(model_path,initial_model))

        secondary_model_name = clean_model_name(t2iii_model)
        secondary_model_path = ""
        if t2iii_model != "Same":
            secondary_model_name = clean_model_name(t2iii_model)
            secondary_model_path = os.path.abspath(os.path.join(model_path,secondary_model_name))



        if p.seed == -1: p.seed = randint(0,1000000000)

        initial_CLIP = opts.data["CLIP_stop_at_last_layers"]
        p.do_not_save_samples = not t2iii_save_first
        initial_prompt = deepcopy(p.prompt)
        initial_negative_prompt = deepcopy(p.negative_prompt)

        n_iter=p.n_iter
        for j in range(n_iter):

            if t2iii_model != "Same":
                if not is_model_loaded(initial_model):
                    print()
                    sd_models.load_model(sd_models.CheckpointInfo(initial_model_path))

            p.n_iter=1
            if t2iii_clip > 0:
                opts.data["CLIP_stop_at_last_layers"] = initial_CLIP

            p.prompt = initial_prompt
            p.negative_prompt = initial_negative_prompt

            # PROCESS IMAGE
            proc = process_images(p)

            if add_to_prompt != "" or t2iii_replace_prompt:
                if t2iii_replace_prompt:
                    p.prompt = add_to_prompt
                else:
                    p.prompt = initial_prompt+add_to_prompt

            if add_to_negative_prompt != "" or t2iii_replace_negative_prompt:
                if t2iii_replace_negative_prompt:
                    p.negative_prompt = add_to_negative_prompt
                else:
                    p.negative_prompt = initial_negative_prompt+add_to_negative_prompt

            basename = ""
            extra_gen_parms = {
            'Initial steps':p.steps,
            'Initial CFG scale':p.cfg_scale,
            "Initial seed": p.seed,
            'Initial sampler': p.sampler_name,
            'Reprocess amount':t2iii_reprocess
            }
            for i in range(t2iii_reprocess):
                if t2iii_upscale_x > 64:
                    upscale_x = t2iii_upscale_x
                else:
                    upscale_x = p.width
                if t2iii_upscale_y > 64:
                    upscale_y = t2iii_upscale_y
                else:
                    upscale_y = p.height
                if t2iii_2x_last > 1 and i+1 == t2iii_reprocess:
                    upscale_x = int(upscale_x*t2iii_2x_last)
                    upscale_y = int(upscale_y*t2iii_2x_last)
                if t2iii_seed_shift == -1:
                    reprocess_seed = randint(0,999999999)
                else:
                    reprocess_seed = p.seed+t2iii_seed_shift*(i+1)
                if t2iii_clip > 0:
                    opts.data["CLIP_stop_at_last_layers"] = t2iii_clip

                if state.interrupted:
                    if t2iii_clip > 0:
                        opts.data["CLIP_stop_at_last_layers"] = initial_CLIP
                    break

                if t2iii_model != "Same":
                    if not is_model_loaded(secondary_model_name):
                        print()
                        sd_models.load_model(sd_models.CheckpointInfo(secondary_model_path))

                if i == 0:
                    proc_temp = proc
                else:
                    proc_temp = proc2
                if t2iii_noise > 0 :
                    proc_temp.images[0] = add_noise_to_image(proc_temp.images[0],p.seed,t2iii_noise)

                img2img_processing = StableDiffusionProcessingImg2Img(
                    init_images=proc_temp.images,
                    resize_mode=0,
                    denoising_strength=remap_range(i,0,t2iii_reprocess-1,t2iii_denoising_strength,t2iii_patch_end_denoising) if t2iii_patch_end_denoising > 0 else t2iii_denoising_strength,
                    mask=None,
                    mask_blur=t2iii_patch_mask_blur,
                    inpainting_fill=1,
                    inpaint_full_res=False,
                    inpaint_full_res_padding=0,
                    inpainting_mask_invert=0,
                    sd_model=p.sd_model,
                    outpath_samples=p.outpath_samples,
                    outpath_grids=p.outpath_grids,
                    prompt=p.prompt,
                    styles=p.styles,
                    seed=reprocess_seed,
                    subseed=proc_temp.subseed,
                    subseed_strength=p.subseed_strength,
                    seed_resize_from_h=p.seed_resize_from_h,
                    seed_resize_from_w=p.seed_resize_from_w,
                    #seed_enable_extras=p.seed_enable_extras,
                    sampler_name=t2iii_sampler,
                    #sampler_index=img2img_sampler_index,
                    batch_size=p.batch_size,
                    n_iter=p.n_iter,
                    steps=t2iii_steps,
                    cfg_scale=remap_range(i,0,t2iii_reprocess-1,t2iii_cfg_scale,t2iii_cfg_scale_end) if t2iii_cfg_scale_end > 0 else t2iii_cfg_scale,
                    width=upscale_x,
                    height=upscale_y,
                    restore_faces=(t2iii_face_correction or (t2iii_face_correction_last and t2iii_reprocess-1 == i)) and not (t2iii_reprocess-1 == i and not t2iii_face_correction_last),
                    tiling=p.tiling,
                    do_not_save_samples=True,
                    do_not_save_grid=p.do_not_save_grid,
                    extra_generation_params=extra_gen_parms,
                    overlay_images=p.overlay_images,
                    negative_prompt=p.negative_prompt,
                    eta=p.eta
                    )
                # if t2iii_reprocess-1 == i and t2iii_override_s_noise:
                #         img2img_processing.s_noise = t2iii_sampler_noise
                if not t2iii_patch_upscale:
                    proc2 = process_images(img2img_processing)
                    if ((t2iii_only_last and t2iii_reprocess-1 == i) or not t2iii_only_last):
                        images.save_image(proc2.images[0], p.outpath_samples, "", proc_temp.seed, proc2.prompt, opts.samples_format, info=proc_temp.info, p=p)
                else:
                    proc_temp.images[0] = proc_temp.images[0].resize((upscale_x, upscale_y), Image.Resampling.LANCZOS)
                    width_for_patch, height_for_patch = proc_temp.images[0].size
                    real_square_size = int(t2iii_patch_square_size)
                    overlap_pass = int(real_square_size/t2iii_reprocess)*i
                    patch_seed = reprocess_seed
                    for x in range(0, width_for_patch+overlap_pass if i>0 and t2iii_patch_shift else width_for_patch, real_square_size):
                        for y in range(0, height_for_patch+overlap_pass if i>0 and t2iii_patch_shift else height_for_patch, real_square_size):
                            if (
                                x-overlap_pass > width_for_patch    or
                                y-overlap_pass > height_for_patch   or
                                x+real_square_size-overlap_pass < 0 or
                                y+real_square_size-overlap_pass < 0
                                ): continue
                            if t2iii_seed_shift == -1:
                                patch_seed = randint(0,999999999)
                            else:
                                patch_seed = patch_seed+t2iii_seed_shift
                            patch = proc_temp.images[0].crop((x-t2iii_patch_padding-overlap_pass,
                                                              y-t2iii_patch_padding-overlap_pass,
                                                              x + real_square_size + t2iii_patch_padding-overlap_pass,
                                                              y + real_square_size + t2iii_patch_padding-overlap_pass))
                            img2img_processing.init_images = [patch]
                            img2img_processing.do_not_save_samples = True
                            img2img_processing.width  = patch.size[0]
                            img2img_processing.height = patch.size[1]
                            img2img_processing.seed   = patch_seed
                            mask = create_mask(patch.size[0],t2iii_patch_padding+t2iii_patch_border)
                            img2img_processing.image_mask = mask
                            proc_patch_temp = process_images(img2img_processing)
                            patch = proc_patch_temp.images[0]
                            patch = patch.crop((t2iii_patch_padding, t2iii_patch_padding, patch.size[0] - t2iii_patch_padding, patch.size[1] - t2iii_patch_padding))
                            proc_temp.images[0].paste(patch, (x-overlap_pass, y-overlap_pass))
                    proc2 = proc_patch_temp
                    proc2.images[0] = proc_temp.images[0]
                    images.save_image(proc2.images[0], p.outpath_samples, "", proc2.seed, proc2.prompt, opts.samples_format, info=proc2.info, p=p)


            p.subseed = p.subseed + 1 if p.subseed_strength  > 0 else p.subseed
            p.seed    = p.seed    + 1 if p.subseed_strength == 0 else p.seed
        if t2iii_model != "Same":
            if not is_model_loaded(initial_model):
                print()
                sd_models.load_model(sd_models.CheckpointInfo(initial_model_path))
        if t2iii_clip > 0:
            opts.data["CLIP_stop_at_last_layers"] = initial_CLIP
        return proc
