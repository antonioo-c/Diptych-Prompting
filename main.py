import torch
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from PIL import Image, ImageDraw
import numpy as np

check_min_version("0.30.2")

def create_mask_on_image(image, xyxy):
    """
    Create a white mask on the image given xyxy coordinates.
    Args:
        image: PIL Image
        xyxy: List of [x1, y1, x2, y2] coordinates
    Returns:
        PIL Image with white mask
    """
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Create mask
    mask = Image.new('RGB', image.size, (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    
    # Draw white rectangle
    draw.rectangle(xyxy, fill=(255, 255, 255))
    
    # Convert mask to array
    mask_array = np.array(mask)
    
    # Apply mask to image
    masked_array = np.where(mask_array == 255, 255, img_array)
    
    return Image.fromarray(mask_array), Image.fromarray(masked_array)

def create_diptych_image(image):
    # Create a diptych image with original on left and black on right
    width, height = image.size
    diptych = Image.new('RGB', (width * 2, height), 'black')
    diptych.paste(image, (0, 0))
    return diptych

# Set image path , mask path and prompt
image_path = '../omini_haofan/assets/penguin.jpg'

subject_name='toy penguin'
target_text_prompt='wearing a christmas hat, in a busy street'
prompt=f'A two side-by-side image of same {subject_name}. LEFT: a photo of the {subject_name}; RIGHT: a photo of the {subject_name} {target_text_prompt}.'

# Build pipeline
controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
    )
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)

# Load image and mask
size = (1536, 768)
image = load_image(image_path).convert("RGB").resize((768, 768))
diptych_image = create_diptych_image(image)
# mask = load_image(mask_path).convert("RGB").resize(size)
# mask, mask_image = create_mask_on_image(image, [250, 275, 500, 400])
mask, mask_image = create_mask_on_image(diptych_image, [768, 0, 1536, 768])
generator = torch.Generator(device="cuda").manual_seed(24)

# diptych_image.save('diptych_image.png')
# mask.save('mask.png')

# Calculate attention scale mask
attn_scale_factor = 1.5
# Create a tensor of ones with same size as diptych image
H, W = size[1]//16, size[0]//16
attn_scale_mask = torch.zeros(size[1], size[0])
attn_scale_mask[:, 768:] = 1.0 # height, width
attn_scale_mask = torch.nn.functional.interpolate(attn_scale_mask[None, None, :, :], (H, W), mode='nearest-exact').flatten()
attn_scale_mask = attn_scale_mask[None, None, :, None].repeat(1, 24, 1, H*W)
# Get inverted attention mask by subtracting from 1.0
transposed_inverted_attn_scale_mask = (1.0 - attn_scale_mask).transpose(-1, -2)

cross_attn_region = torch.logical_and(attn_scale_mask, transposed_inverted_attn_scale_mask)

cross_attn_region = cross_attn_region * attn_scale_factor
cross_attn_region[cross_attn_region < 1.0] = 1.0

full_attn_scale_mask = torch.ones(1, 24, 512+H*W, 512+H*W)

full_attn_scale_mask[:, :, 512:, 512:] = cross_attn_region
# Convert to bfloat16 to match model dtype
full_attn_scale_mask = full_attn_scale_mask.to(device=pipe.transformer.device, dtype=torch.bfloat16)


# Convert attention mask to PIL image format
# Take first head's mask after prompt tokens (shape is now H*W x H*W)
# attn_vis = full_attn_scale_mask[0, 0]
# attn_vis[attn_vis <= 1.0] = 0
# attn_vis[attn_vis > 1.0] = 255
# attn_vis = attn_vis.cpu().numpy().astype(np.uint8)
# # Convert to PIL Image 
# attn_vis_img = Image.fromarray(attn_vis)
# attn_vis_img.save('attention_mask_vis.png')

# Inpaint
result = pipe(
    prompt=prompt,
    height=size[1],
    width=size[0],
    control_image=diptych_image,
    control_mask=mask,
    num_inference_steps=35,
    generator=generator,
    controlnet_conditioning_scale=0.95,
    guidance_scale=3.5,
    negative_prompt="",
    true_guidance_scale=1.0, # default: 3.5 for alpha and 1.0 for beta
    attn_scale_mask=full_attn_scale_mask,
).images[0]

# # Create a side-by-side comparison
# comparison = Image.new('RGB', (size[0] * 2, size[1]))
# comparison.paste(mask_image, (0, 0))
# comparison.paste(result, (size[0], 0))
# comparison.save('flux_inpaint_comparison.png')

result.save('flux_inpaint.png')
print("Successfully inpaint image")
