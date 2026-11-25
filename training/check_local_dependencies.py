import torch,bitsandbytes,diffusers,transformers,inspect
from diffusers import FluxTransformer2DModel,FluxPipeline,BitsAndBytesConfig
from transformers import SiglipModel,AutoProcessor,T5EncoderModel
print("IMPORTS",torch.__version__,diffusers.__version__,transformers.__version__,bitsandbytes.__version__,flush=True)
print("FLUX",inspect.signature(FluxPipeline.from_pretrained),flush=True)
print("OFFLOAD",inspect.signature(FluxTransformer2DModel.enable_group_offload),flush=True)
