import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


class UrbanPlanner:
    """
    Reusable Generative Urban Planning Service
    Uses pretrained diffusion model (Stable Diffusion)
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = "E:/citysense360/outputs",
        device: str | None = None,
    ):
        """
        Initialize the diffusion model once.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        self.pipe = self.pipe.to(self.device)

    def generate_plan(
        self,
        prompt: str,
        filename: str,
        steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
    ):
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        ).images[0]

        save_path = os.path.join(self.output_dir, filename)
        image.save(save_path)

        return image, save_path


if __name__ == "__main__":
    planner = UrbanPlanner()

    prompts = {
        "green_city.png":
            "A sustainable smart city with green parks, solar panels, pedestrian zones, aerial view",

        "transport_city.png":
            "Urban city planning with efficient public transport, metro lines, bus corridors, smart roads",

        "mixed_use_city.png":
            "Modern city layout with residential, commercial and industrial zones, smart infrastructure"
    }

    for filename, prompt in prompts.items():
        path = planner.generate_plan(prompt, filename)
        print(f"Generated: {path}")
