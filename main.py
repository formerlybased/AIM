import json
from diffusers import StableDiffusionPipeline

print("Starting program")

prompts = []

ai = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

print("Opening config")
config_file = open("config.json")

print("Parsing config")
config = json.load(config_file)

print("Closing config")
config_file.close()

print("Loading prompts")
file = open(config["input_file"], "r")

print("Saving prompts")
for line in file:
    prompts.append(line)

print("Closing prompt file")
file.close()

print("Starting generation...")
for a in prompts:
    a.removeprefix("\n")
    print("Generating " + config["image_count"] + " images from prompt: " + a)
    for i in config["image_count"]:
        img = ai(a).images[0]
        img.save(f"prompt{a}count{i}.png")
        print("Image: " + a + "saved.")

print("Program over.")