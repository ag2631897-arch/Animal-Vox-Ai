import asyncio
from inference.pipeline import AnimalVoxPipeline

async def test():
    pipe = AnimalVoxPipeline(device="cpu")
    pipe.load_models()
    pipe.load_modules()
    audio = r"data/sample_audio/chihuahua_bark.wav"
    res = await pipe.translate(audio, species="dogs", enable_modules=True)
    print("\n========================================")
    print("       ANIMALVOX AI INFERENCE RESULT    ")
    print("========================================")
    print(f"  Species Detected  : {res.get('species')} {res.get('species_emoji', '')}")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("========================================")

if __name__ == "__main__":
    asyncio.run(test())
