# whisker

This is a refactor of the code we were provided in class ([decal-lecture-2](https://github.com/berkeleyaisafety.com/decal-lecture-2); also see the `main` branch).
It's designed to replicate the behavior of the 
original codebase we were given, but instead of leveraging Tinker and OpenRouter, 
it runs locally on Apple Silicon using [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders). This is made possible thanks to HuggingFace's [`peft`](https://github.com/huggingface/peft) library.

The original codebase had separate scripts to run, e.g., `uv run gen_data.py`, `uv run chat.py`, etc. In this refactor, these various scripts have been centralized
into a single command-line interface called `whisker`. You can `uv run whisker` in the repository's root to get an overview
of available scripts to run:
```plaintext
Usage: whisker [OPTIONS] COMMAND [ARGS]...

 whisker: local SFT + probing harness for Qwen3.5-4B.

 whisker gen      make a dataset with any OpenAI-compatible model (.env)
 whisker prompts  grow a prompt pool
 whisker inspect  look at a dataset: stats, openers, and what the model is trained on
 whisker train    LoRA fine-tune locally (MPS)
 whisker runs     list finished / in-progress runs
 whisker chat     talk to base or a run (hot-swap adapters in-chat)
 whisker compare  same prompts through base + runs, side by side 
 whisker plot     utils for generating data visualizations
```
Each subcommand supports an optional `--help` flag that displays usage details and required/optional arguments.

> [!TIP]
> Most of this code was written by Claude Opus 5.5, but this README was written solely by me. So if I left anything out that you want to explore, drop Claude into the repo. It could probably point out some things I missed in this doc.

## Downloading the Tokenizer and Weights
Because training is now happening locally, we need local copies of both the Tokenizer and the model's weights. This is done automatically for you the first time you run `whisker train` -- it downloads them from HuggingFace and stores them in your local cache, at `~/.cache/huggingface/hub`. You can also download the necessary components manually by running `uvx hf download Qwen/Qwen3.5-4B`, but I'd recommend just letting `whisker` take care of it for you.

## Generating Synthetic Data
Another avenue we were dependent upon OpenRouter for was generating our synthetic data. I've refactored this so that we can use any OpenAI-compatible
API endpoint, such as your own provider's API keys, or even a small locally-running model through something like LM Studio. There's three environment variables to set in `.env`.

- `GEN_BASE_URL` - For example, if you're using the Anthropic API, you'd put `https://api.anthropic.com/v1/`
- `GEN_API_KEY` - Your API key for the provider you've chosen. If you're using a local model API, like LM Studio, you can leave this blank.
- `GEN_MODEL` - The model you want to use, for example, `claude-sonnet-5`, `gpt-5.6-sol`, etc.

> [!NOTE]
> You only need a `.env` file with these variables set *if you want to create new a dataset* with `whisker gen`. All other 
> whisker features can be used without these environment variables.

I'm still working with the dataset we generated in class, `refusal_cats3.jsonl`, so I haven't done a full pass through the code yet in this part of the 
refactor. Let me know if you run into any bugs here and we can patch it.
