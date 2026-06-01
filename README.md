# Ordloshet

In Norwegian it means the inability to speak because meaning exceeds words.

## EPUB Translator
For now it is still WIP.
Translates English EPUB to French using local large language models.

### Usage

```bash
python epub.py --model nllb --strategy light --book file.epub
```

### Models

| Model | Type | Notes |
|-------|------|-------|
| `nllb` / `nllb-ct2` | Draft | NLLB-200 3.3B |
| `madlad` | Draft | MADLAD-400 10B |
| `opus` / `opus-ct2` | Draft | Helsinki OPUS-MT |
| `gemma` | Direct | Gemma Translate 12B |
| `mistral` / `qwen` / `tower` | Patcher | LLM post-editors |
| `nllb-qwen`, `madlad-literary`, ... | Composite | Draft + patcher pipelines |

> Warning side effect : transform your computer as heatsink and occupy a lot of place in your disk ( due to model weight ) 
