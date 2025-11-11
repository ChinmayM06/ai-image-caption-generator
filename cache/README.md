# Cache Directory

This directory stores cached data for the AI Image Caption Generator.

## Directory Structure

```
cache/
├── analytics.json       # Usage statistics and analytics (git-ignored, auto-created)
└── models/              # Cached ML models from Hugging Face (git-ignored)
    ├── models--microsoft--git-large-coco/
    └── models--Salesforce--blip-image-captioning-base/
```

## What's Ignored

- `analytics.json`: Contains user-specific analytics data
- `models/`: Contains downloaded ML models (can be several GB)
- All other cache files

## What's Tracked

- `.gitkeep` files: Preserve directory structure in Git
- This README: Documentation for the cache directory

## Automatic Creation

**No manual setup required!** Both files and directories are created automatically:

- ✅ `analytics.json` is created automatically on first run with the correct structure
- ✅ `models/` directory is created automatically when models are downloaded
- ✅ The cache directory structure is preserved via `.gitkeep` files

### Analytics JSON Structure

The `analytics.json` file is automatically initialized with this structure:

```json
{
    "total_captions": 0,
    "style_usage": {
        "None": 0,
        "Professional": 0,
        "Creative": 0,
        "Social Media": 0,
        "Technical": 0
    },
    "avg_processing_time": 0.0,
    "total_processing_time": 0.0,
    "model_usage": {
        "blip": 0,
        "git": 0
    },
    "error_count": 0,
    "last_updated": null
}
```

**You don't need to create this file manually** - it will be generated automatically when the app runs for the first time.

## Notes

- The cache directory is automatically created by the application
- Models are downloaded on first use and cached here
- Analytics data is stored in `analytics.json` (created automatically)
- All cache contents are ignored by Git to avoid committing large files and user data
- Each user gets their own analytics.json file with their usage statistics

