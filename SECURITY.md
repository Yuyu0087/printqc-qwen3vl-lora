# Security

PrintQC v0.1.0 performs local inference only. It does not send images to an external AI API and does not control a printer.

The adapter downloader accepts only the fixed GitHub Release URL recorded in `release_manifest.json`, verifies SHA256, rejects unsafe ZIP entries, and writes the cache READY marker only after validation.

Do not provide GitHub tokens, Hugging Face tokens, archive passwords, private image paths, or printer credentials to this program.
