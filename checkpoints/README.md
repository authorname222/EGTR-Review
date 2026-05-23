# Checkpoints

The trained student model weights are too large for this anonymous repository.

During the double-blind review, please download the LoRA adapter and tokenizer from
the anonymous mirror below and place them under `checkpoints/egtr_student/`:

```
# Anonymous download link (placeholder — to be filled in by the anonymous-mirror admin)
EGTR_STUDENT_URL="<ANONYMIZED_DOWNLOAD_URL>"
mkdir -p checkpoints/egtr_student
# Example (replace with the actual command for the chosen mirror):
#   curl -L "$EGTR_STUDENT_URL/adapter_model.safetensors" -o checkpoints/egtr_student/adapter_model.safetensors
#   curl -L "$EGTR_STUDENT_URL/adapter_config.json"      -o checkpoints/egtr_student/adapter_config.json
#   curl -L "$EGTR_STUDENT_URL/tokenizer.json"            -o checkpoints/egtr_student/tokenizer.json
```

After download, run:

```bash
bash scripts/run_inference.sh data/samples/eval/demo_papers.json
```

⚠️ The anonymous mirror is hosted on a newly-created account that is not linked to
the authors. Upon acceptance, the weights will be relocated to a regular release
channel under the authors' names.
