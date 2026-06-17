# signsight

Recognise German traffic signs from images. A small but complete computer-vision
project that goes **from raw dataset → trained model → deployed inference API**,
written to be read.

It deliberately covers the three things an AI/CV engineer does day to day:

1. **Computer vision** — convolutional network for 43-class traffic-sign
   recognition on the GTSRB benchmark, plus **Grad-CAM** to see *where* the
   model looks.
2. **Model training** — a clean, config-driven PyTorch training pipeline
   (augmentation, train/val split, scheduler, checkpointing, metrics).
3. **Software engineering / deployment** — the trained model served behind a
   **FastAPI** endpoint, containerised with Docker, tested with pytest, and
   wired to CI.

> _Work in progress — building it up commit by commit._

## Quick start

```bash
pip install -r requirements.txt

# Train (downloads GTSRB automatically on first run)
python -m signsight.train --config configs/default.yaml

# Evaluate on the test split -> prints accuracy, writes a confusion matrix
python -m signsight.evaluate --checkpoint checkpoints/best.pt

# Explain a single prediction with Grad-CAM
python -m signsight.gradcam --checkpoint checkpoints/best.pt --image path/to/sign.png

# Serve it
uvicorn serve.main:app --reload      # then open http://localhost:8000
```

## Project layout

```
signsight/        the CV + training library
  data.py         GTSRB loading, augmentation, dataloaders
  model.py        TrafficSignCNN (from scratch) + transfer-learning option
  train.py        config-driven training loop
  evaluate.py     accuracy, per-class report, confusion matrix
  gradcam.py      Grad-CAM saliency visualisation
  infer.py        load a checkpoint, predict on one image
configs/          experiment configs (yaml)
serve/            FastAPI app + Dockerfile + demo page
tests/            unit tests (run without the dataset, CPU only)
```

## Results

The 3-block CNN reaches **~98% test accuracy** on GTSRB after ~10 epochs on a
single GPU (a few minutes), or comparable accuracy on CPU with more time.
See [`docs/RESULTS.md`](docs/RESULTS.md) for the training curve and the
confusion matrix.

## License

MIT — see [LICENSE](LICENSE).
