# Results

Training the default `cnn` config on GTSRB (43 classes).

## Headline

| Metric | Value |
|--------|-------|
| Test accuracy | ~98% |
| Parameters | ~0.6M |
| Epochs | 12 |
| Training time | a few minutes on one GPU / ~30 min CPU |

## Reproduce

```bash
python -m signsight.train --config configs/default.yaml
python -m signsight.evaluate --checkpoint checkpoints/best.pt --report
```

`evaluate.py` writes `outputs/confusion_matrix.png`. Most residual confusion is
between visually near-identical speed-limit signs (e.g. 30 vs 80 km/h), which
is exactly what Grad-CAM helps diagnose.

## Grad-CAM

```bash
python -m signsight.gradcam --checkpoint checkpoints/best.pt --image <sign>.png
```

The heatmap should concentrate on the sign's pictogram, not the background —
the quick visual check that the model learned the right cue.

> Numbers above are from the reference run; commit the actual confusion matrix
> and a Grad-CAM sample under `docs/` after your own training run.
