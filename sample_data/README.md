# sample_data

A small, offline sample of GTSRB **test** images (22 images across 11 classes),
so the repo runs end to end without downloading anything. Filenames encode the
ground-truth class: `<class_id>_<class-name>_<original-id>.png`.

Use them with the bundled model:

```bash
python -m signsight.infer --checkpoint models/signsight-cnn.pt \
                          --image sample_data/00_speed-limit-20km-h_00243.png
```

The full GTSRB dataset (~39k train / ~12k test images) is **not** committed —
it is a third-party dataset and is downloaded automatically by `torchvision`
the first time you run `python -m signsight.train`.
