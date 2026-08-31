# RandLA-Net retained result

## Artifacts

- `best.pth`: selected checkpoint, epoch 92
- `metrics.csv`: per-epoch training and evaluation metrics
- `preds/`: 19 classified LAZ tiles, 76,351,005 total points

All prediction files preserve the source CRS and contain only ASPRS classes
1, 2, 5 and 6.

## Checkpoint-selection metrics

| Metric | Value |
|---|---:|
| mIoU | 0.97727 |
| others IoU | 0.93565 |
| ground IoU | 0.99992 |
| vegetation IoU | 0.99522 |
| building IoU | 0.97829 |

Training used four labeled tiles: BS842712, BS841710, BS838711 and BS843711.
The metrics above were computed on labeled points from the training tiles in
evaluation mode. They are checkpoint-selection metrics, not an independent
geographic test-set estimate.
