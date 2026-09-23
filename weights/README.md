# Weights

| Model | Download | Configuration |
| --- | --- | --- |
| Dynamic | [dynamic.pth](https://github.com/Tyreal-Izual/DeTurb/releases/download/v0.1.0/dynamic.pth) | `configs/dynamic.json` |
| Static | [static.pth](https://github.com/Tyreal-Izual/DeTurb/releases/download/v0.1.0/static.pth) | `configs/static.json` |

From the repository root:

```bash
curl -fL https://github.com/Tyreal-Izual/DeTurb/releases/download/v0.1.0/dynamic.pth -o weights/dynamic.pth
curl -fL https://github.com/Tyreal-Izual/DeTurb/releases/download/v0.1.0/static.pth -o weights/static.pth
```

These checkpoints contain model parameters and support inference or initialization for training. Optimizer and training-run state are excluded. File hashes are in [SHA256SUMS](SHA256SUMS) and [catalog.json](catalog.json).
