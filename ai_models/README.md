# AI models

All the AI model code for the Foot X-ray Assistant, one folder per model: training, testing and prediction. No app code here (the app is in `../flatfoot_app/`, and it loads `heel_spur/` and `bunion/` from this folder).

| Folder | What it does | X-ray | Test score |
|---|---|---|---|
| [`flat_foot/`](flat_foot/) | Flat foot: classifier + heel bone angle (flat if under 17.7°) | Side | 93.8% of 211 unseen X-rays |
| [`heel_spur/`](heel_spur/) | Finds the heel bone, then: normal / heel spur / child's heel (Sever's) | Side | 97.5% of 595 (spurs caught 96.0%, normals cleared 98.2%) |
| [`bunion/`](bunion/) | Bunion angles HVA and IMA from 6 points per foot, plus a foot finder for two-foot X-rays | Top-down | Within 3°: HVA 85.0%, IMA 95.6% (160 feet) |
| [`fracture/`](fracture/) | Fracture probability plus a map of where the break is | Any leg view | Whole leg AUC 0.92; foot/ankle AUC 0.67 (demo only) |
| [`view_check/`](view_check/) | Says what an X-ray is: side / top-down / angled / not a foot | Any | 99.1% correct; 83.1% balanced on a new hospital |

Each folder's README explains the data, how to train and how to test it.

## What is not here

- **Model weights that can't be shared.** The flat foot and heel spur `.onnx` files are in `../flatfoot_app/`. Bunion and X-ray check weights are not in git (their training data licences are non-commercial / no-derivatives), and neither is the fracture model.
- **X-rays, labels and masks.** Each README says where to download the dataset.
- **The newer fracture retrain** (X-rays at every angle, versions A and B) is still on the training PC and not committed yet.
- **The original flat foot training notebooks** (Google Colab) are not in the repo.

Paths in the commands are relative to each model's own folder, so `cd` into it first.
