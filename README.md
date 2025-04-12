# GOttack: Universal Adversarial Attacks on Graph Neural Networks via Graph Orbits Learning (ICLR 2025)

Here we provide the implementation of GOttack for our ICLR 2025 paper "GOttack: Universal Adversarial Attacks on Graph Neural Networks via Graph Orbits Learning (ICLR 2025)".

Other resources: [Project page]() - [Paper](https://openreview.net/forum?id=YbURbViE7l) - [Video(Slideslive)]()

Please cite our paper if you use the method in your own work:

```
@inproceedings{alom2025gottack,
title={{GO}ttack: Universal Adversarial Attacks on Graph Neural Networks via Graph Orbits Learning},
author={Zulfikar Alom and Tran Gia Bao Ngo and Murat Kantarcioglu and Cuneyt Gurcan Akcora},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=YbURbViE7l}
}
```

## Environment setup

Python compiler version: `3.6`

1. Set up virtual environment (conda should work as well).

```
python -m venv gottack/
source gottack/bin/activate
```

2. Upgrade pip (Optional)

```
install external packages
```

3. Install Gottack dependencies
```
pip install -r requirements.txt
```

4. Results of Gottack can be re-produced by running ```main.py```. Please see ```main.py``` for
further instructions.

## Code documentation

- Implementation of `GOttack` can be found at `OrbitAttack.py`
- A sample script to conduct experiments on `GOttack` is provided in `main.py`
- If you want to conduct `GOttack` on new dataset, you may need to perform orbit discovery on the dataset. A sample of code for orbit discovery can be found at `utility/obg_orbit_count.py`.
  - To set up `orca` library, we recommend to see instructions provided [here.](https://github.com/qema/orca-py)

---

**Credits:**    
- The GOttack is implemented with reference to ````Nettack````'s source code,
a method proposed in the paper: 'Adversarial Attacks on Neural Networks for Graph Data'
by Daniel Zügner, Amir Akbarnejad and Stephan Günnemann
- GOttack is built on top of ````DeepRobust```` - A PyTorch Library for Adversarial
Attacks and Defenses developed by Yaxin Li, Wei Jin, Han Xu and Jiliang Tang

- To perform orbit discovery on graph data, we utilize `PyORCA: ORCA orbit Counting Python Wrapper` implemented by Andrew Wang and Rex Ying. The original source code can be found [here.](https://github.com/qema/orca-py)






