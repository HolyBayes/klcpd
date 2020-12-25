# Unofficial PyTorch implementation of KL-CPD
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**KL-CPD** is an algorithm for change point and anomaly detection in time series.

More information can be found in the 2019 (https://openreview.net/forum?id=r1GbfhRqF7)[paper] *Kernel Change-point Detection with Auxiliary Deep Generative Models*.

## Usage

```python
dim, seq_length = 1, 100
ts = np.random.randn(seq_length,dim)
device = torch.device('cuda')
model = KL_CPD(dim).to(device)
model.fit(ts)
preds = model.predict(ts)
print(preds)
```


## Installation

## Authors

    @article{chang2019kernel,
      title={Kernel change-point detection with auxiliary deep generative models},
      author={Chang, Wei-Cheng and Li, Chun-Liang and Yang, Yiming and P{\'o}czos, Barnab{\'a}s},
      journal={arXiv preprint arXiv:1901.06077},
      year={2019}
    }

## Contacts

Artem Ryzhikov, LAMBDA laboratory, Higher School of Economics, Yandex School of Data Analysis

**E-mail:** artemryzhikoff@yandex.ru

**Linkedin:** https://www.linkedin.com/in/artem-ryzhikov-2b6308103/

**Link:** https://www.hse.ru/org/persons/190912317
