# Multi Agent Diverse Genererative Adversarial Network (MAD-GAN) Tensorflow

Code for MAD-GAN repository

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
- Linux with Tensorflow GPU edition + cuDNN in anaconda distribution with Tensorflow v 1.2 or newer
```

# pix2pix-tensorflow

Based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.


Tensorflow implementation of madgan on pix2pix architechture. Multiple generator learns a mapping from input images to output images to model disjoint modes, like these examples from the original paper:
This port is based directly on the tensorflow implementation by [Christopher Hesse ](https://github.com/affinelayer/pix2pix-tensorflow)


### Getting Started

```sh
# clone this repo
https://github.com/rishabh135/MAD-GAN-MLCAMP.git
cd MAD-GAN-MLCAMP
# download the CMP Facades dataset (generated from http://cmp.felk.cvut.cz/~tylecr1/facade/)
python download-dataset.py facades
# train the model (this may take 1-8 hours depending on GPU, on CPU you will be waiting for a bit)

python madgan_compete.py --mode train --output_dir madgan_facades_train --max_epochs 200  --input_dir facades/train 

```

## Datasets and Trained Models

The data format used by this program is the same as the original pix2pix format, which consists of images of input and desired output side by side.

Some datasets have been made available by the authors of the pix2pix paper.  To download those datasets, use the included script `download-dataset.py`. 

Tip : The `facades` dataset is the smallest and easiest to get started with.


## Citation
If you use this code for your research, please cite the papers this code is based on: <a href="https://arxiv.org/pdf/1611.07004v1.pdf">Image-to-Image Translation Using Conditional Adversarial Networks</a>: <a href="https://arxiv.org/abs/1704.02906.pdf"> Multi Agent Diverse Generative Adversarial Networks</a>:

```
@article{pix2pix2016,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={arxiv},
  year={2016}
}



@article{DBLP:journals/corr/GhoshKNTD17,
  author    = {Arnab Ghosh and
               Viveka Kulharia and
               Vinay P. Namboodiri and
               Philip H. S. Torr and
               Puneet Kumar Dokania},
  title     = {Multi-Agent Diverse Generative Adversarial Networks},
  journal   = {CoRR},
  volume    = {abs/1704.02906},
  year      = {2017},
  url       = {http://arxiv.org/abs/1704.02906},
  timestamp = {Wed, 07 Jun 2017 14:42:36 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/GhoshKNTD17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}

```

## Acknowledgments
Thanks to the Tensorflow team for making such a quality library!  And special thanks to Arnab Ghosh , Sanghoon Hong and Namju Kim for answering my questions about the codes.


## Contributing

Please contact me at rishabh1351995@gmail.com for contributions requests. 

## Authors

* **Aranb Ghosh and Viveka Kulharia** - *Paper and implementation help* - [Paper Link](https://github.com/arnabgho)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Arnab Ghosh and Viveka Kulharia
* Sanghoon Hong and Namju Kim
* Ferenc Huszár , Ian Goodfellow
