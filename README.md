# Story Visualization with Language Model Visual Details​ and Latent Diffusion Models LM-LDM


## Environment
```shell
conda create -n arldm python=3.8
conda activate arldm
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
git clone https://github.com/Flash-321/ARLDM.git
cd ARLDM
pip install -r requirements.txt
```
## Data Preparation
* Download the FlintstonesSV dataset [here](https://drive.google.com/file/d/1kG4esNwabJQPWqadSDaugrlF4dRaV33_/view?usp=sharing).
```shell
python data_script/vist_img_download.py
--json_dir /path/to/dii_json_files
--img_dir /path/to/save_images
--num_process 32
```
* To accelerate I/O, using the following scrips to convert your downloaded data to HDF5
```shell
python data_script/flintstones_hdf5.py
--data_dir /path/to/flintstones_data
--save_path /path/to/save_hdf5_file
 ```

## Training
Specify your directory and device configuration in `config.yaml` and run
```shell
python main.py
```
## Sample
Specify your directory and device configuration in `config.yaml` and run
```shell
python main.py
```

## Acknowledgment
Thanks a lot to (https://github.com/ARLDM) for kindly sharing FlintstonesSV dataset (and the code).

```bibtex
@article{pan2022synthesizing,
  title={Synthesizing Coherent Story with Auto-Regressive Latent Diffusion Models},
  author={Pan, Xichen and Qin, Pengda and Li, Yuhong and Xue, Hui and Chen, Wenhu},
  journal={arXiv preprint arXiv:2211.10950},
  year={2022}
}
```
