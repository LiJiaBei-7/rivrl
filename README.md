# Reading-strategy Inspired Visual Representation Learning for Text-to-Video Retrieval

Source code of our paper [Reading-strategy Inspired Visual Representation Learning for Text-to-Video Retrieval](https://arxiv.org/abs/). 

![image](framework.png)



## Table of Contents

* [Environments](#environments)

* [Required Data](#required-data)

* [RIVRL on MSRVTT10K](#RIVRL-on-MSRVTT10K)
  
  * [Model Training and Evaluation](#model-training-and-evaluation)
  * [Evaluation using Provided Checkpoints](#Evaluation-using-Provided-Checkpoints)
  * [Expected Performance](#Expected-Performance)
  
* [RIVRL on VATEX](#RIVRL-on-VATEX)

  * [Model Training and Evaluation](#model-training-and-evaluation-1)
  * [Expected Performance](#Expected-Performance-1)

* [RIVRL on TGIF](#RIVRL-on-TGIF)

  * [Model Training and Evaluation](#model-training-and-evaluation-2)
  * [Expected Performance](#Expected-Performance-2)

  

## Environments

* CUDA 10.1

* Python 3.8.5

* PyTorch 1.7.0


We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install the required packages.  

```shell
conda create --name rivrl_env python=3.8.5
conda activate rivrl_env
git clone https://github.com/LiJiaBei-7/rivrl.git
cd rivrl
pip install -r requirements.txt
conda deactivate
```



## Required Data

<table>
  <tr align="center">
      <th >DataSets</th> <th>Splits <th>BERT</th>
  </tr>	
  <tr>
    <td colspan='2' align="center">MST-VTT</td>
    <td align='center'><a href='https://pan.baidu.com/s/1K7KJ_dc9UgL5u9UQ9DXndw
      '>url</a>,
      password:6knd</td>
  </tr>
  <tr align="center">
    <td colspan='2' align="center">VATEX</td>
    <td align='center'><a href='https://pan.baidu.com/s/1K7KJ_dc9UgL5u9UQ9DXndw
      '>url</a>,
      password:6knd</td>
  </tr>
  <tr align="center">	  
    <td rowspan='2'>TGIF</td>
    <td>Chen</td>
    <td align='center'><a href='https://pan.baidu.com/s/1K7KJ_dc9UgL5u9UQ9DXndw
      '>url</a>,
      password:6knd</td>
  </tr>
  <tr align="center">	  
    <td>Li</td>
    <td align='center'><a href='https://pan.baidu.com/s/1K7KJ_dc9UgL5u9UQ9DXndw
      '>url</a>,
      password:6knd</td>
</table>



As we use the same datasets with the [Dual Encoding](https://arxiv.org/abs/2009.05381) , please refer to [here](https://github.com/danieljf24/hybrid_space) for details about how to download and extract datasets and pre-trained word2vec, and to [here](https://github.com/danieljf24/hybrid_space/tree/master/dataset) for more information about the dataset.

Run the following script to download the features of BERT, the extracted data is placed in `$HOME/VisualSearch/`.

```shell
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH

# download the features of BERT
cd $ROOTPATH
tar zxf bert_extract.tar.gz -C $ROOTPATH
```

## RIVRL on MSRVTT10K

The datasets and their split names are shown in the following table: 

<table>
  <tr align="center">
    <th>DataSets</th><th>Split-Name</th>
  </tr>
  <tr align="center">
    <td rowspan='3'>MSR-VTT</td>
    <td>msrvtt10k</td>
  </tr>
  <tr align="center">
    <td>msrvtt10kmiech</td>
  </tr>
  <tr align="center">
    <td>msrvtt10kyu</td>
  </tr>
</table>

### Model Training and Evaluation

Run the following script to train and evaluate `RIVRL` network. Specifically, it will train `RIVRL` network and select a checkpoint that performs best on the validation set as the final model. Notice that we only save the best-performing checkpoint on the validation set to save disk space

```shell
ROOTPATH=$HOME/VisualSearch

conda activate rivrl_env

# To train the model on the MSRVTT, which the feature is resnext-101_resnet152-13k 
./do_all_msrvtt.sh $ROOTPATH <split-Name> <useBert> <gpu-id>

# Example:
# Train RIVRL with the BERT on the official split of MSR-VTT
./do_all_msrvtt.sh $ROOTPATH msrvtt10k 1 0
```

Where the `<split-Name>`  indicates different partitions of the data set which are msrvtt10k，msrvtt10kmiech and msrvtt10yu, respectively,  the `<gpu-id>` is the index of the GPU to train on and the `<useBert>` indicates whether training with BERT, the value of it is 1 (use) or 0 (not use).

### Evaluation using Provided Checkpoints

The overview of pre-trained checkpoints on MSR-VTT is as follows.

<table>
  <tr align='center'>
    <th>DataSets</th><th>Splits</th><th>Bert</th><th>Pre-trained Checkpoints</th>
  </tr>
  <tr align='center'>
  	<td rowspan='6'>MSR-VTT</td>
    <td rowspan='2'>MV-Yu</td>
    <td>w/o</td>
    <td rowspan='6'><a href='https://pan.baidu.com/s/1dEmmhpS3x_WvuMyzDZI40Q'>url</a>，password:wb3c</td>
  </tr>
  <tr align='center'>
  	<td>with</td>
  </tr>
  <tr align='center'>
    <td rowspan='2'>MV-Miech</td>
    <td>w/o</td>
  </tr>
  <tr align='center'>
  	<td>with</td>
  </tr>
  <tr align='center'>
    <td rowspan='2'>MV-Xu</td>
    <td>w/o</td>
  </tr>
  <tr align='center'>
  	<td>with</td>
</table>

Run the following script to download and evaluate our trained checkpoints. The trained checkpoints can also be downloaded from Baidu pan ([url](https://pan.baidu.com/s/1dEmmhpS3x_WvuMyzDZI40Q), password:wb3c).

```shell
ROOTPATH=$HOME/VisualSearch
MODELDIR=$HOME/VisualSearch/checkpoints
mkdir -p $MODELDIR

# download trained checkpoints and evaluate 
./do_test.sh $ROOTPATH <split-Name> $MODELDIR <gpu-id>

# Example:
# evaluate on the official split of MSR-VTT
./do_test.sh $ROOTPATH msrvtt10k $MODELDIR <gpu-id>
```



### Expected Performance

The expected performance of RIVRL on MSR-VTT is as follows. Notice that due to random factors in SGD based training, the numbers differ slightly from those reported in the paper.

<table>
  <tr align="center">
    <th rowspan='2'>DataSets</th><th rowspan="2">Splits</th><th rowspan='2'>BERT</th><th colspan='5'>Text-to-Video Retrieval</th> <th rowspan='2'>SumR</th>
    </tr>
  <tr align="center">
        <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> MedR </th> <th>	mAP </th>
  </tr>
 	<tr align="center">	  
    <td rowspan='6'>MSR-VTT</td>
    <td rowspan='2'>MV-Xu</td>
    <td>w/o</td>
    <td>12.9</td><td>33.0</td><td>44.6</td><td>14</td><td>23.07</td> 
    <td>90.5</td> 
  </tr>
  <tr align="center">
    <td>with</td>
     <td>13.7</td><td>34.6</td><td>46.4</td><td>13</td><td>24.19</td> 
   	 <td>94.6</td> 
  </tr>  
  <tr align="center">
    <td rowspan='2'>MV-Miech</td>
    <td>w/o</td>
    <td>25.3</td><td>53.6</td><td>67.0</td><td>4</td><td>38.5</td> 
    <td>145.9</td> 
  </tr>
  <tr align="center">
    <td>with</td>
    <td>26.2</td><td>56.6</td><td>68.2</td><td>4</td><td>39.92</td> 
    <td>151.0</td> 
  </tr>
  <tr align="center">
    <td rowspan='2'>MV-Yu</td>
    <td>w/o</td>
    <td>24.2</td><td>51.5</td><td>63.8</td><td>5</td><td>36.86</td> 
    <td>139.5</td>  
  </tr>
   <tr align="center">
    <td>with</td>
    <td>27.9</td><td>59.3</td><td>71.3</td><td>4</td><td>42.0</td> 
    <td>158.4</td>
  </tr>
</table>



## RIVRL on VATEX

### Model Training and Evaluation

Run the following script to train and evaluate `RIVRL` network on VATEX.

```shell
ROOTPATH=$HOME/VisualSearch

conda activate rivrl_env

# To train the model on the VATEX
./do_all_vatex.sh $ROOTPATH <useBert> <gpu-id>
```

### Expected Performance

Run the following script to download and evaluate our trained model on the VATEX from Baidu pan ([url](https://pan.baidu.com/s/1dEmmhpS3x_WvuMyzDZI40Q), password:wb3c).

```shell
ROOTPATH=$HOME/VisualSearch
MODELDIR=$HOME/VisualSearch/checkpoints
mkdir -p $MODELDIR

# download trained checkpoints and evaluate 
./do_test.sh $ROOTPATH vatex $MODELDIR <gpu-id>
```

The expected performance of Dual Encoding with hybrid space learning on VATEX is as follows.

<table>
  <tr align="center">
    <th rowspan='2'>DataSets</th><th rowspan="2">Splits</th><th rowspan='2'>BERT</th><th colspan='5'>Text-to-Video Retrieval</th> <th rowspan='2'>SumR</th>
    </tr>
  <tr align="center">
        <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> MedR </th> <th>	mAP </th>
  </tr>
  <tr align="center">
    <td colspan='2' rowspan ='2' align="center">VATEX</td>
    <td>w/o</td>
    <td>39.4</td><td>76.1</td><td>84.8</td><td>2</td><td>55.3</td> 
    <td>200.4</td 
  </tr>
	<tr align="center">
    <td>with</td>
    <td>39.1</td><td>76.7</td><td>85.4</td><td>2</td><td>55.4</td> 
    <td>201.0</td> 
  </tr>
</table>


## RIVRL on TGIF

The datasets and their split names are shown in the following table: 

<table>
  <tr align="center">
    <th>DataSets</th><th>Split-Name</th>
  </tr>
  <tr align="center">
    <td rowspan='2'>TGIF</td>
    <td>tgif_li</td>
  </tr>
  <tr align="center">
    <td>tgif_chen</td>
  </tr>
</table>



### Model Training and Evaluation

Run the following script to train and evaluate `RIVRL` network on TGIF.

```shell
ROOTPATH=$HOME/VisualSearch

conda activate rivrl_env

# To train the model on the tgif-li
./do_all_tgif_li.sh $ROOTPATH <useBert> <gpu-id>

# To train the model on the tgif-chen
./do_all_tgif_chen.sh $ROOTPATH <useBert> <gpu-id>
```

### Expected Performance

Run the following script to download and evaluate our trained model on the TGIF from Baidu pan ([url](https://pan.baidu.com/s/1dEmmhpS3x_WvuMyzDZI40Q), password:wb3c).

```shell
ROOTPATH=$HOME/VisualSearch
MODELDIR=$HOME/VisualSearch/checkpoints
mkdir -p $MODELDIR

# download trained checkpoints and evaluate on the tgif-li
./do_test.sh $ROOTPATH tgif-li $MODELDIR <gpu-id>
# download trained checkpoints and evaluate on the tgif-chen
./do_test.sh $ROOTPATH tgif-chen $MODELDIR <gpu-id>
```

The expected performance of Dual Encoding with hybrid space learning on TGIF is as follows.

<table>
  <tr align="center">
    <th rowspan='2'>DataSets</th><th rowspan="2">Splits</th><th rowspan='2'>BERT</th><th colspan='5'>Text-to-Video Retrieval</th> <th rowspan='2'>SumR</th>
    </tr>
  <tr align="center">
        <th> R@1 </th> <th> R@5 </th> <th> R@10 </th> <th> MedR </th> <th>	mAP </th>
  </tr>
  <tr align="center">	  
    <td rowspan='4'>TGIF</td>
    <td rowspan='2'>TGIF-Li</td>
    <td>w/o</td>
    <td>11.3</td><td>25.3</td><td>33.6</td><td>34</td><td>18.7</td> 
    <td>70.3</td> 
  </tr>
   <tr align="center">	  
    <td>with</td>
    <td>12.1</td><td>26.6</td><td>35.1</td><td>29</td><td>19.75</td> 
    <td>73.8</td> 
  </tr>
   <tr align="center">	  
    <td rowspan='2'>TGIF-Chen</td>
    <td>w/o</td>
    <td>6.4</td><td>16.1</td><td>22.4</td><td>91</td><td>11.81</td> 
    <td>44.9</td>  
  </tr>
  <tr align="center">	  
    <td>with</td>
    <td>6.8</td><td>17.2</td><td>23.5</td><td>79</td><td>12.45</td> 
    <td>47.4</td> 
  </tr>
</table>



## References

If you find the package useful, please consider citing our paper:
