In this assignment, you will train machine translation models for a low-resource lanaguage (to English). You will complete three models: 
1. Bilingual baseline model
2. Multlingual trained model
3. Finetuning pre-trained models 

# Resources and GCP
Having a GPU is not necessary but highly recommended. Google Cloud provides for $300 free credits for new users. Instructions on how to set it up can be found in `GCP.md`. Additionally, please visit TA office hours if y'all still face issues in setting this up.

# Kaggle Leaderboard
You are expected to upload your test predictions on this [kaggle leaderboard](https://www.kaggle.com/t/b2580bc99bd24082ac9518d0fbe62d7b). Instructions on how to do this can be found below. You are expected to do this for each of the three baselines (for a B, B+ and A- grade respectively; more details in the grading section). Benchmark submissions for these have already been made and you are expected to achieve similar values, accounting for variance. Additional submissions with new methods to improve performance are encouraged. Names for baseline submissions are expected to be `<andrew_id>-bilingual`, `<andrew_id>-multilingual`, and `<andrew_id>-flores`. Additional submissions can be named `<andrew_id>-exp<n>`, where `n` denotes the `nth` additional submission.

# Step 1: Create a conda environment with python=3.10

```
conda create --name mnlp-assn1 python=3.10
conda activate mnlp-assn1
```

# Step 2: Install the required packages

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install fairseq==0.12.2
pip install unbabel-comet==2.0.2
pip install sacremoses sentencepiece
```

Additionally, one needs to clone the fairseq repo in their local directory. This can be done as follows:

```
git clone https://github.com/facebookresearch/fairseq.git
```

# Step 3: Unzip the data

```
unzip data/assign1-data.zip
rm -rf data/assign1-data.zip
```


# Bilingual Baselines

As a first step, we will train NMT systems using only parallel data from the language of interest. In this assignment, we will consider one low-resource language: Belarusian (bel), translating to English (eng).

We provide scripts for complete data processing, including simple cleaning and (subword) tokenization as well as training and evaluation. You should read the scripts to understand the data processing pipeline, training and evaluation.

To perform preprocessing for the bel-eng parallel corpora, run:

```
bash preprocess-ted-bilingual.sh
```

Then, you can train and evaluate models on the preprocess data by running:

```
bash traineval_bel_eng.sh
```

You should be able to reproduce the numbers below on the validation set. To account for variance in experiments, a drop of 0.5 BLEU or 0.05 COMET are acceptable: 

| LP | BLEU | COMET |
| --- | --- | --- |
| bel-eng | 1.92 | -1.4185 |

With slight modifications in the script, you can obtain predictions for the test set. These prediction files need to be submitted to the kaggle leaderboard and can be converted to a CSV format compatible for submission using the `convert_to_csv.py` script. The benchmark submission for this baseline is titled `test-bilingual-prediction.csv` on the leaderboard. Please name your submission `<andrew_id>-bilingual`.


# Multilingual Training

Note that since the languages we consider have very limited amount of parallel training data, the NMT model performs really badly, with BLEU scores of less than 10 and (very) negative COMET scores. This is a known issue for neural models in general - they are much more data-hungry than statistical methods. Luckily, we can use multilingual training to boost the performance of these low resource languages. In this section, you will learn to use data from the high resource-related languages to improve the NMT performance.

Crucially, for cross-lingual transfer to work, the high-resource must be similar to the low-resource language we are transferring to. For example, as mentioned in the first lecture, several languages from North India (belonging to the Indo-aryan family) share similarities with Hindi. Hence, if your target is one of these languages, data from Hindi can positively transfer to the target and improve its performance. This is more nuanced as similarity can be across multiple axes like morphology, phonology, syntax etc., and also depends on data quality. To train a model multilingually, we simply need to concatenate the data from both languages and train the model on this data. The idea is that the knowledge learned about the source, can also help the model to translate the target.

We provide scripts for training and evaluating a multilingual model for Belarusian, trained also on Polish (pol). Note that Polish is not necessarily the best transfer language for Belarusian (this is just a sample, you can explore other languages which can transfer better to Belarusian).

To start, run the following script to preprocess the data:

```
bash preprocess-ted-multilingual.sh
```

Then run training and evaluation to English.
```
bash traineval_belpol_eng.sh
```

You should be able to reproduce the numbers below on the validation set. To account for variance in experiments, a drop of 0.5 BLEU or 0.05 COMET are acceptable: 

| LP | BLEU | COMET |
| --- | --- | --- |
| bel-eng | 16.11 | -0.3967 |

With slight modifications in the script, you can obtain predictions for the test set. These prediction files need to be submitted to the kaggle leaderboard and can be converted to a CSV format compatible for submission using the `convert_to_csv.py` script. The benchmark submission for this baseline is titled `test-multilingual-prediction.csv` on the leaderboard. Please name your submission `<andrew_id>-multilingual`.

# Finetuning Pretrained Multilingual Models

Another option to improve the performance is to leverage massive multilingual pretrained models. These models were trained on gigantic corpora with over 100 languages and have been shown to improve performance on low-resource languages by extensively leveraging cross-lingual transfer across the languages considered.

In this assignment, we will consider finetuning the small FLORES-101 models on our low-resource languages.

To start, download the fairseq checkpoints for the model by running:

```
mkdir -p checkpoints && cd checkpoints
wget https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz
tar -xvf flores101_mm100_175M.tar.gz
cd ..

```

We provide scripts for preprocessing the data and finetuning the model for bel.

Note that we need to run preprocessing again since we need to use the original (subword) tokenization that the FLORES model was trained with. To preprocess the data, run:

```
bash preprocess-ted-flores-bilingual.sh
```

You can then finetune the model and evaluate its translation to English.
```
bash traineval_flores_bel_eng.sh
```

You should be able to reproduce the numbers below on the validation set. To account for variance in experiments, a drop of 0.5 BLEU or 0.05 COMET are acceptable: 

| LP | BLEU | COMET |
| --- | --- | --- |
| bel-eng | 24.46 | 0.0208 |

With slight modifications in the script, you can obtain predictions for the test set. These prediction files need to be submitted to the kaggle leaderboard and can be converted to a CSV format compatible for submission using the `convert_to_csv.py` script. The benchmark submission for this baseline is titled `test-flores-prediction.csv` on the leaderboard. Please name your submission `<andrew_id>-flores`.


# Improving Multilingual Training and Transfer


## Better Language-specific Architectures
There are many ways to improve the design of model architecture. 
One key bottleneck for multlingual modelling is the language interference in neural models. 
Yuan et al [1] develops a massive multiway detachable model (LegoMT).
Fan et al [2] and Costa-jussà et al [3] uses mixture-of-expert models (MoE) for multilingual MT. [4] introduce better training strategy for routing in experts. 
Lin et al [5] develops multilingual MT models to alliviate the interference explicity with learned langauge-specific sub-networks (LaSS).

Adapter  [6] [7] is a general method that introduces a new small set of parameters to the model when finetuning, leaving the original parameters fixed. Adapter finetuning has been shown to improve the performance of multilingual models when finetuning in new languages.
Zhang et. [8] propose adding language-aware modules in the NMT model.



## Data Augmentation
Extra monolingual data is often helpful for improving NMT performance. Specifically, back-translation [9,10] has been widely used to boost the performance of low-resource NMT. 
Creating mix-coded data for multilingual training is particularly useful, as in mRASP [11] and Xia et al. [12]. 

## Better Training Objectives
Pan et al [13] develops contrastive learning for multlingual many-to-many translation (mRASP2). 
[14] adds a masked language model objective for monolingual data while training a multilingual model.

## Using Pre-trained Language Models
Yang et al [15] uses a pre-trained BERT model in machine translation. Sun et al [16] combines a pre-trained multlingual BERT and multlingual GPT together and obtains a stronger mutlingual MT model. 
Liu et al [17] is another way to pre-train a sequence-to-sequence model using raw data and further fine-tune on mutlingual parallel data for translation. 


## Better Word Representation or Tokenization
Vocabulary differences between the low-resource language and its related high-resource language is an important bottleneck for multilingual transfer. 
Xu et al [18] formulates the vocabulary learning as an optimal transport problem. Wang et al [19] propose a character-aware embedding method for multilingual training. For morphological rich languages, such as Turkish and Azerbaijani, it is also useful to use the morphology information in word representations [20].

Recently, several approaches are proposed to improve the word segmentation for standard NMT models [21], [22]. It is possible that these improvements would be helpful for multilingual NMT as well.

## Choosing Transfer Languages
For the provided multilingual training method, we simply use a closely related high-resource language as a transfer language. However, it is likely that data from other languages would be helpful as well. Lin et al. [23] has done a systematic study of choosing transfer languages for NLP tasks. 


## Efficient Finetuning
Prefix-tuning [24] is an alternative finetuning paradigm that adds a parametrized prefix to inputs (embeddings) to the model and trains these prefixes, leaving the original model parameters fixed. 

## Other Approaches
There are many potential directions for improving multilingual training and finetuning. We encourage you to do more literature research, and even come up with your own method!

## Grading
Write and submit a report (one-pager) describing and analyzing the results. The style and format is flexible. It is expected for you to be concise in describing what you tried and why, and how the associated code can be run.

*Basic Requirement*: Reproduce the results with the bilingual baseline. To account for variance in experiments, a drop of 0.5 BLEU or 0.05 COMET are acceptable. This will earn you a passing B grade.

*Multilingual Baseline*: Reproduce results with the multilingual baseline. Try to think about why this improves performance and whether other transfer languages can be used. To account for variance in experiments, a drop of 0.5 BLEU or 0.05 COMET are acceptable. This will earn you a B+ grade. 

*Try pre-trained models*: Try to understand how multilingual pre-training helps performance. Reproducing the FLORES baseline and including why you get better performance using this model in your write-up will earn you a A- grade.

*Implement at least one pre-existing method to try to improve multilingual transfer*: Compare the performance of the implemented method with the baselines, clearly documenting results and briefly analysing why it does or does not work. This will earn you a A grade.

*Implement several methods to improve multilingual transfer*: For example, you can implement multiple pre-existing methods or one pre-existing method and one novel method. Compare the performance with the baselines, clearly documenting results and briefly analysing why it does or does not work. This will earn you a A+ for particularly extensive or interesting improvements and analysis.
If using existing code, please cite your sources.

## Submission
Your submission consists of three parts: *code*, *model outputs* and *writeup*. Put all your code in a folder named `code` and instructions on how to run if you have implemented additional code. Include the output of your models in an outputs directory, with a description of what model each file is associated with. Rename the writeup as `writeup.pdf` and compress all of them as `<andrew_id>-assn1.zip`. This file must be submitted to Canvas.

References
[1] Yuan et al. Lego-MT: Learning Detachable Models for Massively Multilingual Machine Translation. ACL 2023.

[2] Fan et al. Beyond english-centric multilingual machine translation. JMLR 2021.

[3] Costa-jussà et al. No language left behind: Scaling human-centered machine translation. 2022.

[4]: StableMoE: Stable Routing Strategy for Mixture of Experts. ACL 2022.

[5] Lin et al. Learning Language Specific Sub-network for Multilingual Machine Translation. ACL 2021.

[6] Bapna and Firat. Simple, Scalable Adaptation for Neural Machine Translation. EMNLP 2019.

[7] Zhu et al. Counter-Interference Adapter for Multilingual Machine Translation. EMNLP 2021.

[8] Zhang et al. Improving massively multilingual neural machine translation and zero-shot translation

[9]: Edunov et al. Understanding back-translation at scale.

[10]: Sennrich et al. Improving neural machine translation models with monolingual data

[11]: Lin et al. Pre-training Multilingual Neural Machine Translation by Leveraging Alignment Information, EMNLP 2020.

[12]: Xia et al. Generalized data augmentation for low-resource translation, 2019.

[13] Pan et al. Contrastive Learning for Many-to-many Multilingual Neural Machine Translation. ACL 2021. 

[14]: Siddhant et al. Leveraging monolingual data with self-supervision for multilingual neural machine translation.


[15] Yang et al. Towards Making the Most of BERT in Neural Machine Translation. AAAI 2019.

[16] Sun et al. Multilingual Translation via Grafting Pre-trained Language Models. EMNLP 2021

[17] Liu et al. Multilingual Denoising Pre-training for Neural Machine Translation. 2020

[18] Xu et al. Vocabulary Learning via Optimal Transport for Neural Machine Translation. ACL 2021.

[19] Wang et al. Multilingual neural machine translation with soft decoupled encoding

[20] Chaudhary et al. Adapting word embeddings to new languages with morphological and phonological subword

[21] Provilkov et al. BPE-dropout: Simple and effective subword regularization

[22] He et al. Dynamic programming encoding for subword segmentation in neural machine translation

[23] Lin et al. Choosing transfer languages for cross-lingual learning.

[24]: Li et al. Prefix-Tuning: Optimizing Continuous Prompts for Generation















