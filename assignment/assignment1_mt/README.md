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


# Improving Multilingual Transfer

## Data Augmentation
Extra monolingual data is often helpful for improving NMT performance. Specifically, back-translation [2,3] has been widely used to boost the performance of low-resource NMT. A closely related method, self-training, is recently proven to be effective as well [4]. Recently, several methods have been proposed to combine different methods of using monolingual data with multilingual training. Xia et al. [5] explores several different strategies for modifying the related language data to improve multilingual training. [6] adds a masked language model objective for monolingual data while training a multilingual model.

## Choosing Transfer Languages
For the provided multilingual training method, we simply use a closely related high-resource language as a transfer language. However, it is likely that data from other languages would be helpful as well. Lin et al. [7] has done a systematic study of choosing transfer languages for NLP tasks. There are also methods designed to choose multilingual data for specific NLP tasks [8].

## Better Word Representation or Segmentation
Vocabulary differences between the low-resource language and its related high-resource language is an important bottleneck for multilingual transfer. Wang et al [9] propose a character-aware embedding method for multilingual training. For morphological rich languages, such as Turkish and Azerbaijani, it is also useful to use the morphology information in word representations [10].

Recently, several approaches are proposed to improve the word segmentation for standard NMT models [11, 12]. It is possible that these improvements would be helpful for multilingual NMT as well.

## Better Modeling
You can also improve the NMT architecture or optimization to better model data from multiple languages. Wang et al. [13] propose three strategies to improve one-to-many translation, including better initialization and language-specific embedding. Zhang et. [14] propose adding language-aware modules in the NMT model.

## Efficient Finetuning
While pretrained multilingual models are generally more data-efficient efficient than bilingual models trained from scratch, they are still relatively data-hungry and can result in poor performance in extremely low-resource settings. Recently several approaches have been proposed to improve both data and parameter efficiency of pretrained machine translation models.

Adapter finetuning [15] is a general method that introduces a new small set of parameters to the model when finetuning, leaving the original parameters fixed. Adapter finetuning has been shown to improve the performance of multilingual models when finetuning in new languages [16].

Prefix-tuning [17] is an alternative finetuning paradigm that adds a parametrized prefix to inputs (embeddings) to the model and trains these prefixes, leaving the original model parameters fixed. Prefix-tuning can be even more parameter and data-efficient than adapter finetuning [18], but little research has been done in applying prefix-tuning to machine translation.

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
[1]: Qi et al. When and Why are pre-trained word embeddings useful for Neural Machine Translation

[2]: Edunov et al. Understanding back-translation at scale.

[3]: Sennrich et al. Improving neural machine translation models with monolingual data

[4]: He et al. Revisiting self-training for neural sequence generation

[5]: Xia et al. Generalized data augmentation for low-resource translation

[6]: Siddhant et al. Leveraging monolingual data with self-supervision for multilingual neural machine translation

[7]: Lin et al. Choosing transfer languages for cross-lingual learning

[8]: Wang et al. Target conditioned sampling: Optimizing data selection for multilingual neural machine translation

[9]: Wang et al. Multilingual neural machine translation with soft decoupled encoding

[10]: Chaudhary et al. Adapting word embeddings to new languages with morphological and phonological subword

[11]: Provilkov et al. BPE-dropout: Simple and effective subword regularization

[12]: He et al. Dynamic programming encoding for subword segmentation in neural machine translation

[13]: Wang et al. Three strategies to improve one-to-many multilingual translation

[14]: Zhang et al. Improving massively multilingual neural machine translation and zero-shot translation

[15]: Houlsby et al. Parameter-Efficient Transfer Learning for NLP

[16]: Philip et al. Monolingual Adapters for Zero-Shot Neural Machine Translation

[17]: Li et al. Prefix-Tuning: Optimizing Continuous Prompts for Generation

[18]: He et al. Towards a Unified View of Parameter-Efficient Transfer Learning













