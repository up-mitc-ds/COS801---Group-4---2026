# COS801---Group-4---2026
Group 4: Topic number 3: Sentiment Detection in Code-Switched Social Media Text

# African-Language Sentiment Analysis: Monolingual and Code-Switching Evaluation

### Project Overview

This project investigates the performance of multilingual transformer models for sentiment classification across selected African languages, with a particular focus on cross-language generalisation and code-switching.
Using the publicly available Masakhane AfriSenti dataset, the study evaluates whether African-language-focused multilingual encoders outperform a general-purpose multilingual encoder, examines performance differences across language groups, and investigates the impact of multilingual training data augmentation.

## Research Questions
RQ1: To what extent do pretrained African-language-focused multilingual encoders outperform a general-purpose multilingual encoder on code-switched sentiment classification across the selected African languages?

RQ2: How does sentiment classification performance vary across the selected languages, and between monolingual and code-switched text, within each model?

RQ3: Does synthetic code-switching augmentation improve sentiment classification performance on code-switched text without degrading performance on the corresponding monolingual text?


Important: The implemented augmentation experiment uses authentic multilingual training data rather than synthetically generated code-switched tweets. Consequently, RQ3 is only partially addressed.

### Dataset

Source: Masakhane AfriSenti — Hugging Face
AfriSenti provides sentiment-labelled social media text in African languages.

The sentiment classification task consists of three classes:
•	Negative
•	Neutral
•	Positive

### Language Selection
An initial screening exercise involved manually annotating 100 tweets per language across 10 African languages.
Tweets were classified as:
•	MONO - Monolingual
•	CS - Code-switched
•	UNCERTAIN - Insufficient evidence for a confident classification

Six languages were subsequently selected based on their observed language characteristics.
Language group	Selected languages
Monolingual-heavy:	Amharic, Algerian Arabic, Swahili
Code-switching-heavy:	Igbo, Nigerian Pidgin, Twi

These classifications represent language-level tendencies, not verified code-switching labels for every tweet in the complete datasets.

### Models Evaluated
Three pretrained transformer architectures were evaluated.
Model	Category
XLM-R	General-purpose multilingual encoder

AfriBERTa	African-language-focused encoder

AfroXLM-R	African-language-focused multilingual encoder


Each model was fine-tuned for three-class sentiment classification.


### Methodology
1. Data Preparation
The selected AfriSenti language datasets were loaded and prepared for model training and evaluation.
Preprocessing included:
•	Text cleaning and Unicode normalisation.
•	Handling URLs, mentions and unnecessary whitespace.
•	Preserving relevant multilingual text characteristics.
•	Standardising sentiment labels.
•	Preparing separate training, validation and test datasets.
2. Baseline Model Training
Each of the three transformer architectures was independently fine-tuned on each of the three monolingual-heavy languages.
This produced nine baseline model configurations:
3 architectures × 3 training languages = 9 configurations
The objective was to establish language-specific baseline performance and investigate generalisation to other African languages.
3. Cross-Language Evaluation
The trained baseline models were evaluated on the selected code-switching-heavy language datasets.
Performance was assessed on both pooled and individual target-language test datasets.
This enabled comparisons of:
•	Performance across model architectures.
•	Differences between training and target languages.
•	Cross-language sentiment classification generalisation.
•	Performance across sentiment classes.
4. Multilingual Data Augmentation
A further experiment investigated whether expanding the training data to include authentic examples from both language groups would improve classification performance.
The augmented training dataset combined:
•	Monolingual-heavy language training data.
•	Code-switching-heavy language training data.
The selected model architecture was trained on this combined multilingual dataset and evaluated on the code-switching-heavy test datasets.
This is an authentic multilingual augmentation experiment, not synthetic code-switching generation.
5. Performance Evaluation
The evaluation considered:
•	Macro-F1: Primary metric, treating all sentiment classes equally.
•	Accuracy: Overall classification correctness.
•	Weighted-F1: Class-frequency-weighted classification performance.
•	Class-level precision, recall and F1: Additional diagnostic measures.

## Key Findings
RQ1: Encoder Performance
African-language-focused encoders demonstrated promising performance across the selected monolingual-heavy languages.
AfroXLM-R achieved the strongest reported monolingual Macro-F1 results for Algerian Arabic and Swahili, while AfriBERTa performed best on Amharic.
The findings suggest that African-language-focused models can offer advantages over general-purpose multilingual encoders, although performance remains language-dependent.

RQ2: Cross-Language Generalisation
Sentiment classification performance varied across the selected languages.
Models trained on one language did not consistently maintain their performance when evaluated on other languages.
For example, the Algerian Arabic-trained AfroXLM-R achieved a Macro-F1 of 0.5750 on its monolingual test dataset, compared with 0.4082 on the pooled code-switching-heavy test dataset.
This highlights the challenges of cross-language transfer.
However, differences between these datasets cannot be attributed exclusively to code-switching because language identity and other dataset characteristics also differ.

RQ3: Multilingual Augmentation
Authentic multilingual augmentation improved Macro-F1 across all three selected code-switching-heavy target languages.
Target language	Baseline Macro-F1	Augmented Macro-F1	Absolute improvement
Igbo	0.3571	0.7402	+0.3831
Nigerian Pidgin	0.4425	0.4686	+0.0261
Twi	0.3068	0.5040	+0.1972


The pooled Macro-F1 increased from 0.4082 to 0.6941, representing an absolute improvement of 0.2859.
The largest improvement was observed for Igbo, followed by Twi and Nigerian Pidgin.
These findings suggest that broader multilingual training exposure can substantially improve sentiment classification performance on the selected target datasets.
The experiment does not establish whether improvements resulted specifically from code-switching exposure, increased training data volume or greater language diversity.
Predictability, Computability and Stability (PCS)
The study also considers the PCS framework.
Predictability: Evaluates model performance using sentiment classification metrics and cross-language comparisons.
Computability: Considers the feasibility of fine-tuning and evaluating transformer models within the available computational environment.
Stability: Examines performance sensitivity to changes in language and evaluation datasets.
Repeated-run statistical stability and comparative computational efficiency were not comprehensively established.

### Limitations
The study has several important limitations:
1.	Language-level grouping: The selected MONO-heavy and CS-heavy datasets are proxies for code-switching prevalence, rather than tweet-level code-switching evaluation sets.
2.	Training data differences: Baseline and augmented experiments use different training data volumes and language coverage.
3.	Synthetic augmentation: Synthetic code-switched tweets were not generated in the implemented augmentation experiment.
4.	Monolingual retention: The documented augmentation comparison does not establish whether performance on the original monolingual-heavy languages was preserved.
5.	Experimental stability: Additional training seeds and controlled experiments would strengthen the reliability of the conclusions.


### Future Work
Future extensions could include:
•	Constructing a manually verified tweet-level code-switching test dataset.
•	Generating and validating synthetic code-switched training examples.
•	Comparing authentic and synthetic augmentation under controlled training conditions.
•	Evaluating monolingual performance retention following augmentation.
•	Repeating experiments across multiple random seeds.
•	Investigating language-specific factors influencing model performance.
Technologies

#### The experimental implementation uses:
•	Python
•	Google Colab
•	Hugging Face Transformers
•	Hugging Face Datasets
•	Scikit-learn
•	Pandas
•	NumPy

### Reproducibility

#### To reproduce the experiments:
1.	Open the relevant Jupyter notebooks in Google Colab.
2.	Enable a GPU runtime where available.
3.	Install the required Python dependencies.
4.	Load the Masakhane AfriSenti dataset.
5.	Prepare the selected language datasets.
6.	Train and evaluate the baseline models.
7.	Train the multilingual augmented model.
8.	Compare model performance using the reported evaluation metrics.
The experiments use a fixed random seed of 42 where configured.
Model performance may vary across environments and software versions.


### Guide to running the notebook:



### Dataset Acknowledgement

This project uses the publicly available Masakhane AfriSenti dataset, developed through collaborative research efforts supporting natural language processing for African languages.

Dataset: https://huggingface.co/datasets/masakhane/afrisenti

Project Context

Module: COS801

Research Area: Natural Language Processing, African Languages, Sentiment Analysis and Code-Switching

Project Type: Academic Research

### Authors:
Kgoboko Maripane
Nqobile Ngidi
Phakisho Kupa

N.B:
This repository supports reproducibility and further investigation of multilingual sentiment classification and cross-language transfer in African-language NLP.


