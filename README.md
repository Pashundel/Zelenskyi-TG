# Replication materials for the thesis

This repository contains the replication materials for the thesis:

**Framing Wartime Leadership: Zelenskyi’s Telegram Communication During Russia’s Full-Scale Invasion of Ukraine**

The repository includes the Python scripts, intermediate CSV files, LDA model outputs, and the Excel workbook used to produce the descriptive tables reported in the thesis. The materials are provided to make the data-processing, topic-modelling, and descriptive aggregation workflow transparent and verifiable. <ins>For the convenience, some files have been compressed, please use unarchivator to unpack and use files with the .zip extensions.</ins>

## 1. Repository structure

### `scripts/`

This folder contains the Python scripts used during the computational part of the analysis. The scripts cover the main stages of the pipeline: conversion of Telegram export files into CSV format, date filtering, phase assignment, text cleaning, tokenisation, lemmatisation, preparation of the LDA-ready corpus, and LDA model estimation.

The scripts should be read as the computational workflow used to move from the raw Telegram export to the final topic-model outputs.

### `csv_outputs/`

This folder contains intermediate and final CSV files produced during preprocessing. These files document the transformation of the Telegram export into an analytical dataset.

The key files include cleaned posts, phased posts, lemmatised posts, and the final LDA-ready corpus. The exact filenames may differ by processing stage, but the general sequence is:

1. raw Telegram JSON export;
2. structured CSV file filtered to the thesis period;
3. CSV file with assigned invasion phases;
4. CSV file with cleaned text;
5. CSV file with tokenised and lemmatised text;
6. final CSV file prepared for LDA modelling.

### `lda_outputs/`

This folder contains the outputs produced by the LDA grid-search script. The script estimates LDA models for alternative topic numbers, calculates model-selection metrics, exports topic-word outputs and document-topic matrices, and saves technical model artefacts for selected candidate models.

The main files in this folder are described below.

`lda_metrics.csv` contains model-selection metrics for each tested number of topics. It includes the number of topics, log likelihood, perplexity, C_v coherence, number of documents, and vocabulary size. This file was used to compare candidate LDA models.

`lda_model_selection_summary.json` summarises the model-selection run. It records the input file, token column, number of input rows, number of rows used for LDA, minimum document-token threshold, vocabulary size, best k by coherence, best k by perplexity, and the inspected k.

`lda_coherence_plot.png` visualises C_v coherence across candidate topic numbers. It was used as one of the diagnostics for selecting the final topic model.

`lda_perplexity_plot.png` visualises perplexity across candidate topic numbers. It was used as a model-fit diagnostic alongside coherence and substantive interpretability.

`lda_topics_k15.csv` contains the top terms and weights for each topic in the selected 15-topic model. This file was used to interpret and label the LDA topics as provisional frames.

`lda_doc_topics_k15.csv` contains the document-topic matrix for the selected 15-topic model. Each retained post receives a topic-proportion vector and a dominant-topic assignment. This file was used as the basis for later descriptive aggregation in Excel.

The files with names such as `lda_model_k*.joblib`, `vectorizer_k*.joblib`, `dtm_k*.npz`, and `kept_idx_k*.pkl` are technical reproducibility artefacts for selected candidate topic models. They allow the trained models, vectorisers, document-term matrices, and retained row indices to be reloaded without rerunning the full grid-search process.

The files `pyldavis_k*.html` are interactive pyLDAvis visualisations for selected candidate topic models. They were used to inspect topic separation, overlap, and salient terms during model interpretation.

### `analysis_workbook.xlsx`

This Excel workbook was used for descriptive aggregation after the LDA outputs were generated. It combines the selected LDA output with post metadata, phase labels, topic labels, topic-family mapping, and Telegram reaction data.

The workbook was used to calculate the descriptive tables reported in the Results chapter, including:

- overall topic prevalence;
- topic prevalence by invasion phase;
- topic-family shares;
- reaction volume by topic;
- reaction volume by topic family;
- emoji reaction composition by topic.

The calculations in this workbook are descriptive. They are based on formulas and/or pivot tables and are included to make the aggregation logic transparent and verifiable.

### `README.md`

This file explains the structure of the repository and the purpose of the main files.

## 2. Analytical workflow

The analysis followed the sequence below.

First, the Telegram channel export was converted from JSON into structured CSV format. During this stage, posts were filtered to retain only entries published between 24 February 2022 and 31 December 2025.

Second, each post was assigned to one of the invasion phases used in the thesis. The phase variable was later used for temporal comparison of topic prevalence.

Third, the text was cleaned by removing URLs and mentions, normalising hashtags, lowercasing text, reducing repeated punctuation, and normalising whitespace.

Fourth, the cleaned text was tokenised and lemmatised. The resulting lemmas and part-of-speech tags were used to prepare the LDA-ready corpus.

Fifth, the LDA-ready corpus was prepared by filtering tokens according to part of speech, stopword lists, token length, document frequency, and document-token thresholds.

Sixth, LDA models were estimated across a grid of topic numbers. Models were evaluated using perplexity, C_v coherence, and substantive interpretability. The selected model used 15 topics.

Seventh, the final LDA outputs were exported. These included topic-word distributions, document-topic distributions, dominant-topic assignments, model-selection metrics, and visual diagnostic files.

Finally, the selected LDA output was combined with phase labels and Telegram reaction data in Excel. The Excel workbook was used to calculate the final descriptive tables on topic prevalence and platform-visible audience response.

## 3. Notes on topic numbering

The raw LDA output uses zero-based topic numbering. This means that `topic_0` in the exported LDA files corresponds to Topic 1 in the thesis, `topic_1` corresponds to Topic 2, and so on.

For readability, the thesis reports topics using one-based numbering: Topic 1 to Topic 15.

## 4. Notes on Excel aggregation

The topic-prevalence and reaction-analysis tables in the thesis were not generated by a separate Python script. They were calculated in Excel from the final LDA outputs and analytical dataset.

The Excel workbook is included in this repository to allow verification of:

- source data used for aggregation;
- topic-family mapping;
- phase assignment used in descriptive tables;
- formulas or pivot tables used to calculate shares, means, and reaction composition;
- final tables used in the thesis.

## 5. Reproducibility limitations

The repository is intended to make the analytical workflow transparent and verifiable. However, several limitations should be noted.

First, Telegram reaction data are platform-visible behavioural signals and should not be interpreted as direct measures of public opinion, persuasion, or full audience interpretation.

Second, the dataset does not include exposure metrics such as post views or subscriber counts at the time of publication. Therefore, reaction volume is interpreted descriptively rather than as an engagement rate.

Third, LDA topics are treated as provisional frames. They are computationally derived semantic clusters that require interpretive validation and should not be treated as full theoretical frames automatically.

Fourth, some descriptive aggregation was conducted in Excel rather than in Python. The workbook is included to make those calculations visible and checkable.

## 6. Suggested replication order

To reproduce or inspect the analysis, follow this order:

1. Review the scripts in `scripts/` and use the `results.json` as a source file to replicate.
2. Review or replicate the intermediate CSV files in `csv_outputs/`.
3. Review or replicate `lda_metrics.csv`, `lda_coherence_plot.png`, and `lda_perplexity_plot.png` in `lda_outputs_final/`.
4. Review or replicate `lda_topics_k15.csv` to inspect the topic-word output.
5. Review or replicate `lda_doc_topics_k15.csv` to inspect document-topic assignments.
6. Open `analysis_workbook.xlsx` to inspect descriptive aggregation and final tables.
7. Use `pyldavis_k*.html` files for interactive inspection of candidate topic models.
