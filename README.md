# Development of a Software Application for Drug Sensitivity Prediction

This repository contains the results of the thesis titled "Development of a software application based on deep learning to predict drug sensitivity of cancer cell lines". The thesis explores two study cases: "Single input drug sensitivity prediction" and "Predicting drug synergy in cancer cell lines".

## Study Case 1: Single Input Drug Sensitivity Prediction

This study case investigates the impact of different compound representations on drug sensitivity prediction. The folder "Drug_Sensitivity_Prediction" contains the following files:

- `drug_sensitivity.py`: Python script to run each experiment and evaluate the performance of different compound representations.
- `data`: Folder containing the data and splits used in the experiments.
- `results.xlsx`: Excel file presenting the results of the experiments.

The experiments conducted in this study case replicate the experiments described in the paper [1].

## Study Case 2: Predicting Drug Synergy in Cancer Cell Lines

This study case focuses on evaluating the effect of several methodological variables in predicting drug synergy. The folder "Drug_Synergy_Prediction" includes the following files:

- `drug_synergy.py`: Python script used to perform each experiment.
- `results.xlsx`: Excel file containing the results obtained from the experiments.

The experiments conducted in this study case replicate the experiments described in the paper [2]. To generate the necessary data and splits for this study case, please refer to the scripts provided by the authors. These scripts are available at https://github.com/BioSystemsUM/drug_response_pipeline/blob/master/almanac/README.md.

## References

[1] Reference for Study Case 1: [A Comparison of Different Compound Representations for Drug Sensitivity Prediction](https://link.springer.com/chapter/10.1007/978-3-030-86258-9_15)

[2] Reference for Study Case 2: [A systematic evaluation of deep learning methods for the prediction of drug synergy in cancer](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010200)
