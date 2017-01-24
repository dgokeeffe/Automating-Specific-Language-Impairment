# SLI_ML
A ML framework for a binary classification problem. Specific language impairment (SLI) is an often difficult to diagnose disorder which commonly affects young children. This program explores the potential to automate this diagnosis using advances in NLP and Machine Learning over the last decade. Using the TnT tagger trained on the Brown corpus and the CHILDES program CLAN, this program extracts several lexical and morphological features from three corpora available in CHILDES Talkbank, and runs them through various ML algorithms.

There are five corpora in total:
  - Conti4
  - ENNI
  - Gillam
  - ENNI + Gillam
  - All 3 combined

Conti4 uses LOOCV because N=119, the rest use k-fold validation = 10.
