# ML-AIM: Machine Learning and Artificial Intelligence for Medicine

In this fork, I present some of my own adjustments to the original work. I have also developed a conditional MTS gan, [here](https://github.com/firmai/mtss-gan). 

This repository contains the implementations of algorithms developed
by the [ML-AIM](http://www.vanderschaar-lab.com) Laboratory.

1. [AutoPrognosis](https://icml.cc/Conferences/2018/Schedule?showEvent=2050): Automated Clinical Prognostic Modeling, ICML 2018 [software](alg/autoprognosis)
2. [GAIN](http://proceedings.mlr.press/v80/yoon18a.html): a GAN based missing data imputation algorithm, ICML 2018 [software](alg/gain)
3. [INVASE](https://openreview.net/forum?id=BJg_roAcK7): an Actor-critic model based instance wise feature selection algorithm, ICLR 2019 [software](alg/invase)
4. [GANITE](https://openreview.net/forum?id=ByKWUeWA-): a GAN based algorithm for estimating individualized treatment effects, ICLR 2018 [software](alg/ganite)
5. [DeepHit](http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit): a Deep Learning Approach to Survival Analysis with Competing Risks, AAAI 2018 [software](alg/deephit)
6. [PATE-GAN](https://openreview.net/forum?id=S1zk9iRqF7): Generating Synthetic Data with Differential Privacy Guarantees, ICLR 2019 [software](alg/pategan)
7. [KnockoffGAN](https://openreview.net/pdf?id=ByeZ5jC5YQ): generating knockoffs for feature selection using generative adversarial networks, ICLR 2019 [software](alg/knockoffgan)
8. [Causal Multi-task Gaussian Processes](https://papers.nips.cc/paper/6934-bayesian-inference-of-individualized-treatment-effects-using-multi-task-gaussian-processes.pdf): Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes, NIPS 2017 [software](alg/causal_multitask_gaussian_processes_ite)
9. [Limits of Estimating Heterogeneous Treatment Effects:Guidelines for Practical Algorithm Design](http://proceedings.mlr.press/v80/alaa18a/alaa18a.pdf)[software](alg/causal_multitask_gaussian_processes_ite)
10. [ASAC](https://arxiv.org/abs/1906.06796): Active Sensing using Actor-Critic Models, MLHC 2019 [software](alg/asac)
11. [DGPSurvival](https://papers.nips.cc/paper/6827-deep-multi-task-gaussian-processes-for-survival-analysis-with-competing-risks.pdf): Deep Multi-task Gaussian Processes for Survival Analysis with Competing Risks, NIPS 2018 [software](alg/dgp_survival)
12. [Symbolic Metamodeling](https://papers.nips.cc/paper/9308-demystifying-black-box-models-with-symbolic-metamodels) Demystifying Black-box Models with Symbolic Metamodels, NeurIPS 2019 [software](alg/symbolic_metamodeling)
13. [DPBAG](https://papers.nips.cc/paper/8684-differentially-private-bagging-improved-utility-and-cheaper-privacy-than-subsample-and-aggregate) Differentially Private Bagging: Improved utility and cheaper privacy than subsample-and-aggregate, NeurIPS 2019 [software](alg/dpbag)
14. [TimeGAN](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks) Time-series Generative Adversarial Networks, NeurIPS 2019 [software](alg/timegan)
15. [Attentiveness](https://papers.nips.cc/paper/9311-attentive-state-space-modeling-of-disease-progression) Attentive State-Space Modeling of Disease Progression, NeurIPS 2019 [software](alg/attentivess)
16. [GCIT](https://arxiv.org/pdf/1907.04068.pdf): Conditional Independence Testing with Generative Adversarial Networks, NeurIPS 2019 [software](alg/gcit)
17. [Counterfactual Recurrent Network](https://openreview.net/forum?id=BJg866NFvB): Estimating counterfactual treatment outcomes over time through adversarially balanced representations, ICLR 2020 [software](alg/counterfactual_recurrent_network)
18. [C3T Budget](https://arxiv.org/abs/2001.02463): Contextual constrained learning for dose-finding clinical trials, AISTATS 2020 [software](alg/c3t_budgets)
19. [DKLITE](https://arxiv.org/abs/2001.04754): Learning Overlapping Representations for the Estimation of Individualized Treatment Effects, AISTATS 2020 [software](alg/dklite)
20. [Dynamic disease network ddp](https://arxiv.org/abs/2001.02585): Learning Dynamic and Personalized Comorbidity Networks from Event Data using Deep Diffusion Processes, AISTATS 2020 [software](alg/dynamic_disease_network_ddp)
21. [SMS-DKL](https://arxiv.org/abs/2001.03898): Stepwise Model Selection for Sequence Prediction via Deep Kernel Learning, AISTATS 2020 [software](alg/smsdkl)

Prepared for release and maintained by AvdSchaar

Please send comments and suggestions to mihaelaucla@gmail.com

## Citations

Please cite the [ML-AIM repository](https://bitbucket.org/mvdschaar/mlforhealthlabpub) and or the applicable papers if you use the software.

## License

Copyright 2019, 2020, ML-AIM

The ML-AIM software is released under the [3-Clause BSD license](https://opensource.org/licenses/BSD-3-Clause) unless mentioned otherwise by the respective algorithms.

## [Installation instructions](doc/install.md)

See doc/install.md for installation instructions

## Tutorials and or examples

* AutoPrognosis:
--  alg/autoprognosis/tutorial_autoprognosis_api.ipynb
--  alg/autoprognosis/tutorial_autoprognosis_cli.ipynb
* GAIN: alg/gain/tutorial_gain.ipynb
* INVASE: alg/invase/tutorial_invase.ipynb
* GANITE: alg/ganite/tutorial_ganite.ipynb
* PATE-GAN: alg/pategan/tutorial_pategan.ipynb
* KnockoffGAN: alg/knockoffgan/tutorial_knockoffgan.ipynb
* ASAC: alg/asac/tutorial_asac.ipynb
* DGPSurvival: alg/dgp_survival/tutorial_dgp.ipynb
* Symbolic Metamodeling:
-- alg/symbolic_metamodeling/1-_Introduction_to_Meijer_G-functions.ipynb
-- alg/symbolic_metamodeling/2-_Metamodeling_of_univariate_black-box_functions_using_Meijer_G-functions.ipynb
-- alg/symbolic_metamodeling/3-_Building_Symbolic_Metamodels.ipynb
* Differentially Private Bagging: alg/dpbag/DPBag_Tutorial.ipynb
* Time-series Generative Adversarial Networks: alg/timegan/tutorial_timegan.ipynb
* Attentive State-Space Modeling of Disease Progression: alg/attentivess/Tutorial_for_Attentive_State-space_Models.ipynb
* Conditional Independence Testing with Generative Adversarial Networks: alg/gcit/tutorial_gcit.ipynb
* DKLITE: alg/dklite/tutorial_dklite.ipynb
* SMS-DKL: alg/smsdkl/test_smsdkl.py

### [Presentation Autoprognosis](https://www.youtube.com/watch?v=d1uEATa0qIo)

You can find a presentation by Prof. van der Schaar describing AutoPrognosis here: https://www.youtube.com/watch?v=d1uEATa0qIo

## Version history

- version 1.7: February 27, 2020: SMS-DKL
- version 1.6: February 24, 2020: DKLITE and dynamic disease network ddp
- version 1.5: February 23, 2020: C3T Budget
- version 1.4: February 3, 2020: Counterfactual Recurrent Network
- version 1.3: December 7, 2019: Conditional Independence Testing with Generative Adversarial Networks
- version 1.1: November 30, 2019: Attentive State-Space Modeling
- version 1.0: November 4, 2019: Differentially Private Bagging and Time-series Generative Adversarial Networks
- version 0.9: October 25, 2019: Symbolic Metamodeling
- version 0.8: September 29, 2019: DGP Survival
- version 0.7: September 20, 2019: ASAC
- version 0.6: August 5, 2019: Causal Multi-task Gaussian Processes
- version 0.5: July 24, 2019: KnockoffGAN
- version 0.4: June 18, 2019: Deephit and PATE-GAN

## References
1. [AutoPrognosis: Automated Clinical Prognostic Modeling via Bayesian Optimization with Structured Kernel Learning](https://icml.cc/Conferences/2018/Schedule?showEvent=2050)
2. [Prognostication and Risk Factors for Cystic Fibrosis via Automated Machine Learning](https://www.nature.com/articles/s41598-018-29523-2)
3. [Cardiovascular Disease Risk Prediction using Automated Machine Learning: A Prospective Study of 423,604 UK Biobank Participants](https://www.ncbi.nlm.nih.gov/pubmed/31091238)
4. [GAIN: Missing Data Imputation using Generative Adversarial Nets](http://proceedings.mlr.press/v80/yoon18a.html)
5. [INVASE: Instance-wise Variable Selection using Neural Networks](https://openreview.net/forum?id=BJg_roAcK7)
6. [GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets](https://openreview.net/forum?id=ByKWUeWA-)
7. [KnockoffGAN](https://openreview.net/pdf?id=ByeZ5jC5YQ): generating knockoffs for feature selection using generative adversarial networks
8. [Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes](https://papers.nips.cc/paper/6934-bayesian-inference-of-individualized-treatment-effects-using-multi-task-gaussian-processes.pdf)
9. [Limits of Estimating Heterogeneous Treatment Effects:Guidelines for Practical Algorithm Design](http://proceedings.mlr.press/v80/alaa18a/alaa18a.pdf)
10. [ASAC](https://arxiv.org/abs/1906.06796) Active Sensing using Actor-Critic Models
11. [DGPSurvival](https://papers.nips.cc/paper/6827-deep-multi-task-gaussian-processes-for-survival-analysis-with-competing-risks.pdf): Deep Multi-task Gaussian Processes for Survival Analysis with Competing Risks
12. [GCIT](https://arxiv.org/pdf/1907.04068.pdf): Conditional Independence Testing with Generative Adversarial Networks
13. [Counterfactual Recurrent Network](https://openreview.net/forum?id=BJg866NFvB): Estimating counterfactual treatment outcomes over time through adversarially balanced representations
14. [C3T Budget](https://arxiv.org/abs/2001.02463): Contextual constrained learning for dose-finding clinical trials
15. [SMS-DKL](https://arxiv.org/abs/2001.03898): Stepwise Model Selection for Sequence Prediction via Deep Kernel Learning
16. Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science.
17. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
18. [TensorFlow](tensorflow.org): Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.
19. [GPyOpt](http://github.com/SheffieldML/GPyOpt): A Bayesian Optimization framework in python
20. [scikit-survival](https://github.com/sebp/scikit-survival) survival analysis built on top of scikit-learn
21. [3-Clause BSD license](https://opensource.org/licenses/BSD-3-Clause)
