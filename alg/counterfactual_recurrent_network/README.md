# [Estimating counterfactual treatment outcomes over time through adversarially balanced representations](https://openreview.net/forum?id=BJg866NFvB)

### Ioana Bica, Ahmed M. Alaa, James Jordon, Mihaela van der Schaar

#### International Conference on Learning Representations (ICLR) 2020

Last Updated Date: January 15th 2020

Code Author: Ioana Bica (ioana.bica95@gmail.com)

The CRN (Counterfactual Recurrent Network) estimates the effects of treatments assigned over time. The CRN model consists of an encoder performing one-step ahead prediction of treatment responses and a decoder estimating the effects of sequences of treatments. 

Outputs:

   - RMSE for one-step-ahead prediction.  
   
   - RMSE for five-step-ahead prediction. 
   
   - Trained encoder and decoder models. 

The CRN uses data from a Pharmacokinetic-Pharmacodynamic model of tumour growth which 
which is a bio-mathematical model that simulates the combined effects of 
chemotherapy and radiotherapy in non-small cell lung cancer patients
(Geng et al 2017, https://www.nature.com/articles/s41598-017-13646-z). The same simulation model was used by Lim et al. 2018 
(https://papers.nips.cc/paper/7977-forecasting-treatment-responses-over-time-using-recurrent-marginal-structural-networks.pdf)
We adopt their implementation from: https://github.com/sjblim/rmsn_nips_2018 and extend it to incorporate counterfactual outcomes. 

The chemo_coeff and radio_coeff in the simulation specify the amount of time-dependent confounding
for the chemotherapy and radiotherapy applications. The results in the paper were obtained by varying the
chemo_coeff and radio_coeff, and thus obtaining different datasets. For gamma=chemo_coeff=radio_coeff in {1, 2, 3, 4, 5} there are results for both one-step-ahead 
prediction and sequence prediction in the paper (Figure 4).

The synthetic dataset for each setting of chemo_coeff and radio_coeff is over 1GB in size, which is why it is re-generated every time the code is run. 

### To test the Counterfactual Recurrent Network, run (this will use a default settings of hyperparameters):

python test_crn.py --chemo_coeff=2 --radio_coeff=2 --model_name=crn_test_2

### To perform hyperparameter optimization and test the Counterfactual Recurrent Network , run:

python test_crn.py --chemo_coeff=2 --radio_coeff=2 --model_name=crn_test_2 --b_encoder_hyperparm_tuning=True --b_decoder_hyperparm_tuning=True

For the results in the paper, hyperparameter optimization was run (this can take about 8 hours on an
NVIDIA Tesla K80 GPU). 
 
