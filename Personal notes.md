# Personal notes

1. Do we need to include feature importance? Seems not
2. PCA might not be working when training GAN model. Consider make one more version of dataset that does not apply the PCA.
3. GAN: Epoch 100 & Batch size 1024 -> Seriously overfitted
4. GAN: Epoch 25 & Batch size 1024 -> Little bit underfit
5. GAN: Epoch 25 Batch size 128 -> Completely underfit
6. GAN: Epoch 16 Batch size 128 -> overfitted
7. GAN: Epoch 50 & Batch size 1024 -> overfitted
8. GAN: Epoch 30 & Batch size 1024 -> underfit, but improved <<-- Best so far
9. GAN: Epoch 35 & Batch size 1024 -> little bit overfit 
10. GAN: Epoch 40 & Batch size 1024 -> overfit

# Few problems:
1. BGRU hyper param: 100 epochs, 10 early stop patient, dropout 0.3 & 0.5 cause underfitting 