# Contrastive Learning
This repo demostrates how to use the concept of contrastive learning in an anommaly detection setting with autoencoders (also know as discriminative autoencoders). The code is explained in my post https://medium.com/@santiagof/contrastive-learning-effective-anomaly-detection-with-auto-encoders-98c6e1a78ada

Discriminative autoencoders aim at learning low-dimensional discriminative representations for positive (X+) and negative (X−) classes of data. The discriminative autoencoders build a latent space representation under the constraint that the positive data should be better reconstructed than the negative data. This is done by minimizing the reconstruction error for positive examples while ensuring that those of the negative class are pushed away from the manifold.

This implementation is based on S. Razakarivony and F. Jurie, "Discriminative autoencoders for small targets detection," in Pattern Recognition (ICPR), 2014
ref: https://arxiv.org/pdf/1801.03149.pdf


