# Multimodal-finetuning-ImgText-pair
This project is about creating a multimodal model which can embed the product title and Images of an product in same embedding space.
But why?
First and foremost, this will help to flag out wrong image-title pair, which can lead to very bad customer shopping experience and negative trust to the platform.
Secondly, this is a common practice in C2C ecommerce platfolms to tamper the competitor sellers' product page to unfairly get advantage in the competition. So, a good embedding model which can detect fake image/title when bad actors are trying to upload wrong informations can save a good seller from both potentially loss of business and bad reputation.

The approach to tackle this problem?
Here I chose the CLIP a.k.a Contrastive Language Image Pair model. As the name suggests, this model leverages the contrastive learning approach to embed positive samples closer in vector space while embedding the negative samples further from each other. For negative sampling used hard negative mining technique (using samples closer to the anchor, which are images in this case but not the actual title related to it). Used the sentence_transformers.util.mine_hard_negatives() function to do the sampling.
For evaluation I used recall@k where K=1 and used multiplenegativerankinglosss as the preferred loss function.
