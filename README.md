
#                                                    Image Colorization Using Conditional GANs

These are the final results obtained after performing iterations over 50 Epochs.

## Image -1:

![res](https://user-images.githubusercontent.com/81563306/174302440-728890c9-647a-46e2-b2ce-ceb22d29f670.png)

## image-2:

![image](https://user-images.githubusercontent.com/81563306/174473361-3e186c6c-60ec-44f3-b83f-f0142db963ce.png)

First row  => test gray images

Second row => generated color images

Third row  => Real images

### Strategy used :

In this approach we use 2 losses .One is L1 loss which makes it a regression task, and GAN loss, which helps to solve the problem in an unsupervised manner by assigning the outputs a number indicating how real they look.
we will use pretrained ResNet18 as the backbone of  U-Net and to accomplish the second stage of pretraining, we are going to train the U-Net on our training set with only L1 Loss. Then we will move to the combined adversarial and L1 loss.
Then we will train the model for 10 to 20 epochs then compare these pretrainded weights with further 50 epochs that we are going to train.This will give promising results by iteratng over 50 Epochs!


## GAN (Generative Adversarial Network)

One neural network, called the generator, generates new data instances, while the other, the discriminator, evaluates them for authenticity; i.e. the discriminator decides whether each instance of data that it reviews belongs to the actual training dataset or not.

The goal of the generator is to generate passable hand-written digits: to lie without being caught. The goal of the discriminator is to identify images coming from the generator as fake.

![image](https://user-images.githubusercontent.com/81563306/174475255-86b6f5c4-cd7f-4be8-b8ed-6f9953d21b83.png)

### Loss function

Lets take x as the grayscale image, z as the input noise for the generator, and y as the 2-channel output we want from the generator . Also, G is the generator model and D is the discriminator. Then the loss for our conditional GAN will be:
![image](https://user-images.githubusercontent.com/81563306/174475467-51c0dc60-8612-4540-82e6-25766e275f2f.png)

And the L1 loss function we are using is :
![image](https://user-images.githubusercontent.com/81563306/174475456-609f0ef1-f26d-43ec-87bb-6ac3f0f1ec8b.png)

If we use L1 loss fn alone, the model still learns to colorize the images but it will be conservative and most of the time uses colors like "gray" or "brown" because when it doubts which color is the best, it takes the average and uses these colors to reduce the L1 loss as much as possible. Also, the L1 Loss is preferred over L2 loss  because it reduces that effect of producing gray-ish images. So, our combined loss function will be:
![image](https://user-images.githubusercontent.com/81563306/174475472-5b11ad03-7447-4571-aef9-acdded1b85a5.png)




## Code Implementation









