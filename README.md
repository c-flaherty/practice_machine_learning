# Practicing Machine Learning

## Real or Not? NLP with Disaster Tweets
This is a Kaggle competition (https://www.kaggle.com/c/nlp-getting-started/overview).

Results: My model achieved 83% accuracy on the evaluation dataset, placing me at 820/3170 on the public leaderboards. If you ignore the cheaters with perfect scores, this is closer to 500/3170.

Methodology: My model was simple. I used the pretrained "BERT-base-cased" model provided in the Transformers package along with a sequence classification head. I trained it with 4 epochs on 90% of the training data.

## MNIST Competition
This is a Kaggle competition (https://www.kaggle.com/c/digit-recognizer/overview).

Results: My model achieved 97% accuracy.

Methodology: The algorithm was written with PyTorch, and its archictecture is Convolution + ReLU -> Max Pooling -> Convolution + ReLU -> Max Pooling -> Feed-Forward Neural Network -> Softmax.

## Iris Data Challenge 

In this algorithm, I implement a neural network on the classic IRIS dataset. Following the architecture outlined in TTIC 31230: Fundamentals of Deep Learning, Lecture 2, the network consists of: 1 hidden layer, softmax activation functions, Cross-Entropy Loss, Cross-Validation, and Stochastic Gradient Descent. In all the times that I trained and tested it, it exhibited an accuracy >90%.

## Titanic Competition
This is a Kaggle competition (https://www.kaggle.com/c/titanic/overview). 

I used a gradient boosting classifier. I cleaned data (filled in NaN's), one-hot'd categorical variables, and used performed a grid search to optimize the hyperparameters of the gradient boosing classifier.


