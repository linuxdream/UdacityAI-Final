# Udacity AI Programming in Python Nanodegree Final Project

## Introduction

Well hi there! If you are here, I'm guessing you are either very curious about my ability to program in Python and specifically AI programming...or you are a student in the [AI Programming with Python Udacity nanodegree](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089) program looking for the answer to the final project.

If you fall into the latter camp, great! Take a look at my code and use it to _partially_ guide your own code. Maybe I did something differently than you did or maybe you just need a little push in the right direction for one of the functions. All I ask is that you don't just copy/pasta and turn this in as your own. It's far better for you to invest the time in writing things out on your own and use my (and other's) code more as a resource and alternative opinion than as something you should turn in.

If you fall into the former camp, then welcome; most of this document is for you! It is my personality to be brutally honest in all things regarding my personal skill set so you will see no lies from me in here. Everything here was coded by me with references being official documentation, Udacity resources/starter code, and blog posts/generic tutorials.

If for whatever reason GitHub isn't rendering the notebook file (which seems to happen a lot), it can be [viewed here](https://nbviewer.jupyter.org/github/linuxdream/UdacityAI-Final/blob/master/Image%20Classifier%20Project.ipynb).

### Why did I do this program?

I have been a software/web engineer for almost 15 years now (as of 2019) and at the time of this writing, I am two classes away from completing my Master of Computer Science degree from UIUC where I initially focused in their Data Science track. Unfortunately, with two courses to go, I had to change my emphasis away from Data Science to just the basic MCS despite the fact that I followed nearly all Data Science requirements but _one_ class. That one class was CS 498 - Applied Machine Learning. I'm not going to lie, the course was just too much for me. I tried it twice and withdrew about half way through each time. I could sit here and complain about how unfair and unguided the class was and how much time it took me as a working father doing this degree in my spare time (~25hours/week for the 4-unit course) but I'll spare the details. I just couldn't complete the course's requirements in the time I had. It's as simple as that.

So I decided that I should find some other way to demonstrate that I was still capable of learning, understanding, and performing the functions of someone who, in a different time, might otherwise complete the class. Thus, I decided that I would spend a semester and take the Udacity AI Programming with Python course.

### Was it worth it? Did it fill the void of CS-498?

In short, yes it was worth it. Did it fill the void? Not entirely. I think I learned more in this program than I did trying and struggling in CS-498. The final homework for CS-498 Applied Machine Learning (at least when I took it) was to build a convolutional neural network with TensorFlow. Guess what...that's largely this course's final project as well. Did I miss stuff? Absolutely. We didn't cover means classifiers, we didn't need to code an SVM from scratch, didn't cover linear/logistic regression, no EM algorithms from scratch (ugh), etc. Despite this, I do know what these are and have coded means classifiers, PCA, SVM, and others on my own.

The biggest difference is that this course is more practical in nature while CS-498 was, well, highly technical and you had to do everything from scratch with _very unclear_ instructions. Yes, different purposes and different audiences and that's perfectly fine. But here I am at the end of it actually knowing _roughly_ how a deep neural network works but I am _fully able_ to implement it quickly and efficiently (for the most part). The introductions to Python, Pandas, Numpy, and Matplotlib were essential learning for me that never occurred in my MCS program...rather it was taken for granted that I already knew it. I know for a fact that a lot of MCS students struggled with that. I also got a good refresher on linear algebra and calculus which the MCS program assumes you know (as you should)...even though they for sure know you probably took them a decade+ ago. The program is marketed at working professional after all and let's be honest, if you're not already in a data science field, you very likely don't use things like the chain rule and matrix multiplication very often without just using a pre-coded function/library.

### So where are my skills at right now? (end of 2019)

I can tell you that after this nanodegree, my Python and data science library skills are far better than where they were before I did the program. Again, I've been in the web software engineering field for almost 15 years so I had the programming background but I lacked the Python and data science library use knowledge. I feel much more comfortable with that now. I'm most certainly not an expert and I'd consider myself an entry-level data scientist at this point.

Between this course and my MCS classes I feel like I have good grasp on a lot of the machine learning concepts and could implement most of the more common ones. Where I think I lack is in the production implementation knowledge. I can code it but I don't know what it looks like to implement this as a service backend. I think I have an idea just from my cloud architecture and API design background but I have yet to see an actual implementation.

### So what's next?

Well, I may do more nanodegree programs but I also might just follow some online tutorials and books that outline some of the procedures that I haven't had much opportunity to play with. I'd like to build a more robust k-means classifier than what I have previously done and I need to better understand how recurrent neural networks work. Finally, I'd love to build out an API-driven cloud-based classifier or recommendation engine so I can get a solid grasp on how these system work for everyday apps and services.

## Where to start with this repo.

The `Image Classifier Project.html` (or the `.ipynb`) file is the primary file that contain the bulk of the project code. Most of the markdown in that file are project guidelines with only some sample code provided. The rest was a culmination of pieces of previous smaller projects in the course that I wrote that were put together and modified into the final notebook.

The second part of the final project was to write the classifier as a command line script so that it could be trained and perform classifications in the terminal. The `predict.py` and `train.py` files are the loader files that define argparser and kickoff the main scripts. The bulk of the code are in the `predict_functions.py` and `train_functions.py` files.

### Things I could have done differently if time weren't a concern.

- I could (should) have written these as classes but, for the purposes of the project requirements, it wasn't necessary and thus I skipped that. I think the readability and ease of use of a class would have been a benefit here.
- Parameter passing here just pain terrible. I did it this way for ease of coding and my own understanding vs best practice. In the JS world, I would have passed an object but I understand that using `*args` or `**kwargs` is the Pythonic way.
- I was able to achieve about 80% accuracy with DenseNet121 and the default params I coded but I'm pretty sure given more time to experiment I could have gotten that higher. In the project, we were given limited time to use the GPU processor so I didn't want to just burn cycles for the sake of a high score.
