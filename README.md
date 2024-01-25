# Repo with gzip and BERT comparison on binary classification (IMDB movie review)

### Some instructions

- To test the work on your review, place it in a \<root>/request_data/single_request.txt and run the *run_experiments.py* with flag -f. Or You can write review in console if run the program without -f or -d flags. 



### Results

- pre-training time for base BERT is about 14 minutes
- Average gzip-based model inference time is 5 seconds
- Average gzip-based model f1 score is 0.703 for 1 neigbour (knn), 100 test reviews and 49900 train reviews from IMDB dataset


### Ideas for the future

- [ ] Plot dependence of inference time on dataset size for gzip
- [ ] Plot dependence of f1 value on count of neighbours for knn
- [ ] Find minimum dataset size to achieve 1sec gzip inference
- [ ] Parallelize calculations for gzip
- [ ] Console output of computer characteristics
 