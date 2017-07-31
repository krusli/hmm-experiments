# Experiments on "A Hidden Markov Model for Collaborative Filtering"

Experiments using the MyAnimeList dataset for recommending anime titles (work-in-progress).

Since the dataset lacks timestamps for the rating, the sequential order of anime_ids are used as a "proxy" for the time users added the anime to their ratings list.

The code in `HMM.py` was mostly developed during my internship at Blibli.com (July 2017), but significantly modified afterwards to take advantage of numpy and Python's multiprocessing library.
