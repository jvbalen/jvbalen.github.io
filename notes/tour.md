# recommender systems, an opinionated tour

## recsys flavors pt. 1
roughly in hierarchical order

- content-based or behavioral
    - content-based = out of scope
- online vs. offline
    - online (bandits, RL stuff) may be better but it is still very hard
    - perhaps useful for picking the "best of O(10)" if you have really good engineers
- implicit vs. explicit
    - these days, mostly implicit
- sequential or not
    - IMO, sequential information isn't often important, but think about the domain
    - if you really need it, think of it as a small language model -> GRU's appear to work well


## intermezzo: the data

in what follows, assume
- offline recommendation, "behavioral", non-sequential, implicit feedback data
we have some data of the following form:
- user-item sparse matrix of "interaction counts"
- the rows of this matrix encode a user's "negatives" (zeros) and "positives"
- important to remember: the negatives are *implicit* a negative / zero doesn't mean the user does not like the item it just means that didn't engage with it (yet) (see also one-class or "PU" learning, for ML problems with only Positive and Unlabeled data)
the goal:
- given some (past) positives, predict other (future) positives


## recsys flavors pt. 2

- shallow or deep 
    - you probably don't need deep learning
        - not much evidence in favor
            - most top DL recsys papers' results up to 2019 could not be reproduced
            - deep sequential models were recently shown to regularly underperform
        - theoretical argument (later)
        - brittle
    - exception: rich side-information
        - however, consider a two-stage approach featuring a re-ranking ensemble
- factors/embeddings or full-rank
    - beyond 10K items/actions, factorize
- explicit ("") vs. implicit user representation
    - theoretical argument: if you can learn a good user representation, you definitely don't need deep
    - in practice, if n_users >> n_items, use implicit representations
    - implicit: one vs. multiple vs. |Xu| user representations
        - in theory, one is enough
        - in practice, one is enough unless you have good reasons to believe otherwise, in which case, try both
- optimization: SGD, (alternating) closed-form, ADMM, CG...
- loss function:
    - classification loss: predict the next item (MSE, BCE, CCE)
    - ranking loss: rank positives above negatives (point-wise rank loss, pairwise rank loss, creative variants e.g. WARP)
    - practical observation is that "bad" loss functions often still work well and v.v. e.g. MSE loss -> good NDCG; BPR -> not necessarily
- exact vs. approximate loss
    - mostly: negative sampling
    - distribution of negatives matters
- counterfactual or non-counterfactual
    - idem online: still difficult


## some useful examples

implicit, non-sequential, embedding-based, SGD
- item similarity models
- learned item similarity models: SLIM & EASE
- item-item matrix factorization with SGD
    - ASA 'asymmetric' SVD, AutoRec, FISM, 
    - word2vec is a widely-used semi-sequential variant
    - original word2vec implementation in C (works great if you're not going to customize) https://github.com/tmikolov/word2vec
- autoencoders
    - it's the same thing
    - this the point I usually try to drive home the most
    - many DL recommenders are not AE, but overkill for other reasons
- user-item matrix factorization with MSE
    - Cython implementation (works great too): https://github.com/benfred/implicit
    - we're kind of full-circle, because of our "free variables"
    - however efficiency, rank, choice of loss, optimization, etc. technique still matter: see 'trade-offs'


## reality check
potential headaches when shipping a personalized product

- the model is only a tiny part
    - good data pipelines are hard
    - business rules are everywhere
- what does an api look like
    - user ids or no user ids
- training and inference are split across different part of a complex application (or training isn't part of the app at all)
- prediction efficiency
    - beyond 10^5-6 items, use approximate nearest neighbors
- versioning: ways to avoid mix up components (embeddings, item sets, id spaces) of an old and newer model
- big recommender systems will involve two or three stages
- most applications, you will not be asked to do a second version of the model
    - the implications of this will depend on context, not sure if there's a lesson there


## evaluation strategies & tips

- online/offline
    - AB tests are best
    - offline evaluation are very noisy yet probably useful!
- metrics
    - make sure you measure something close to what you care about most —and ideally more than one proxy for it
    - don't MSE (except when predicting explicit feedback)
    - use NDCG if this is your final stage
    - use recall @ n_candidates in a candidate generation context
- if the model deals with sensitive data / decisions, add mechanisms to flag bad judgement early on (not much experience here, but I hear this is often missing)
- don't forget qualitative evaluation: make sure you can look at predictions for real data 
- in offline evaluation of implicit feedback models, you will probably need to choose between using
    - a random splits of the positives
    - time-based splits (from past positives, predict future positives) making the right decision here can makes a big difference and time-based splits are *probably* better, but not much hard proof out there (since comparing offline evaluation typically strategies requires expensive AB testing)
- apples to apples [pet peeve]
    - compare only models with similar access to capacity for learning item popularities
        - a cosine-based item similarity model in itself has no notion of popularity, be wary of results comparing it to user-item factorization model that does
        - another way around this is to always evaluate a simple ensemble that includes your model and an item popularity term
    - more generally, when comparing models, hold the feature set constant and v.v.
    - even more generally, evaluate end-to-end if you can
- cross-validation is probably overkill (though I may be wrong on this)
- track the worst fail cases (e..g, in a spreadsheet) and go back to them once in a while


## example roadmap
building a recommender system: what to prioritize when

ask yourself:
- is any personalization at all better than no personalization (great! ship your first model)
- if there are risks (there probably are!), you will probably want to set up your evaluation strategy and evaluate several models before shipping any of them

### your first model
- if you can choose, focus on a problem with a small item space (but real data!)
- prefer a linear model:
    - n_items > 50,000: user-item or item-item matrix factorization
    - else: EASE

### v1.1 or v.2.0

what to focus on?
- a (better) evaluation pipeline, see evaluation strategies
- more/better data
    - e.g.: find more of it (consider a more simpler and more efficient model, if needed)
    - e.g.: be picky about which users, items are part of the training set; the benefits of including side-information are comparatively small, in my experience
    - NOTE: changes to your dataset may change your baselines, so some changes can be hard to evaluate
- model loss functions
- model optimization (e.g. try ALS instead of SGD or v.v.)
- when using negative sampling: the negative sampling distribution (and the selection criteria for positives!)
- evaluate
- what if you want to recommend new stuff only
- avoiding stale recommendations: when you come back, there should be something new
- building an api that allows you to swap out models easily
- sanity checks

what *not* to focus on:
- any big experiments before your have a minimum of trust in your evaluation metrics
- building complex (e.g. DL-based) models if you didn't try something simpler first
- doing hyper-parameter search over complex models if you didn't do it for simpler models first
- implement *all* the evaluation metrics—many are correlated, especially as you start out, and you'll need to focus on a few anyway

some common trade-offs
i.e., things to look into if you have already have a good evaluation pipeline and some experimentation (time, money, devs) budget 
- loss vs. optimization techniques closed-form solutions (incl. ALS) are superior to SGD but rank losses are superior to MSE yet AFAIK we don't have great closed form solutions for rank losses, though [1] might come close
- rank vs. efficiency: performance goes up as rank goes up, but full-rank models like EASE and SLIM are inefficient for big catalogs
perhaps the most important one is a bit meta:
- trade off model complexity for bandwidth for experimentation / hyper-parameter search, the best simple model may be better than a suboptimal complex one


[1] Takacs and Tikk, Alternating Least Squares for Personalized Ranking (2012)
