#import libraries
import csv
import pandas as pd
import numpy as np
from numpy.linalg import norm
import time

#preprocessing
ratings = np.genfromtxt('ratings.csv', delimiter=',')[1:]
ratings[:,2] = ratings[:,2]*2 #met le rating sur 10
ratings = ratings.astype(int)

movies = pd.read_csv('movies.csv', header=None).values[1:]
movies[:,0] = movies[:,0].astype(int)
for idx, genres in enumerate(movies[:,2]):
    movies[idx][2] = genres.split('|')
    
#creates dictionnary that links idx to movies ids
movie_idx_from_id = {}
for i in range(movies.shape[0]):
    movie_idx_from_id[movies[i,0]] = i
    
#retrieves all existing genres    
genres = set() 
for gs in movies[:,2]:
    for g in gs:
        genres.add(g)

#creates dictionnary that links ids with genres string
id_from_genre = {}
for idx, genre in enumerate(genres):
    id_from_genre[genre] = idx
    
links = np.genfromtxt('links.csv', delimiter=',')[1:].astype(int)

tags = pd.read_csv('tags.csv', header=None).values[1:]
tags[:,0:2] = tags[:,0:2].astype(int)
tags[:,3] = tags[:,3].astype(int)

#creates a dictionnary that links a user id and a movie to a rating
rating_dict = {}
for r in ratings:
    rating_dict[(r[0],r[1])] = r[2]

#content-based method
def get_sets(ratings, ratio):
    temp = ratings[ratings[:,0] == 1]
    step = int(np.round(temp.shape[0]*ratio))

    train_ratings = temp[:step]
    test_ratings = temp[step:]

    users = set(ratings[:,0])
    users.remove(1)

    for i in set(users): #for all users expect the first
        temp = ratings[ratings[:,0] == i]
        step = int(np.round(temp.shape[0]*ratio))

        train_ratings = np.concatenate((train_ratings, temp[:step]))
        test_ratings = np.concatenate((test_ratings, temp[step:]))
        
    print(f"Training set shape: {train_ratings.shape[0]}\nTest set shape: {test_ratings.shape[0]}\n")
    return train_ratings, test_ratings

def get_profils(ratings):
    #computing TF matrix
    TF = np.zeros((len(genres), movies.shape[0]))
    for idx, gs in enumerate(movies[:,2]):
        for g in gs:
            TF[id_from_genre[g],idx] = 1
            
    #computing the inverse frequency
    occurences = np.zeros(TF.shape[0])
    for g in genres:
        occurences[id_from_genre[g]] = int(np.sum(TF[:][id_from_genre[g]]))
    IDF = np.log(TF.shape[1]/occurences)
    
    #computing the TD-IDF score for every pair of feature-item
    TF_IDF = IDF*TF.T
    
    nb_ids = len(set(ratings[:,0])) #610 ids with at least 1 rating from each
    like_threshold = 8
    liked_films = [[] for _ in range(nb_ids)] #real id is +1 because array starts with 0
    given_scores = [[] for _ in range(nb_ids)] 
    for rating in ratings:
        if(rating[2] >= like_threshold):
            liked_films[rating[0]-1].append(movie_idx_from_id[rating[1]])
            given_scores[rating[0]-1].append(rating[2])
            
    #computing the user profils
    user_profils = np.zeros((nb_ids,len(genres)))
    for i in range(nb_ids):
        profil = np.zeros(len(genres))
        for j in range(len(liked_films[i])): #using the weighted average aggregation method
            profil += given_scores[i][j]*TF_IDF[liked_films[i][j]] #sum the profils of liked movies with the weight given by the user rating score
        profil /= max(np.sum(given_scores[i]),1) #avoid division by 0 when user has no liked movie
        user_profils[i] =profil
        
    return user_profils, TF_IDF, liked_films

def recommendation_CB(ratings, user_id, nb_reco):
    
    #content based predictions
    
    user_profils, TF_IDF, liked_films = get_profils(ratings)
        
    #cosine similarity avoiding usage of for loop
    top = user_profils@TF_IDF.T #scalar products
    bottom = (np.linalg.norm(user_profils, axis = 1).reshape((user_profils.shape[0],1))@np.linalg.norm(TF_IDF, axis = 1).reshape((1,TF_IDF.shape[0]))) #mul of norms
    bottom[bottom == 0] = 1 #avoids div by 0
    scores = top/bottom
    
    
    scores_not_watched = scores[user_id].copy()
    scores_not_watched[liked_films[user_id]] = 0 #removes already noted movies
    return np.argpartition(scores_not_watched, nb_reco*-1)[nb_reco*-1:] #gets the k unwatched movies with the highest predicted score   

#collaborative filtering
#find similar users
def jaccard_similarity(doc1, doc2):
  if len(doc1)==0 or len(doc2)==0:
    return 0.0
  else:
    inter = doc1.intersection(doc2)
    union = doc1.union(doc2)
    return float(len(inter))/float(len(union))

def cosine_similarity(user_id_1,user_id_2):
    return np.dot(user_id_1,user_id_2)/(norm(user_id_1)*norm(user_id_2))

#prediction ratings
def average_prediction(user,movie,similar_users): #user: user_id    movie:movie_id    similar_users: list of users_id    
    prediction = 0
    count = 0  
    for y in similar_users:  
        r = rating_dict.get((y,movie),-1)
        if(r != -1):
            prediction += r
            count += 1
    
    if(count < 3):  #We want at least 3 ratings to make a prediction 
        return -1 
    return prediction/count 

def get_rating_array(user_id):
    res = []
    for m in movies[:,0]:
        res.append(rating_dict.get((user_id,m),0))
    return res

#get the k most similar users to user_id based on cosine similarity
def get_similar_users(ratings, user_id,k):
    user_id_ratings = get_rating_array(user_id)
    cos = []
    for uid in range(1,len(set(ratings[:,0]))): #number of users
        uid_ratings = get_rating_array(uid)
        cos.append(cosine_similarity(user_id_ratings,uid_ratings))
    
    return [cos.index(i) for i in sorted(cos,reverse=True)][1:k+1] #we don't use the first result because it's the user_id itself     

def get_similar_users_jaccard(ratings, user_id,k):
    user_id_ratings = set(ratings[np.where((ratings[:,0] == user_id)),1].tolist()[0])
    jac = []
    for uid in range(1,len(set(ratings[:,0]))): #number of users
        uid_ratings = set(ratings[np.where((ratings[:,0] == uid)),1].tolist()[0])
        jac.append(jaccard_similarity(user_id_ratings,uid_ratings))
    
    return [jac.index(i) for i in sorted(jac,reverse=True)][1:k+1]

def get_best_movies_predictions(user_id,similar_users,k): #k: nb of recommendations        
    rat = [average_prediction(user_id,m,similar_users) for m in movies[:,0]]   
    return (-np.array(rat)).argsort()[:k]

#user-user collaborative filtering
def recommendation_CF(ratings, user_id,k,nb_similar,similarity_method): #k: nb of recommendations
    if(similarity_method == "jaccard"):    
        similar = get_similar_users_jaccard(ratings, user_id,nb_similar)
    else:
        similar = get_similar_users(ratings, user_id,nb_similar)
    prediction = get_best_movies_predictions(user_id,similar,k)
    return prediction

def recommendation(ratings, user_id,k,nb_similar,similarity_method):
    s1 = time.time()
    pred_CB = recommendation_CB(ratings, user_id-1, k)
    tCB = time.time() - s1
    s2 = time.time()
    pred_CF = recommendation_CF(ratings, user_id,k,nb_similar,similarity_method)
    tCF = time.time() - s2
    pred = []
    idx = 0
    while(len(pred) != k):
        if(idx%2 == 0):
            if(pred_CB[int(np.floor(idx/2))] not in pred):
                pred.append(pred_CB[int(np.floor(idx/2))])
                idx+=1
        else:
            if(pred_CF[int(np.floor(idx/2))] not in pred):
                pred.append(pred_CF[int(np.floor(idx/2))])
                idx+=1
    for l in pred:
        print(f"id:{l}{(4-int(np.log10(l)))*' '} movieId:{movies[l,0]}{(6-int(np.log10(movies[l,0])))*' '} Title:{movies[l,1]}")
    print(f"Elapsed time: {round(time.time() - s1,3)},    Content based: {round(tCB,3)},   Collaborative filtering: {round(tCF,3)}")

def evaluate(ratings, ratio):
    train_ratings, test_ratings = get_sets(ratings, ratio)
    user_profils, TF_IDF, lf = get_profils(train_ratings)
    
    l = np.max(train_ratings[:,0])
    tot = 0
    bot = 0
    for i in range(l):
        test_ids_of_user = [movie_idx_from_id[j] for j in test_ratings[test_ratings[:,0] == i+1][:,1]] #gets the ids of the test film of the user
        
        top = user_profils[i]@TF_IDF[test_ids_of_user,:].T
        bottom = np.linalg.norm(user_profils[i])*np.linalg.norm(TF_IDF[test_ids_of_user,:], axis = 1) #mul of norms
        bottom[bottom == 0] = 1 #avoids div by 0
        scores = top/bottom #computes the scores for each film
        rank = np.argsort(scores) #gets the ranks of films (1st arg is the worst score and last is the bets)
        
        similar = get_similar_users(train_ratings, i+1,100) #gets similar users
        pred_CF = get_best_movies_predictions(i+1,similar,-1) #movies ranked
        pred_CF = [p for p in pred_CF if p in test_ids_of_user] #keeps movies that are in the test set 
        pred_CF = np.array([test_ids_of_user.index(p) for p in pred_CF]) #gets the ids in test_ids_of_user while keeping the rank
        
        if(rank.size == pred_CF.size): #bug solution
            v = []
            for j in range(len(test_ids_of_user)):
                v.append(np.where(rank == j)[0][0] + np.where(pred_CF == j)[0][0]) #gets the sum of ranks
            pred = np.argsort(v) #gets the final rank of every movie

            ra = np.array([test_ratings[test_ratings[:,0] == i+1][:,2][j] for j in pred]) #gets the ratings of the films with the permutation given in rank
            nb_disliked = np.sum(ra < 8)
            ra = ra[nb_disliked:]#we dismiss the nb_disliked films which have the worst score for our evaluation
            if(ra.size != 0): #we only consider cases with more than 1 recommendation
                tot += np.sum(ra >= 8)/np.max([ra.size,1]) #computes the ratio of films recommended that have a rating higher than 8
                bot += 1
    print(f"Average rate of prediction accuracy: {tot/bot}")


while(True):
    mode = input("Choose your mode (reco/eval): ")
    if(mode == 'reco'):
        user_id = int(input("Userid (1-610): "))
        k = int(input("\nNumber of recommendations: "))
        recommendation(ratings, user_id,k,100,'cosine')
        print('\n')
    elif(mode == 'eval'):
        ratio = input('\ntrain/test ratio (between 0 and 1): ')
        print('\This evaluation will take 15 minutes\n')
        evaluate(ratings, 0.8)
        print('\n')
    else:
        print(f"\nMode {mode} does not exists\n")