import numpy as np
import os
from scipy.sparse import csr_matrix, kron
from scipy.sparse.linalg import cg
from scipy.stats import pearsonr
from scipy.optimize import minimize

def logpstr(prms, X, y, lmbda, L, sign=-1.0):
    w = prms[:-1]
    sigma = prms[-1]
    term = (np.transpose(X)).dot(L)
    reg = term.dot(X)
    term1 = lmbda*((np.transpose(w)).dot(reg)).dot(w)
    obj = sign * ( -(len(y)/2)*np.log(sigma**2) - (1/(2*sigma**2)) * (np.transpose((X.dot(w) - y))).dot((X.dot(w) - y)) - 0.5 * term1)
    return obj

def gmrf(X, y, lmbda, L):
    initial_guess = np.ones(X.shape[1]+1)
    result = minimize(logpstr, initial_guess, (X, y, lmbda, L), method='L-BFGS-B', jac=grad)
    return result.x

def grad(prms, X, y, lmbda, L, sign=-1.0):
    w = prms[:-1]
    sigma = prms[-1]
    term = (np.transpose(X)).dot(L)
    reg = term.dot(X)
    term1 = lmbda*((np.transpose(w)).dot(reg)).dot(w)
    g = np.ones(X.shape[1]+1)
    g[:-1] = (1/(2*sigma**2)) * (2*((np.transpose(X)).dot(X)).dot(w) - 2*(np.transpose(X)).dot(y)) + 0.5 * term1
    g[-1]= len(y)/(2*sigma**2) - (1/(2*sigma**4)) * (np.transpose((X.dot(w) - y))).dot((X.dot(w) - y))
    return g

def gmrf_predict(model, Xhat, lmbda):
    w = model[:-1]
    sigma = model[-1]
    pyhat = np.zeros([Xhat.shape[0],7])
    for i in range(Xhat.shape[0]):
        for j in range(7):
            a = (1/np.sqrt(2*np.pi*sigma**2))
            b = np.exp(-(np.transpose(w).dot(Xhat[i,:]) - j+1)**2)
            pyhat[i,j] = a * b

    return pyhat


""" ECM """
""" Cross validation to choose lmbda """
splitval = int(len(users_obs)/2)
yobs_train = yobs[1:splitval]
yobs_valid = yobs[splitval+1:]

bow_obs_train = bow[1:splitval,:]
bow_obs_valid = bow[splitval+1:,:]
tfidf_obs_train = tfidf_obs[1:splitval,:]
tfidf_obs_valid = tfidf_obs[splitval+1:,:]
user_to_w2v_obs_train = user_to_w2v_obs[1:splitval,:]
user_to_w2v_obs_valid = user_to_w2v_obs[splitval+1:,:]
user_to_liwc_obs_train = user_to_liwc_obs[1:splitval,:]
user_to_liwc_obs_valid = user_to_liwc_obs[splitval+1:,:]


minErr = np.inf
best_lmbda = 0
best_model_1 = np.array([np.ones(bow_obs_train.shape[1]), np.ones(1)], dtype=object)
best_model_2 = np.array([np.ones(tfidf_obs_train.shape[1]), np.ones(1)], dtype=object)
best_model_3 = np.array([np.ones(user_to_w2v_obs_train.shape[1]), np.ones(1)], dtype=object)
best_model_4 = np.array([np.ones(user_to_liwc_obs_train.shape[1]), np.ones(1)], dtype=object)

for lmbda in [2.0**i for i in np.arange(-12,12)]:
    model_1 = gmrf(bow_obs_train, yobs_train, lmbda, L[1:splitval,1:splitval])
    model_2 = gmrf(tfidf_train, yobs_train, lmbda, L[1:splitval,1:splitval])
    model_3 = gmrf(user_to_w2v_obs_train, yobs_train, lmbda, L[1:splitval,1:splitval])
    model_4 = gmrf(user_to_liwc_obs_train, yobs_train, lmbda, L[1:splitval,1:splitval])
    m = np.array([gmrf_predict(model_1, bow_obs_valid,lmbda), gmrf_predict(model_2, tfidf_obs_valid,lmbda), gmrf_predict(model_3, user_to_w2v_obs_valid,lmbda), gmrf_predict(model_4, user_to_liwc_obs_valid,lmbda)], dtype=object)
    pyhat_mean = np.mean(m, 0)
    yhat = np.argmax(pyhat_mean,1) + 1
    validError = np.sum((yhat - yobs_valid)**2)/(len(users_obs)/2)
    if validError < minErr:
        best_lmbda = lmbda
        minErr = validError
        best_model_1 = model_1
        best_model_2 = model_2
        best_model_3 = model_3
        best_model_4 = model_4

w_1 = best_model_1[:-1]
sigma_1 = best_model_1[-1]
w_2 = best_model_2[:-1]
sigma_2 = best_model_2[-1]
w_3 = best_model_3[:-1]
sigma_3 = best_model_3[-1]
w_4 = best_model_4[:-1]
sigma_4 = best_model_4[-1]

print("best_lmbda=", best_lmbda)
for t in range(niter):
    print(t)

    """ E - Step """
    e_y_bow_tilde = bow_tilde.dot(w_3)
    e_y_sqr_bow_tilde = sigma_3**2 + e_y_bow_tilde**2
    e_y_tfidf_tilde = tfidf_tilde.dot(w_4)
    e_y_sqr_tfidf_tilde = sigma_4**2 + e_y_tfidf_tilde**2
    e_y_user_to_w2v_tilde = user_to_w2v_tilde.dot(w_3)
    e_y_sqr_user_to_w2v_tilde = sigma_3**2 + e_y_user_to_w2v_tilde**2
    e_y_user_to_liwc_tilde = user_to_liwc_tilde.dot(w_4)
    e_y_sqr_user_to_liwc_tilde = sigma_4**2 + e_y_user_to_liwc_tilde**2
    users = np.concatenate([users_obs, users_tilde])
    """ CM - Step """
    X = np.concatenate([bow_obs, bow_tilde])
    y = np.concatenate([yobs, e_y_bow_tilde])
    y2 = np.concatenate([np.power(yobs,2), e_y_sqr_bow_tilde])
    w_1 = cg( (1/(sigma_3**2)) * np.transpose(X).dot(X) + best_lmbda * ((np.transpose(X)).dot(L)).dot(X), (1/(sigma_3**2)) * (np.transpose(X)).dot(y) )
    w_1 = w_1[0]
    s = 0
    for i in range(len(users)):
        s = s + y2[i] - 2*y[i]*np.transpose(w_3).dot(X[i]) + (np.transpose(w_3).dot(X[i]))**2

    sigma_1 = np.sqrt((1/len(users)) * s)

    X = np.concatenate([tfidf_obs, tfidf_tilde])
    y = np.concatenate([yobs, e_y_tfidf_tilde])
    y2 = np.concatenate([np.power(yobs,2), e_y_sqr_tfidf_tilde])
    w_3 = cg( (1/(sigma_3**2)) * np.transpose(X).dot(X) + best_lmbda * ((np.transpose(X)).dot(L)).dot(X), (1/(sigma_3**2)) * (np.transpose(X)).dot(y) )
    w_3 = w_3[0]
    s = 0
    for i in range(len(users)):
        s = s + y2[i] - 2*y[i]*np.transpose(w_3).dot(X[i]) + (np.transpose(w_3).dot(X[i]))**2

    sigma_3 = np.sqrt((1/len(users)) * s)


    X = np.concatenate([user_to_w2v_obs, user_to_w2v_tilde])
    y = np.concatenate([yobs, e_y_user_to_w2v_tilde])
    y2 = np.concatenate([np.power(yobs,2), e_y_sqr_user_to_w2v_tilde])
    w_3 = cg( (1/(sigma_3**2)) * np.transpose(X).dot(X) + best_lmbda * ((np.transpose(X)).dot(L)).dot(X), (1/(sigma_3**2)) * (np.transpose(X)).dot(y) )
    w_3 = w_3[0]
    s = 0
    for i in range(len(users)):
        s = s + y2[i] - 2*y[i]*np.transpose(w_3).dot(X[i]) + (np.transpose(w_3).dot(X[i]))**2

    sigma_3 = np.sqrt((1/len(users)) * s)

    X = np.concatenate([user_to_liwc_obs, user_to_liwc_tilde])
    y = np.concatenate([yobs, e_y_user_to_liwc_tilde])
    y2 = np.concatenate([np.power(yobs,2), e_y_sqr_user_to_liwc_tilde])
    w_4 = cg( (1/(sigma_4**2)) * np.transpose(X).dot(X) + best_lmbda * ((np.transpose(X)).dot(L)).dot(X), (1/(sigma_4**2)) * (np.transpose(X)).dot(y) )
    w_4 = w_4[0]
    s = 0
    for i in range(len(users)):
        s = s + y2[i] - 2*y[i]*np.transpose(w_4).dot(X[i]) + (np.transpose(w_4).dot(X[i]))**2

    sigma_4 = np.sqrt((1/len(users)) * s)


m = np.array([bow_obs.dot(w_3), tfidf_obs.dot(w_3), user_to_w2v_obs.dot(w_3),user_to_liwc_obs.dot(w_4)])
yhat_mean = np.mean(m, 0)

print(pearsonr(yobs, yhat_mean))
