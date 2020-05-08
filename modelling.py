#!/usr/bin/env python

import time,os,re,csv,sys,uuid,joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error

PROJECT_ROOT_DIR = "."
MODEL_DIR = os.path.join("models")
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series"

## load from project package
from data_ingestion import DEV
from data_engineering import engineer_features
from data_visualization import save_fig
from logger import update_train_log, update_predict_log

def plot_learning_curve(estimator, X, y, ax=None, cv=5):
    """
    an sklearn estimator 
    """

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    train_sizes=np.linspace(.1, 1.0, 6)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                            cv=cv, n_jobs=-1,
                                                            train_sizes=train_sizes,scoring="neg_mean_squared_error")
    train_scores_mean = np.mean(np.sqrt(-train_scores), axis=1)
    train_scores_std = np.std(np.sqrt(-train_scores), axis=1)
    test_scores_mean = np.mean(np.sqrt(-test_scores), axis=1)
    test_scores_std = np.std(np.sqrt(-test_scores), axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    ## axes and lables
    buff = 0.05
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    xbuff = buff * (xmax - xmin)
    ybuff = buff * (ymax - ymin)
    ax.set_xlim(xmin-xbuff,xmax+xbuff)
    ax.set_ylim(ymin-ybuff,ymax+ybuff)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("MSE Score")
    
    ax.legend(loc="best")


def make_compare_plot(X, y, models):
    """
    create learning curves for SGD, RF, GB and ADA
    """
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    print("...creating learning curves")
    
    ## SGD
    reg1 = SGDRegressor(**models["SGD"][1])
    pipe1 = Pipeline(steps=[("scaler", StandardScaler()),
                            ("reg", reg1)])
    plot_learning_curve(pipe1, X, y, ax=ax1)
    ax1.set_title("Stochastic Gradient Regressor")
    ax1.set_xlabel("")
    
    ## random forest
    reg2 = RandomForestRegressor(**models["RF"][1])
    pipe2 = Pipeline(steps=[("scaler", StandardScaler()),
                            ("reg", reg2)])
    plot_learning_curve(pipe2, X, y, ax=ax2)
    ax2.set_title("Random Forest Regressor")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    
    ## gradient boosting
    reg3 = GradientBoostingRegressor(**models["GB"][1])
    pipe3 = Pipeline(steps=[("scaler", StandardScaler()),
                            ("reg", reg3)])
    plot_learning_curve(pipe3, X, y, ax=ax3)
    ax3.set_title("Gradient Boosting Regressor")
    
    ## ada boosting
    reg4 = AdaBoostRegressor(**models["ADA"][1])
    pipe4 = Pipeline(steps=[("scaler", StandardScaler()),
                            ("reg", reg4)])
    plot_learning_curve(pipe3, X, y, ax=ax4)
    ax4.set_title("Ada Boosting Regressor")
    ax4.set_ylabel("")
    
    ymin, ymax = [], []
    for ax in [ax1, ax2, ax3, ax4]:
        ymin, ymax = ax.get_ylim()
        
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylim([ymin.min(), ymax.max()])
        
        
def plot_feature_importance(estimator, feature_names):
    """
    plot feature importance
    """
    fig = plt.figure(figsize=(8, 6), facecolor="white")
    ax = fig.add_subplot(111)
    
    # make importances relative to max importance
    feature_importance = estimator.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    ax.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, feature_names[sorted_idx])
    ax.set_xlabel('Relative Importance')
    ax.set_title('Variable Importance')
    

def _train_model(X, y, feature_names, tag="total", rs=42, save_img=False, dev=DEV):
    """
    train four models (SGD, RF, GB, ADA) and select the best one
    """
    
    ## start timer for runtime
    time_start = time.time()
    
    ## split the dataset into train and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=rs)
    
    ## build models
    regressor_names = ["SGD","RF", "GB", "ADA"]
    
    regressors = (SGDRegressor(random_state=rs),
                  RandomForestRegressor(random_state=rs),
                  GradientBoostingRegressor(random_state=rs),
                  AdaBoostRegressor(random_state=rs))
    
    params = [
        {"reg__penalty":["l1","l2","elasticnet"],
         "reg__learning_rate":["constant","optimal","invscaling"]},
        {"reg__n_estimators":[10,30,50],
         "reg__max_features":[3,4,5],
         "reg__bootstrap":[True, False]},
        {"reg__n_estimators":[10,30,50],
         "reg__max_features":[3,4,5],
         "reg__learning_rate":[1, 0.1, 0.01, 0.001]},
        {"reg__n_estimators":[10,30,50],
         "reg__learning_rate":[1, 0.1, 0.01, 0.001]}]
    
    
    ## train models
    models = {}
    total = len(regressor_names)
    for iteration, (name,regressor,param) in enumerate(zip(regressor_names, regressors, params)):
        
        end = "" if (iteration+1) < total else "\n"
        print("\r...training model: {}/{}".format(iteration+1,total), end=end)
        
        pipe = Pipeline(steps=[("scaler", StandardScaler()),
                               ("reg", regressor)])
        
        grid = GridSearchCV(pipe, param_grid=param, 
                            scoring="neg_mean_squared_error",
                            cv=5, n_jobs=-1, return_train_score=True)
        
        grid.fit(X_train, y_train)
        models[name] = grid, grid.best_estimator_["reg"].get_params()
        
    ## plot learning curves
    if save_img:
        make_compare_plot(X, y, models=models)
        save_fig("{}_learning_curves".format(tag))
    
    ## evaluation on the validation set
    val_scores = []
    for key, model in models.items():
        y_pred = model[0].predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_pred, y_valid))
        val_scores.append(rmse)
        
    ## select best model
    bm = regressor_names[np.argmin(val_scores)]
    opt_model, params = models[bm]
    
    vocab = {"RF":"Random Forest",
             "SGD":"Stochastic Gradient",
             "GB":"Gradient Boosting",
             "ADA":"Ada Boosting"}

    print("...best model:{}".format(vocab[bm]))

    ## retrain best model on the the full dataset
    opt_model.fit(X, y)

    ## Check the data directory
    model_path=MODEL_DIR
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if dev:
        saved_model = os.path.join(model_path,"test-{}-model-{}.joblib".format(tag,re.sub("\.","_",str(MODEL_VERSION))))
    else:
        saved_model = os.path.join(model_path,"prod-{}-model-{}.joblib".format(tag,re.sub("\.","_",str(MODEL_VERSION))))

    ## save the best model
    joblib.dump(opt_model, saved_model)
    
    if save_img:
        if 'feature_importances_' in dir(opt_model.best_estimator_["reg"]):
            plot_feature_importance(opt_model.best_estimator_["reg"], feature_names=feature_names)
            save_fig("{}_features_importance".format(tag))
    
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)
            
    ## update log
    update_train_log(tag.upper(),vocab[bm],{'rmse':max(val_scores)},runtime,MODEL_VERSION, MODEL_VERSION_NOTE, dev=dev)


def train_models(save_img=False,dev=DEV):
    """
    train models
    """
    
    if dev:
        print("...developing mode")
    else:
        print("...production mode")
    
    ## load engineered features
    datasets = engineer_features(dev=dev, training=True)
    
    ## build, train and save models
    for country in datasets.keys():
        tag = country
        print("training model for {}".format(tag.upper()))
        X, y, dates, feature_names = datasets[tag]
        _train_model(X, y, feature_names, tag=tag, dev=dev, save_img=save_img)

        
def load_models(model_dir=MODEL_DIR, dev=DEV):
    """
    load models
    """
    
    if dev:
        prefix = "test"
    else:
        prefix = "prod"
    
    if not os.path.exists(model_dir):
        raise Exception("Opps! Model dir does not exist")
    
    ## list model files from model directory
    models = [f for f in os.listdir(model_dir) if re.search(prefix,f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))
    
    ## load models
    all_models = {}
    for model in models:
        all_models[re.split("-",model)[1]] = joblib.load(os.path.join(model_dir,model))
        
    return(all_models)

def predict_model(year, month, day, country, dev=DEV):
    """
    make predictions
    """
    
    if dev:
        print("...developing mode")
    else:
        print("...production mode")
    
    ## start timer for runtime
    time_start = time.time()
    
    ## load models
    models = load_models(dev=dev)
    
    ## load data
    datasets = engineer_features(training=False, dev=dev)
    
    ## check if the model is available
    if country not in models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))
    
    ## ckeck if the data is available
    if country not in datasets.keys():
        raise Exception("ERROR (model_predict) - dataset for country '{}' could not be found".format(country))
    
    
    ## ensure the year, month day are numbers
    for d in [year,month,day]:
        if re.search("\D",d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")
    
    ## get the dataset and model for the given country    
    X, y, dates, labels = datasets[country]
    df = pd.DataFrame(X, columns=labels, index=dates)
    model = models[country]
    
    ## check date
    target_date = "{}-{}-{}".format(year,str(month).zfill(2),str(day).zfill(2))
    print(target_date)
    
    if target_date not in df.index.strftime('%Y-%m-%d'):
        raise Exception("ERROR (model_predict) - {} not in range {} and {}".format(target_date,df.index.strftime('%Y-%m-%d')[0],df.index.strftime('%Y-%m-%d')[-1]))
    
    ## query the data
    query = pd.to_datetime(target_date)
    X_pred = df.loc[pd.to_datetime(query),:].values.reshape(1, -1)
    
    ## make prediction
    y_pred = model.predict(X_pred)
    
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)
    
    ## update predict log
    update_predict_log(country.upper(),y_pred,target_date,runtime,MODEL_VERSION, MODEL_VERSION_NOTE,dev=dev)
    
    return({"y_pred":y_pred})
    

if __name__ == "__main__":
    
    run_start = time.time()
    print("...modelling")
  
    ## train models
    print("TRAINING MODELS")
    train_models(dev=DEV)
    
    ## load models
    print("LOADING MODELS")
    models = load_models(dev=DEV)
    
    ## models metadata
    for key, item in models.items():
        print("label:{}, algorithm:{}".format(key, type(item.best_estimator_["reg"]).__name__))
    
    ## test predict
    print("PREDICT")
    if DEV:
        result = predict_model(country="total",year="2018",month="01",day="05", dev=DEV)
    else:
        result = predict_model(country="total",year="2019",month="09",day="05", dev=DEV)
    print(result)
    
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("running time:", "%d:%02d:%02d"%(h, m, s))
    
    print("...done")
