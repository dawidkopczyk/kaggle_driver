
def get_param_grid(modelname):

    if modelname=='Keras':        
        # General parameters for Keras
        LEARNING_RATE=[0.0001] #[0.0001, 0.001, 0.01, 0.1, 0.5]
        NEURONS = [256] #[32,64,128,256]
        ACTIVATION = ['softplus'] #['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        INIT_MODE = ['glorot_uniform'] #['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
        EPOCHS = [20] #[6,10,20]
        BATCH_SIZE = [128] #[32,64,128,265]
        OPTIMIZER = ['Adam']
        DROPOUT = [0.2] #[0.0, 0.2, 0.4, 0.6, 0.8]
        VERBOSE = [1]
        param_grid = dict(learning_rate=LEARNING_RATE, neurons=NEURONS, 
                          activation=ACTIVATION, init_mode=INIT_MODE, epochs=EPOCHS, 
                          batch_size=BATCH_SIZE, optimizer=OPTIMIZER, dropout=DROPOUT, verbose=VERBOSE)
 
#    if modelname=='Keras':        
#        # General parameters for Keras
#        LEARNING_RATE=[0.01] #[0.0001, 0.001, 0.01, 0.1, 0.5]
#        NEURONS = [16] #[32,64,128,256]
#        ACTIVATION = ['softplus'] #['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#        INIT_MODE = ['glorot_uniform'] #['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#        EPOCHS = [6] #[6,10,20]
#        BATCH_SIZE = [64] #[32,64,128,264]
#        OPTIMIZER = ['Adam']
#        DROPOUT = [0.2] #[0.0, 0.2, 0.4, 0.6, 0.8]
#        VERBOSE = [1]
#        param_grid = dict(learning_rate=LEARNING_RATE, neurons=NEURONS, 
#                          activation=ACTIVATION, init_mode=INIT_MODE, epochs=EPOCHS, 
#                          batch_size=BATCH_SIZE, optimizer=OPTIMIZER, dropout=DROPOUT, verbose=VERBOSE)
        
    elif modelname=='XGBoost':
        # General parameters for xgboost
        N_ESTIMATORS = [5000]
        LEARNING_RATE = [0.07]
        OBJECTIVE = ["binary:logistic"]
        COLSAMPLE_BYTREE = [0.8]
        MAX_DEPTH = [4]
        SUBSAMPLE = [0.8]
        GAMMA = [10]
        MIN_CHILD_WEIGHT = [0.77]
        SCALE_POS_WEIGHT = [1.6]
        REG_ALPHA = [8]
        REG_LAMBDA = [1.3]
        N_JOBS = [4]
        param_grid = dict(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, objective=OBJECTIVE, colsample_bytree=COLSAMPLE_BYTREE, max_depth=MAX_DEPTH,
                          subsample=SUBSAMPLE, gamma=GAMMA, min_child_weight = MIN_CHILD_WEIGHT, scale_pos_weight=SCALE_POS_WEIGHT, 
                          reg_alpha=REG_ALPHA, reg_lambda=REG_LAMBDA, n_jobs=N_JOBS)
#    n_estimators=370??,
#    learning_rate=0.07, 0.02
#    min_child_weight=.77,1
#    scale_pos_weight=1.6,
#    gamma=10,1
#    reg_alpha=8,
#    reg_lambda=1.3,
                        
#    elif modelname=='XGBoost':
#        # General parameters for xgboost
#        N_ESTIMATORS = [1000]
#        LEARNING_RATE = [0.05]
#        OBJECTIVE = ["binary:logistic"]
#        COLSAMPLE_BYTREE = [0.6]
#        MAX_DEPTH = [6]
#        SUBSAMPLE = [0.6]
#        GAMMA = [0]
#        N_JOBS = [4]
#        param_grid = dict(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, objective=OBJECTIVE, colsample_bytree=COLSAMPLE_BYTREE, max_depth=MAX_DEPTH,
#                          subsample=SUBSAMPLE, gamma=GAMMA, n_jobs=N_JOBS)
        
    elif modelname=='RandomForest':
        # General parameters for RandomForestClassifier
        N_ESTIMATORS = [800]
        MAX_FEATURES = ['auto']
        MIN_SAMPLES_SPLIT = [15]
        MIN_SAMPLES_LEAF = [50] #[1,5,10,50,100,200,500]
        N_JOBS = [-1]
        VERBOSE = [1]
        param_grid = dict(n_estimators=N_ESTIMATORS, max_features=MAX_FEATURES, 
                          min_samples_leaf=MIN_SAMPLES_LEAF, min_samples_split=MIN_SAMPLES_SPLIT, 
                          n_jobs=N_JOBS, verbose=VERBOSE)
    
    elif modelname=='Linear':    
        # General parameters for LogisticRegression
        C = [1]
        TOL = [0.001]
        N_JOBS = [1]
        VERBOSE = [1]
        param_grid = dict(C=C, tol=TOL, n_jobs=N_JOBS, verbose=VERBOSE)
    
    elif modelname=='LightGBM':
        # General parameters for LGBMClassifier
        BOOSTING = ['gbdt']
        LEARNING_RATE = [0.02] #[0.02,0.05,0.1] 
        N_ESTIMATORS = [800] #[600,800,100] 
        MAX_DEPTH = [5] #[4,5,8,-1]
        SUBSAMPLE = [0.7] #[0.7,0.75,0.8,0.85,0.9]
        #feature_fraction = [0.8]
        #bagging_fraction = [0.7]
        #bagging_freq = [5]
        #min_data = [500]
        N_JOBS = [-1]
        param_grid = dict(boosting_type=BOOSTING, learning_rate=LEARNING_RATE, n_estimators=N_ESTIMATORS,  
                             max_depth=MAX_DEPTH, subsample=SUBSAMPLE, n_jobs=N_JOBS)
        
#    elif modelname=='LightGBM':
#        # General parameters for LGBMClassifier
#        BOOSTING = ['gbdt']
#        LEARNING_RATE = [0.02] 
#        N_ESTIMATORS = [1000] 
#        MAX_FEATURES = ['auto']
#        MAX_DEPTH = [4]
#        SUBSAMPLE = [0.7]
#        N_JOBS = [-1]
#        VERBOSE = [1]
#        param_grid = dict(boosting=BOOSTING, learning_rate=LEARNING_RATE, n_estimators=N_ESTIMATORS, max_features=MAX_FEATURES, 
#                             max_depth=MAX_DEPTH, subsample=SUBSAMPLE, n_jobs=N_JOBS, verbose=VERBOSE) 
    
    return param_grid
