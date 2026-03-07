def has_trained_models():
    # check if a saved model file exists
    import os
    return os.path.exists("Results/saved_model.pkl")