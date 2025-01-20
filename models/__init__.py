MODEL_REGISTRY = {
    # add models
}

def get_model(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]() 