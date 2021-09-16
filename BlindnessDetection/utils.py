import torch

def load_model(filename, model, optimizer):
    print("loading models ...")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer

def save_model(filename, model, optimizer):
    checkpoint = {}
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint,filename )