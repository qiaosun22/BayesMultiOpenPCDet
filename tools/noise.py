import torch
#from GPU import to_device, get_default_device
#device = get_default_device()
import copy

# Adding Exponential Gaussian Noise to the Model
def add_noise_to_weights(mean, std, model):
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.1)
    """
    model = copy.deepcopy(model)
    gaussian_kernel = torch.distributions.Normal(mean, std)
    with torch.no_grad():
        for param in model.parameters():                  
            param.mul_(torch.exp(gaussian_kernel.sample(param.size())).cuda())
    return model
    #print("Noise added, the standard deviation is: ", std)