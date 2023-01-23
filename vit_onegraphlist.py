import torch
from vit_pytorch import ViT

image_size=24
patch_size=8

v = ViT(
    num_graphs = 9,
    num_classes = 1,
    dim = 13, #the graph embedded vector dimentions
    depth = 4, #how many times the encoder blocks repeat
    heads = 16, #how many heads the encoder do the ensemble learning
    mlp_dim = 20, #how many inner dimentions of the MLP block, where the input and output are all of 'dim' dimentions
    dropout = 0.1,
    emb_dropout = 0.1
)

list_graph = torch.randn(1, 9,13)  #data input, a list of 9 graphs which each graph embedded vector has 13 dimentions
preds = v(list_graph) 
print(preds) #binary classification of normal and abnormal