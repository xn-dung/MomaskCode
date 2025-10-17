import random
from math import ceil
from functools import partial
from itertools import zip_longest
from random import randrange
import torch
from torch import nn
import torch.nn.functional as F
from vq_model.quantizer import QuantizeEMAReset, QuantizeEMA

from einops import rearrange, repeat, pack, unpack



def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult



class ResidualVQ(nn.Module):
    def __init__(
        self,
        num_quantizers,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        nb_code=None,
        **kwargs
    ):
        super().__init__()
        assert len(nb_code) == num_quantizers or (len(nb_code) == 1 and shared_codebook == True), f"The number of codebook sizes is {len(nb_code)} and the number of codebooks is {num_quantizers}"
        self.num_quantizers = num_quantizers
        
        if shared_codebook:
            layer = QuantizeEMAReset(nb_code,**kwargs)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.nb_code = nb_code
            self.layers = nn.ModuleList([QuantizeEMAReset(self.nb_code[i],**kwargs) for i in range(num_quantizers)])

        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob

            
    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks 
    
    def get_codes_from_indices(self, indices): 

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        

        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

       

        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b = batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d = codebooks.shape[-1])

       

        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0) 
       
        all_codes = codebooks.gather(2, gather_indices) 


        all_codes = all_codes.masked_fill(mask, 0.)

        return all_codes 

    def get_codebook_entry(self, indices): 
        all_codes = self.get_codes_from_indices(indices) 
        latent = torch.sum(all_codes, dim=0)
        latent = latent.permute(0, 2, 1)
        return latent

    def forward(self, x, return_all_codes = False, sample_codebook_temp = None, force_dropout_index=-1):
        
        num_quant, quant_dropout_prob, device = self.num_quantizers, self.quantize_dropout_prob, x.device

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_perplexity = []


        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob

        start_drop_quantize_index = num_quant
       
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant) 
            null_indices_shape = [x.shape[0], x.shape[-1]]
            null_indices = torch.full(null_indices_shape, -1., device = device, dtype = torch.long)
         

        if force_dropout_index >= 0:
            should_quantize_dropout = True
            start_drop_quantize_index = force_dropout_index
            null_indices_shape = [x.shape[0], x.shape[-1]]  
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)

       

        for quantizer_index, layer in enumerate(self.layers):

            if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                all_indices.append(null_indices)
                continue

            
            quantized, *rest = layer(residual, return_idx=True, temperature=sample_codebook_temp) 
            residual -= quantized.detach()
            quantized_out += quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)


        all_indices = torch.stack(all_indices, dim=-1)
        all_losses = sum(all_losses)/len(all_losses)
        all_perplexity = sum(all_perplexity)/len(all_perplexity)

        ret = (quantized_out, all_indices, all_losses, all_perplexity)

        if return_all_codes:
            
            all_codes = self.get_codes_from_indices(all_indices)
            ret = (*ret, all_codes)

        return ret
    
    def quantize(self, x, return_latent=False):
        all_indices = []
        quantized_out = 0.
        residual = x
        all_codes = []
        for quantizer_index, layer in enumerate(self.layers):

            quantized, *rest = layer(residual, return_idx=True) 

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_codes.append(quantized)

        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        if return_latent:
            return code_idx, all_codes
        return code_idx