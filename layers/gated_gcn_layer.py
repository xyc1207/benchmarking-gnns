import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

class GatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']    
        e_ij = edges.data['Ce'] +  edges.src['Dh'] + edges.dst['Eh'] # e_ij = Ce_ij + Dhi + Ehj
        edges.data['e'] = e_ij
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e) # sigma_ij = sigmoid(e_ij)
        #h = Ah_i + torch.mean( sigma_ij * Bh_j, dim=1 ) # hi = Ahi + mean_j alpha_ij * Bhj 
        h = Ah_i + torch.sum( sigma_ij * Bh_j, dim=1 ) / ( torch.sum( sigma_ij, dim=1 ) + 1e-6 )  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention       
        return {'h' : h}
    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e)
        g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)

    
##############################################################
#
# Additional layers for edge feature/representation analysis
#
##############################################################


class GatedGCNLayerEdgeFeatOnly(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']    
        e_ij = edges.src['Dh'] + edges.dst['Eh'] # e_ij = Dhi + Ehj
        edges.data['e'] = e_ij
        return {'Bh_j' : Bh_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij'] 
        sigma_ij = torch.sigmoid(e) # sigma_ij = sigmoid(e_ij)
        h = Ah_i + torch.sum( sigma_ij * Bh_j, dim=1 ) / ( torch.sum( sigma_ij, dim=1 ) + 1e-6 )  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention
        return {'h' : h}
    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization    
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)


##############################################################


class GatedGCNLayerIsotropic(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']
        return {'Bh_j' : Bh_j}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        h = Ah_i + torch.sum( Bh_j, dim=1 )  # hi = Ahi + sum_j Bhj
        return {'h' : h}
    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h)
        g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization    
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)
    

class GatedGCNLayerV2(GatedGCNLayer):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__(input_dim, output_dim, dropout, batch_norm, residual)
        self.FFN1 = nn.Linear(input_dim, 2 * input_dim, bias=True)
        self.FFN2 = nn.Linear(2 * input_dim, output_dim, bias=True)

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e)  # sigma_ij = sigmoid(e_ij)
        # h = Ah_i + torch.mean( sigma_ij * Bh_j, dim=1 ) # hi = Ahi + mean_j alpha_ij * Bhj
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / (torch.sum(sigma_ij, dim=1) + 1e-6)
        # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention
        h = self.FFN1(h)
        h = F.relu(h)
        h = self.FFN2(h)
        return {'h': h}


class GatedGCNLayerV3(GatedGCNLayer):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False, layer_norm=False):
        if batch_norm is True and layer_norm is True:
            raise Exception("Both types of normalization are True, not allowed")

        super().__init__(input_dim, output_dim, dropout, batch_norm, residual)
        self.layer_norm = layer_norm
        self.ln_node_h = nn.LayerNorm(output_dim)
        self.ln_node_e = nn.LayerNorm(output_dim)

    def forward(self, g, h, e):
        h_in = h  # for residual connection
        e_in = e  # for residual connection

        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization
            e = self.bn_node_e(e)  # batch normalization
        if self.layer_norm:
            h = self.ln_node_h(h)  # batch normalization
            e = self.ln_node_e(e)  # batch normalization

        h = F.relu(h)  # non-linear activation
        e = F.relu(e)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, e


class GatedGCNLayerV4(GatedGCNLayer):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False, layer_norm=False):
        if batch_norm is True and layer_norm is True:
            raise Exception("Both types of normalization are True, not allowed")

        super().__init__(input_dim, output_dim, dropout, batch_norm, residual)
        self.layer_norm = layer_norm
        if batch_norm:
            h_norm = nn.BatchNorm1d
        else:
            h_norm = nn.LayerNorm

        self.norm_h_1 = h_norm(input_dim)
        self.norm_h_2 = h_norm(output_dim)
        self.norm_e_1 = h_norm(input_dim)
        self.norm_e_2 = h_norm(output_dim)

        self.n_FFN1 = nn.Linear(input_dim, 2 * input_dim, bias=True)
        self.n_FFN2 = nn.Linear(2 * input_dim, output_dim, bias=True)
        self.e_FFN1 = nn.Linear(input_dim, 2 * input_dim, bias=True)
        self.e_FFN2 = nn.Linear(2 * input_dim, output_dim, bias=True)

    def forward(self, g, h, e):
        h_in = h  # for residual connection
        e_in = e  # for residual connection

        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution

        h += h_in
        e += e_in
        h = self.norm_h_1(h)  # batch normalization
        e = self.norm_e_1(e)  # batch normalization

        h_in = h
        e_in = e
        h = self.n_FFN1(h)
        h = F.relu(h)
        h = self.n_FFN2(h)

        e = self.e_FFN1(e)
        e = F.relu(e)
        e = self.e_FFN2(e)

        h += h_in
        e += e_in
        h = self.norm_h_2(h)  # batch normalization
        e = self.norm_e_2(e)  # batch normalization

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, e


class GatedGCNLayerV5(GatedGCNLayer):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False, layer_norm=False):
        if batch_norm is True and layer_norm is True:
            raise Exception("Both types of normalization are True, not allowed")
        super().__init__(input_dim, output_dim, dropout, batch_norm, residual)
        self.bn_node_h_2 = nn.BatchNorm1d(output_dim)
        self.bn_node_e_2 = nn.BatchNorm1d(output_dim)

    def forward(self, g, h, e):
        h_in = h  # for residual connection
        e_in = e  # for residual connection

        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution

        h = self.bn_node_h(h)
        e = self.bn_node_e(e)
        h = F.relu(h)
        e = F.relu(e)

        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution

        h = self.bn_node_h_2(h)
        e = self.bn_node_e_2(e)
        h += h_in
        e += e_in
        h = F.relu(h)
        e = F.relu(e)

        return h, e


class GatedGCNLayerV6(GatedGCNLayer):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False, layer_norm=False):
        if batch_norm is True and layer_norm is True:
            raise Exception("Both types of normalization are True, not allowed")
        super().__init__(input_dim, output_dim, dropout, batch_norm, residual)
        self.bn_node_h_2 = nn.BatchNorm1d(output_dim)
        self.bn_node_e_2 = nn.BatchNorm1d(output_dim)

    def forward(self, g, h, e):
        h_in = h  # for residual connection
        e_in = e  # for residual connection

        # first part processing
        h = self.bn_node_h(h)
        e = self.bn_node_e(e)
        h = F.relu(h)
        e = F.relu(e)
        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution

        # second part processing
        h = self.bn_node_h_2(h)
        e = self.bn_node_e_2(e)
        h = F.relu(h)
        e = F.relu(e)
        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution

        h += h_in
        e += e_in

        return h, e
