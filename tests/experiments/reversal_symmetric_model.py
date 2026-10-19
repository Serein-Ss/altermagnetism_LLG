"""Site-order reversal projection for the homogeneous Bauer open chain only.

This is not spin-rotation equivariance, a Hamiltonian input, or a learned-law
certificate. Arbitrary heterogeneous/directed graphs are deliberately unsupported.
For site reversal R, P(v)(x,a) = [v(x,a) + R v(Rx,Ra)]/2.
Since R^2=I, P(v)(Rx,Ra)=R P(v)(x,a). Both terms are tangent
at x when the backbone is tangent, and both preserve its fixed anchor frame.
It doubles backbone calls, but introduces no new trainable parameters.
"""
from torch import nn


class ReversalSymmetricPathNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone=backbone

    def forward(self,state,tau,initial,condition,time,graph,axis,*,tangent=True):
        if graph is not None or axis is not None:
            raise ValueError('homogeneous diagnostic chain interface only')
        if state.ndim!=6 or state.shape[2]!=1 or state.shape[4]!=1:
            raise ValueError('one-basis chain [B,F,1,L,1,3] required')
        direct=self.backbone(state,tau,initial,condition,time,None,None,tangent=tangent)
        reverse=self.backbone(state.flip(-3),tau,initial.flip(-3),condition,time,
                              None,None,tangent=tangent).flip(-3)
        return (direct+reverse)*.5
