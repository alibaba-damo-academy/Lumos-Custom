import os
import os.path as osp
import torch
import numbers
import binascii
import torchvision

class TensorList(object):

    def __init__(self, tensors):
        """
        tensors: a list of torch.Tensor objects. No need to have uniform shape.
        """
        assert isinstance(tensors, (list, tuple))
        assert all(isinstance(u, torch.Tensor) for u in tensors)
        assert len(set([u.ndim for u in tensors])) == 1
        assert len(set([u.dtype for u in tensors])) == 1
        assert len(set([u.device for u in tensors])) == 1
        self.tensors = tensors
    
    def to(self, *args, **kwargs):
        return TensorList([u.to(*args, **kwargs) for u in self.tensors])
    
    def size(self, dim):
        assert dim == 0, 'only support get the 0th size'
        return len(self.tensors)
    
    def pow(self, *args, **kwargs):
        return TensorList([u.pow(*args, **kwargs) for u in self.tensors])
    
    def squeeze(self, dim):
        assert dim != 0
        if dim > 0:
            dim -= 1
        return TensorList([u.squeeze(dim) for u in self.tensors])
    
    def type(self, *args, **kwargs):
        return TensorList([u.type(*args, **kwargs) for u in self.tensors])
    
    def type_as(self, other):
        assert isinstance(other, (torch.Tensor, TensorList))
        if isinstance(other, torch.Tensor):
            return TensorList([u.type_as(other) for u in self.tensors])
        else:
            return TensorList([u.type(other.dtype) for u in self.tensors])
    
    @property
    def dtype(self):
        return self.tensors[0].dtype
    
    @property
    def device(self):
        return self.tensors[0].device
    
    @property
    def ndim(self):
        return 1 + self.tensors[0].ndim
    
    def __getitem__(self, index):
        return self.tensors[index]
    
    def __len__(self):
        return len(self.tensors)
    
    def __add__(self, other):
        return self._apply(other, lambda u, v: u + v)
    
    def __radd__(self, other):
        return self._apply(other, lambda u, v: v + u)
    
    def __sub__(self, other):
        return self._apply(other, lambda u, v: u - v)
    
    def __rsub__(self, other):
        return self._apply(other, lambda u, v: v - u)
    
    def __mul__(self, other):
        return self._apply(other, lambda u, v: u * v)
    
    def __rmul__(self, other):
        return self._apply(other, lambda u, v: v * u)
    
    def __floordiv__(self, other):
        return self._apply(other, lambda u, v: u // v)
    
    def __truediv__(self, other):
        return self._apply(other, lambda u, v: u / v)
    
    def __rfloordiv__(self, other):
        return self._apply(other, lambda u, v: v // u)
    
    def __rtruediv__(self, other):
        return self._apply(other, lambda u, v: v / u)
    
    def __pow__(self, other):
        return self._apply(other, lambda u, v: u ** v)
    
    def __rpow__(self, other):
        return self._apply(other, lambda u, v: v ** u)
    
    def __neg__(self):
        return TensorList([-u for u in self.tensors])
    
    def __iter__(self):
        for tensor in self.tensors:
            yield tensor
    
    def __repr__(self):
        return 'TensorList: \n' + repr(self.tensors)

    def _apply(self, other, op):
        if isinstance(other, (list, tuple, TensorList)) or (
            isinstance(other, torch.Tensor) and (
                other.numel() > 1 or other.ndim > 1
            )
        ):
            assert len(other) == len(self.tensors)
            return TensorList([op(u, v) for u, v in zip(self.tensors, other)])
        elif isinstance(other, numbers.Number) or (
            isinstance(other, torch.Tensor) and (
                other.numel() == 1 and other.ndim <= 1
            )
        ):
            return TensorList([op(u, other) for u in self.tensors])
        else:
            raise TypeError(
                f'unsupported operand for *: "TensorList" and "{type(other)}"'
            )


def randn_like(tensor, generator=None):
    assert isinstance(tensor, (torch.Tensor, TensorList, list))
    if isinstance(tensor, torch.Tensor):
        return torch.randn(
            tensor.shape,
            generator=generator,
            dtype=tensor.dtype,
            device=tensor.device
        )
    else:
        out = [torch.randn(
            u.shape,
            generator=generator,
            dtype=u.dtype,
            device=u.device
        ) for u in tensor]
        if isinstance(tensor, TensorList):
            out = TensorList(out)
        return out


def to_(data, *args, **kwargs):
    if isinstance(data, (list, tuple)):
        return type(data)([to_(u, *args, **kwargs) for u in data])
    elif isinstance(data, dict):
        return type(data)([(k, to_(v, *args, **kwargs)) for k, v in data.items()])
    elif isinstance(data, torch.Tensor):
        data = data.to(*args, **kwargs)
    return data

def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name