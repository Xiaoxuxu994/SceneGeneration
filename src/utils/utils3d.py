import torch
from torch import Tensor

__all__ = [
    'toarray',
    'batched',
]



def totensor(
    *args_dtypes: Union[torch.dtype, Tuple[torch.dtype, torch.device], str, None], 
    _others: Union[torch.dtype, str] = None, 
    **kwargs_dtypes: Union[torch.dtype, Tuple[torch.dtype, torch.device], str]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator generator that converts non-array arguments to array of specified default dtype.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        argnames = list(inspect.signature(func).parameters.keys())
        dtypes_dict = {
            **dict(zip(argnames, args_dtypes)),
            **kwargs_dtypes
        }
        @wraps(func)
        @suppress_traceback
        def wrapper(*args, **kwargs):
            inputs = {
                **{argnames[i]: x for i, x in enumerate(args)},
                **kwargs
            }
            args = tuple(
                torch.tensor(x).to(inputs[dtype_device] if isinstance(dtype_device, str) else dtype_device)
                if isinstance(x, (Number, list, tuple)) \
                    and (dtype_device := dtypes_dict.get(argnames[i], _others)) is not None \
                else x
                for i, x in enumerate(args)
            )
            kwargs = {
                k: torch.tensor(x).to(inputs[dtype_device] if isinstance(dtype_device, str) else dtype_device)
                if isinstance(x, (Number, list, tuple)) \
                    and (dtype_device := dtypes_dict.get(k, _others)) is not None \
                else x
                for k, x in kwargs.items()
            }
            return func(*args, **kwargs)
        return wrapper
    return decorator


def batched(*args_dims: Union[int, None], _others: Union[int, None] = None, **kwargs_dims: Union[int, None]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator generator that extends a function's input and out batch dimensions.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        argnames = list(inspect.signature(func).parameters.keys())
        dims_dict = {
            **dict(zip(argnames, args_dims)),
            **kwargs_dims
        }
        @wraps(func)
        @suppress_traceback
        def wrapper(*args, **kwargs):
            args = list(args)
            # Get arguments non-batch dimensions
            args_dim = tuple(dims_dict.get(argname, _others) for argname in argnames[:len(args)])
            kwargs_dim = {k: dims_dict.get(k, _others) for k in kwargs}
            # Find the common batch shape
            batch_shape = torch.broadcast_shapes(*(
                x.shape[:x.ndim - dim] 
                for x, dim in zip((*args, *kwargs.values()), (*args_dim, *kwargs_dim.values())) 
                if isinstance(x, Tensor) and dim is not None
            ))
            # Broadcast and flatten batch dimensions
            args = tuple(
                torch.broadcast_to(x, (*batch_shape, *x.shape[x.ndim - dim:])).reshape((-1, *x.shape[x.ndim - dim:]))
                if isinstance(x, Tensor) and dim is not None else x
                for x, dim in zip(args, args_dim)
            )
            kwargs = {
                k: torch.broadcast_to(x, (*batch_shape, *x.shape[x.ndim - dim:])).reshape((-1, *x.shape[x.ndim - dim:]))
                if isinstance(x, Tensor) and (dim := kwargs_dim[k]) is not None else x
                for k, x in kwargs.items()
            }
            # Call function
            result = func(*args, **kwargs)
            # Restore batch shape
            if isinstance(result, tuple):
                result = tuple(
                    x.reshape((*batch_shape, *x.shape[1:])) if isinstance(x, Tensor) else x
                    for x in result
                )
            elif isinstance(result, Tensor):
                result = result.reshape((*batch_shape, *result.shape[1:]))
            return result
        return wrapper
    return decorator

def view_to_extrinsics(view: Tensor) -> Tensor:
    """
    OpenGL view matrix to OpenCV camera extrinsics

    ## Parameters
        view (Tensor): [..., 4, 4] OpenGL view matrix

    ## Returns
        (Tensor): [..., 4, 4] OpenCV camera extrinsics matrix
    """
    return view  * torch.tensor([1, -1, -1, 1], dtype=view.dtype, device=view.device)[:, None]


@batched(2, 1)
def make_affine_matrix(M: Tensor, t: Tensor):
    """
    Make an affine transformation matrix from a linear matrix and a translation vector.

    ## Parameters
        M (Tensor): [..., D, D] linear matrix (rotation, scaling or general deformation)
        t (Tensor): [..., D] translation vector

    ## Returns
        Tensor: [..., D + 1, D + 1] affine transformation matrix
    """
    return torch.cat([
        torch.cat([M, t[..., None]], dim=-1),
        torch.tensor([0, 0, 0, 1], dtype=M.dtype, device=M.device).expand(*M.shape[:-2], 1, 4)
    ], dim=-2)

@totensor(_others=torch.float32)
@batched(1, 1, 1)
def view_look_at(
    eye: Tensor,
    look_at: Tensor,
    up: Tensor
) -> Tensor:
    """
    Get OpenGL view matrix looking at something

    ## Parameters
        eye (Tensor): [..., 3] the eye position
        look_at (Tensor): [..., 3] the position to look at
        up (Tensor): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

    ## Returns
        (Tensor): [..., 4, 4], view matrix
    """
    z = eye - look_at
    x = torch.cross(up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], dim=-2)
    R = F.normalize(R, dim=-1)
    t = -torch.matmul(R, eye[..., None])
    return make_affine_matrix(R, t.squeeze(-1))


@totensor(_others=torch.float32)
@batched(1, 1, 1)
def extrinsics_look_at(
    eye: Tensor,
    look_at: Tensor,
    up: Tensor
) -> Tensor:
    """
    Get OpenCV extrinsics matrix looking at something

    ## Parameters
        eye (Tensor): [..., 3] the eye position
        look_at (Tensor): [..., 3] the position to look at
        up (Tensor): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

    ## Returns
        (Tensor): [..., 4, 4], extrinsics matrix
    """
    return view_to_extrinsics(view_look_at(eye, look_at, up))

