import torch
import torch.nn.functional as F

class OT_measure:
    def __init__(self, with_geomloss, blur, std_floor=0.01, stat_clip=0.0):
        from geomloss import SamplesLoss
        self.with_geomloss = with_geomloss
        self.blur = blur
        self.std_floor = float(std_floor)
        self.stat_clip = float(stat_clip)
        if self.stat_clip <= 0:
            self.stat_clip = None
        self.loss_geom = None
        self.backend = None

        def _has_keops():
            try:
                from pykeops.torch import LazyTensor  # noqa: F401
                return True
            except Exception:
                return False

        def _make_loss(backend):
            return SamplesLoss(loss="sinkhorn", p=2, blur=self.blur, backend=backend)

        if with_geomloss == 1:
            # Prefer KeOps online backend when available; otherwise tensorized.
            self.backend = 'online' if _has_keops() else 'tensorized'
            self.loss_geom = _make_loss(self.backend)
            if self.backend == 'online':
                print('You have selected to use Sinkhorn loss (backend=online).')
            else:
                print('You have selected to use Sinkhorn loss (backend=tensorized fallback).')
        elif with_geomloss == 2:
            self.backend = 'online' if _has_keops() else 'tensorized'
            self.loss_geom = _make_loss(self.backend)
            if self.backend == 'online':
                print('You have selected to use MMD loss (backend=online).')
            else:
                print('You have selected to use MMD loss (backend=tensorized fallback).')

    def loss(self, anchor_stats, out_stats):
        if self.loss_geom is None:
            return torch.tensor(0.0, device=anchor_stats.device, dtype=anchor_stats.dtype)
        normalized_std = anchor_stats.std(dim=1, unbiased=False)[:, None, :].repeat(1, anchor_stats.shape[1], 1)
        normalized_std = torch.clamp(normalized_std, min=self.std_floor)
        normalized_mean = anchor_stats.mean(dim = 1)[:, None, :].repeat(1, anchor_stats.shape[1], 1)
        out_stats = (out_stats-normalized_mean) / (normalized_std + 1e-6)
        anchor_stats = (anchor_stats - normalized_mean) / (normalized_std + 1e-6)
        if self.stat_clip is not None:
            out_stats = torch.clamp(out_stats, min=-self.stat_clip, max=self.stat_clip)
            anchor_stats = torch.clamp(anchor_stats, min=-self.stat_clip, max=self.stat_clip)
        out_stats = torch.nan_to_num(out_stats, nan=0.0, posinf=0.0, neginf=0.0)
        anchor_stats = torch.nan_to_num(anchor_stats, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            loss_geom_list = self.loss_geom(anchor_stats.contiguous(), out_stats.contiguous())
        except NameError as exc:
            # Some environments install geomloss without a working KeOps runtime.
            if 'LazyTensor' in str(exc):
                from geomloss import SamplesLoss
                self.backend = 'tensorized'
                self.loss_geom = SamplesLoss(loss="sinkhorn", p=2, blur=self.blur, backend='tensorized')
                print('KeOps LazyTensor unavailable at runtime. Falling back to backend=tensorized.')
                loss_geom_list = self.loss_geom(anchor_stats.contiguous(), out_stats.contiguous())
            else:
                raise
        loss_geom_mean = loss_geom_list.mean()

        return loss_geom_mean
