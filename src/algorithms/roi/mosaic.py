from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from scipy import ndimage
from adapters.cancellation import check_cancelled
from .config import ROI_DTYPE

@dataclass(frozen=True)
class MosaicTile:
    image: np.ndarray | Callable[[], np.ndarray]
    fov_m: float
    center_m: tuple[float, float]
    box_id: int
    candidates: np.ndarray = field(default_factory=lambda: np.empty(0,dtype=ROI_DTYPE))
    descriptions: tuple[str,...] = ()
    shape_px: tuple[int,int] | None = None


@dataclass(frozen=True)
class ROIMosaicResult:
    image: np.ndarray
    coverage: np.ndarray
    origin_m: np.ndarray
    pixel_size_m: float
    metadata: dict

    def save(self, directory):
        from .output import save_mosaic_result
        return save_mosaic_result(self,directory)


def compose_roi_mosaic(tiles, *, max_side_px=4096):
    """Place captures in image-axis physical coordinates, blend overlaps and label ROIs.

    tiles: MosaicTile objects with arrays or lazy providers and known shapes.
    center_m is relative to a common reference in image axes, not raw stage axes.
    Uses metadata placement, without image registration/hysteresis correction.
    """
    tiles = list(tiles)
    if not tiles:
        raise ValueError('No captures for mosaic')
    if not isinstance(max_side_px,int) or isinstance(max_side_px,bool) or max_side_px < 2:
        raise ValueError('max_side_px must be at least 2')
    specs = []
    for tile in tiles:
        check_cancelled()
        shape = tile.shape_px if tile.shape_px is not None else tile.image.shape
        fov = float(tile.fov_m)
        center = np.asarray(tile.center_m, dtype=float)
        if len(shape)!=2 or min(shape)<1 or not np.isfinite(center).all() or center.shape!=(2,) or not np.isfinite(fov) or fov<=0:
            raise ValueError('Invalid tile geometry')
        pixel = fov/max(shape)
        extent = np.array([shape[1],shape[0]])*pixel
        specs.append((tile,center,extent,pixel))
    origin = np.min([center-extent/2 for _,center,extent,_ in specs],axis=0)
    end = np.max([center+extent/2 for _,center,extent,_ in specs],axis=0)
    pixel = max(min(v[3] for v in specs),float(np.max(end-origin))/(max_side_px-1))
    width,height = np.maximum(1,np.ceil((end-origin)/pixel).astype(int))
    total = np.zeros((height,width),dtype=np.float32)
    weight = np.zeros_like(total)
    candidates = []
    for tile,center,extent,native_pixel in specs:
        check_cancelled()
        left,top = center-extent/2
        x0,y0=np.maximum(0,np.floor((np.array([left,top])-origin)/pixel).astype(int))
        x1,y1=np.minimum([width,height],np.ceil((center+extent/2-origin)/pixel).astype(int))
        image=np.asarray(tile.image() if callable(tile.image) else tile.image,dtype=np.float32)
        shape = tile.shape_px if tile.shape_px is not None else tile.image.shape
        if image.shape!=tuple(shape) or not np.isfinite(image).all():
            raise ValueError('Invalid mosaic image data')
        yy,xx=np.meshgrid((origin[1]+(np.arange(y0,y1)+.5)*pixel-top)/native_pixel-.5,
                          (origin[0]+(np.arange(x0,x1)+.5)*pixel-left)/native_pixel-.5,indexing='ij')
        valid=(xx>=-.5)&(xx<image.shape[1]-.5)&(yy>=-.5)&(yy<image.shape[0]-.5)
        data=ndimage.map_coordinates(image,[yy,xx],order=1,mode='nearest')
        total[y0:y1,x0:x1]+=data*valid
        weight[y0:y1,x0:x1]+=valid
        for n,roi in enumerate(tile.candidates):
            bounds=np.array([left+roi['x0_px']*native_pixel,top+roi['y0_px']*native_pixel,
                             left+roi['x1_px']*native_pixel,top+roi['y1_px']*native_pixel])
            candidates.append({'box_id':int(tile.box_id),'roi_id':int(roi['id']),
                'center_m':(center+np.array([roi['offset_x_m'],roi['offset_y_m']])).tolist(),
                'bounds_m':bounds.tolist(),'score_max':float(roi['score_max']),
                'description':str(tile.descriptions[n]) if n<len(tile.descriptions) else ''})
    merged=np.divide(total,weight,out=np.zeros_like(total),where=weight>0)
    metadata={'coordinate_system':'image axes x right/y down, relative to particle-view center',
              'origin_m':origin.tolist(),'pixel_size_m':pixel,'shape_px':[int(height),int(width)],
              'tile_count':len(tiles),'candidates':candidates,
              'placement':'recorded stage positions transformed to image axes; no image registration',
              'overlap':'mean intensity; candidates retain box/ROI identity without deduplication'}
    return ROIMosaicResult(merged, weight, origin, pixel, metadata)


def save_roi_mosaic(tiles, output_directory, *, max_side_px=4096):
    from .output import load_mosaic_tiles
    result=compose_roi_mosaic(load_mosaic_tiles(tiles),max_side_px=max_side_px)
    result.save(output_directory)
    return result.metadata
