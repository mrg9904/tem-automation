"""Result serialization and visualization; no detection decisions."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

def save_roi_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/'find_roi.npz', schema_version=2,
        image=result.image, fov_m=result.fov_m, score_map=result.score_map,
        labels=result.labels, candidates=result.candidates, descriptions=np.asarray(result.descriptions, dtype="U128"), threshold=result.threshold,
        model_id=result.model_id, method='local_spatial_fft_context_v2',
        reference_score_map=np.empty(0) if result.reference_score_map is None else result.reference_score_map,
        context_baseline_map=np.empty(0) if result.context_baseline_map is None else result.context_baseline_map,
        threshold_map=np.empty(0) if result.threshold_map is None else result.threshold_map,
        coordinate_system='image_axes: x right, y down; origin FoV center')
    data = result.image.astype(float)
    low,high = np.percentile(data,[1,99])
    gray = np.uint8(np.clip((data-low)/max(high-low,1e-12),0,1)*255)
    Image.fromarray(gray).save(directory/'haadf.png')
    annotated = Image.fromarray(gray).convert('RGB')
    draw = ImageDraw.Draw(annotated)
    for roi in result.candidates:
        bounds = (int(roi['x0_px']),int(roi['y0_px']),int(roi['x1_px'])-1,int(roi['y1_px'])-1)
        draw.rectangle(bounds, outline=(255,160,0), width=2)
        x,y = float(roi['center_x_px']),float(roi['center_y_px'])
        draw.line((x-3,y,x+3,y),fill=(255,0,0),width=1)
        draw.line((x,y-3,x,y+3),fill=(255,0,0),width=1)
        draw.text((bounds[0]+2,max(0,bounds[1]-12)),f"ROI {int(roi['id'])}: {roi['score_max']:.2f}",
                  fill=(255,255,0),stroke_width=1,stroke_fill=(0,0,0))
    annotated.save(directory/'haadf_roi.png')
    score = np.uint8(np.clip(result.score_map/max(result.threshold*2,1e-12),0,1)*255)
    Image.fromarray(score).save(directory/'roi_score_map.png')
    return directory


def save_mosaic_result(result, output_directory):
    merged,weight,origin,pixel = result.image,result.coverage,result.origin_m,result.pixel_size_m
    candidates=result.metadata["candidates"]
    low,high=np.percentile(merged[weight>0],[1,99])
    gray=np.uint8(np.clip((merged-low)/max(high-low,1e-12),0,1)*255)
    gray[weight==0]=0
    annotated=Image.fromarray(gray).convert('RGB')
    draw=ImageDraw.Draw(annotated)
    for roi in candidates:
        b=np.asarray(roi['bounds_m']).reshape(2,2)
        b=((b-origin)/pixel).ravel()
        draw.rectangle(tuple(b),outline=(255,160,0),width=2)
        cx,cy=(np.asarray(roi['center_m'])-origin)/pixel-.5
        draw.line((cx-3,cy,cx+3,cy),fill=(255,0,0))
        draw.line((cx,cy-3,cx,cy+3),fill=(255,0,0))
        draw.text((b[0]+2,max(0,b[1]-12)),f"B{roi['box_id']}/R{roi['roi_id']} {roi['score_max']:.2f}",
                  fill=(255,255,0),stroke_width=1,stroke_fill=(0,0,0))
    output=Path(output_directory);output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(gray).save(output/'haadf_mosaic.png')
    annotated.save(output/'haadf_mosaic_roi.png')
    (output/'roi_mosaic.json').write_text(json.dumps(result.metadata,indent=2),encoding='utf-8')
    np.savez_compressed(output/'haadf_mosaic.npz',image=merged,coverage=weight,
                        origin_m=origin,pixel_size_m=pixel)
    return output


def load_mosaic_tiles(tiles):
    """Read lightweight geometry now and decode one image at a time during composition."""
    from zipfile import ZipFile
    from adapters.cancellation import check_cancelled
    from .config import ROI_DTYPE
    from .mosaic import MosaicTile
    def load_image(path):
        with np.load(path,allow_pickle=False) as data:
            return data['image']
    for tile in tiles:
        check_cancelled()
        if isinstance(tile,MosaicTile):
            yield tile
            continue
        path=Path(tile['capture_path'])
        with np.load(path,allow_pickle=False) as data:
            fov=float(data['fov_m'])
        with ZipFile(path) as archive, archive.open('image.npy') as stream:
            version=np.lib.format.read_magic(stream)
            if version==(1,0):
                shape,_,_=np.lib.format.read_array_header_1_0(stream)
            elif version==(2,0):
                shape,_,_=np.lib.format.read_array_header_2_0(stream)
            else:
                shape=load_image(path).shape
        candidates=np.empty(0,dtype=ROI_DTYPE)
        descriptions=()
        roi_path=tile.get('roi_path')
        if roi_path is not None and Path(roi_path).is_file():
            with np.load(roi_path,allow_pickle=False) as data:
                candidates=data['candidates'].copy()
                descriptions=tuple(data['descriptions'].tolist()) if 'descriptions' in data.files else ()
        yield MosaicTile(lambda path=path:load_image(path),fov,tuple(tile['center_m']),
                         int(tile['box_id']),candidates,descriptions,tuple(shape))
