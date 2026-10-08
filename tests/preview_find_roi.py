"""Rebuild particle #1 normal reference and review saved particle #2 images.

Uses fixed provisional normal labels from the original preview; target images
are evaluation-only and never enter the reference memory.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from algorithms.FindROI import ROIReferenceBank, find_roi_in_image, save_roi_mosaic


def main():
    project = Path(__file__).resolve().parents[1]
    provenance = json.loads((project / 'src/scripts/roi_reference.json').read_text())
    source = project / provenance['source_run'] / 'particle_0001'
    out = project / 'tests/edge_review_results/context_v2'
    out.mkdir(parents=True, exist_ok=True)
    def sample(identifier):
        with np.load(source / f'edge_box_{identifier:04d}' / 'haadf.npz', allow_pickle=False) as data:
            return data['image'].copy(), float(data['fov_m'])
    bank = ROIReferenceBank().fit([sample(i) for i in provenance['training_box_ids']])
    bank.calibrate([sample(i) for i in provenance['validation_box_ids']], quantile=.995, margin=1.1)
    bank.save(out / 'reference_v2.npz')
    print('Thresholds:', bank.class_thresholds.tolist(), flush=True)
    root = project / 'src/scripts/particle_edge_workflow_results/20261008_164305_658648_dc9ab569/particle_0002'
    state = json.loads((root / 'reference.json').read_text())['state']
    run = json.loads((root.parent / 'run.json').read_text())
    angle = run.get('scan_rotation_rad', 0.)
    inverse = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    boxes = sorted(root.glob('edge_box_*'))
    canvas = Image.new('RGB', (6*256, ((len(boxes)+5)//6)*280), (32,32,32))
    draw = ImageDraw.Draw(canvas)
    summary, tiles = [], []
    for n, box in enumerate(boxes):
        with np.load(box / 'haadf.npz', allow_pickle=False) as data:
            result = find_roi_in_image(data['image'], fov_m=float(data['fov_m']), reference=bank,
                output_directory=out / box.name)
            center = inverse @ (np.asarray(state['stage_position_m'])-data['stage_position_m'])
        row = {'box_id': int(box.name.split('_')[-1]), 'roi_count': len(result.candidates),
            'max_score': float(result.score_map.max()), 'descriptions': list(result.descriptions)}
        summary.append(row)
        print(row['box_id'], row['roi_count'], flush=True)
        tiles.append({'box_id': row['box_id'], 'center_m': center, 'capture_path': box / 'haadf.npz',
            'roi_path': out / box.name / 'find_roi.npz'})
        with Image.open(out / box.name / 'haadf_roi.png') as image:
            canvas.paste(image.resize((256,256)), ((n%6)*256, (n//6)*280+24))
        draw.text(((n%6)*256+4, (n//6)*280+5), f"Box {row['box_id']}: {row['roi_count']} ROI", fill='white')
    (out / 'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    canvas.save(out / 'contact_sheet.png')
    save_roi_mosaic(tiles, out / 'mosaic', max_side_px=2048)
    print('Positive boxes:', [row['box_id'] for row in summary if row['roi_count']], flush=True)


if __name__ == '__main__':
    main()
