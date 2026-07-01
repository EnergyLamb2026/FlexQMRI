"""Example: T2* pipeline — load multi-echo data or vendor map, resample anatomical, compute R2* maps.

Pipeline:
1. Load T2* patient data: raw multi-echo NIfTIs (source: fit) or vendor precomputed map (source: vendor).
2. Normalize the multi-echo signal (source: fit only).
3. Resample an anatomical scan (e.g. T1w) and its mask onto the T2* grid.
4. Compute R2* maps: fit a T2* decay model and invert T2* = 1/R2* (fit), or apply mask to precomputed map (vendor).
"""

import argparse

import flexqmri
import flexqmri.imaging.loaders
import flexqmri.imaging.config
import flexqmri.preprocessing.normalization
import flexqmri.preprocessing.align
from flexqmri.evaluation.map_plots import plot_maps_slice

import flexqmri.imaging.patient as load_data

from flexqmri.compute_maps import r2star
from flexqmri.analysis.roi import save_roi_stats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run full R2* patient pipeline.")
    parser.add_argument("-p", "--patient_id", type=str, required=True,
                        help="Patient identifier.")
    parser.add_argument("-s", "--study", type=str, default="Study1",
                        help="Study folder name inside the patient NIfTI directory.")
    parser.add_argument("-c", "--common_config_path", type=str, required=True,
                        help="Path to the common config file.")
    parser.add_argument("--pipeline", type=str, default="standard",
                        help="Pipeline label used to organise outputs (default: 'standard').")
    return parser.parse_args()

def main():
    args = parse_args()
    config = flexqmri.imaging.config.load_config_hierarchical("t2star", args.common_config_path)
    output_dir = config['paths']['output_data']

    study_path, all_series = flexqmri.imaging.loaders.prepare_study(config, args.patient_id, args.study)

    # --- 1. Load T2* data ---
    t2star_source = config['t2star']['source']
    config['anatomical'] = config['t2star']['anatomical']
    if t2star_source == 'fit':
        t2star_data = load_data.load_t2star_patient_data(
            study_path=study_path,
            all_series=all_series,
            config=config,
            patient_id=args.patient_id,
            slice_idx=config["common"]["slice"],
        )
    else:
        t2star_data = flexqmri.imaging.loaders.load_general_patient_data(
            study_path=study_path,
            all_series=all_series,
            config=config,
            patient_id=args.patient_id,
            keywords=config["t2star"]["loading"]["keywords"],
            include=None,
            exclude=config["t2star"]["loading"]["include"],
            slice_idx=config["common"]["slice"],
        )

    print("PatientData created:")
    print(f"  - Data shape: {t2star_data.data.shape}")
    print(f"  - Has mask: {t2star_data.has_mask()}")
    if t2star_source == 'fit':
        print(f"  - Echo times (ms): {t2star_data.x}")

    # --- 2. Normalization ---
    if t2star_source == 'fit':
        norm_method = config["t2star"]["normalization"]["method"]
        norm_data = flexqmri.preprocessing.normalization.normalize_4D_stack(t2star_data.data, norm_method)
        t2star_data.update_data(norm_data)
        print(f"Normalization applied: method='{norm_method}'")

    # --- 3. Resample anatomical onto T2* grid ---
    anatomical_data = flexqmri.imaging.loaders.load_general_patient_data(
        study_path=study_path,
        all_series=all_series,
        config=config,
        patient_id=args.patient_id,
        keywords=config["anatomical"]["loading"]["keywords"],
        include=config["anatomical"]["loading"]["include"],
        exclude=config["anatomical"]["loading"]["exclude"],
        slice_idx=config["common"]["slice"],
    )
    print(f"Loaded anatomical: series='{anatomical_data.serie_name}', resolution={anatomical_data.pixel_spacing}")

    aligned = flexqmri.preprocessing.align.align_patient_data(
        quantitative_data=t2star_data,
        anatomical_data=anatomical_data,
    )
    print(f"Resampled anatomical onto T2* grid. Shape: {aligned['aligned_anatomical'].shape}")
    if aligned['aligned_mask'] is not None:
        t2star_data.mask_array = aligned['aligned_mask']

    # --- 4. Compute R2* maps ---
    r2star_time = None
    if t2star_source == 'fit':
        r2star_maps, model_name, r2star_time = r2star.compute_r2star_map(
            config=config,
            patient_data=t2star_data,
            patient_id=args.patient_id,
            study=args.study,
            pipeline=args.pipeline,
        )
    else:
        r2star_maps = r2star.compute_r2star_map_from_t2star(
            config=config,
            patient_data=t2star_data,
            patient_id=args.patient_id,
            study=args.study,
            pipeline=args.pipeline,
        )
        model_name = 'vendor'

    plot_maps_slice(
        r2star_maps,
        map_keys=['R2star'],
        titles=['R2* map'],
        colorbar_labels=['s⁻¹'],
        modality='r2star',
        model_name=model_name,
        output_dir=output_dir,
        patient_id=args.patient_id,
        study=args.study,
        mask_array=t2star_data.mask_array,
        anatomical=aligned['aligned_anatomical'],
        slice_idx=config['common']['slice'],
        save=config['common']['save'],
        display=config['common']['display'],
        remove_outliers=True,
    )

    save_roi_stats(
        maps_name='r2star',
        maps=r2star_maps,
        mask=t2star_data.mask_array,
        map_keys=['R2star'],
        model_name=model_name,
        patient_id=args.patient_id,
        study=args.study,
        output_dir=output_dir,
        pipeline=args.pipeline,
        inference_time_per_voxel=r2star_time,
    )

if __name__ == "__main__":
    main()
