"""Example: IVIM pipeline load, normalize, denoise, resample anatomical onto IVIM grid.

Pipeline:
1. Load IVIM patient data from NIfTI files.
2. Normalize the IVIM data by S0.
3. Denoise with PCA, MPPCA, or NLM.
4. Resample an anatomical scan (e.g. T1w) and its mask onto the IVIM grid.
"""

import argparse

import flexqmri
import flexqmri.imaging.loaders
import flexqmri.imaging.config
import flexqmri.preprocessing.denoising
import flexqmri.preprocessing.normalization
import flexqmri.preprocessing.align
from flexqmri.evaluation.map_plots import plot_maps_slice

import flexqmri.imaging.patient as load_data

from flexqmri.compute_maps import ivim
from flexqmri.imaging.savers import save_mask, save_nifti_bvalues
from flexqmri.analysis.roi import save_roi_stats

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run full IVIM patient pipeline.")
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
    config = flexqmri.imaging.config.load_config_hierarchical("ivim", args.common_config_path)
    output_dir = config['paths']['output_data']

    study_path, all_series = flexqmri.imaging.loaders.prepare_study(config, args.patient_id, args.study)

    # Load IVIM data using factory function
    ivim_data = load_data.load_ivim_patient_data(
        study_path=study_path,
        all_series=all_series,
        config=config,
        patient_id=args.patient_id,
        slice_idx=config["common"]["slice"],
    )

    print("PatientData created:")
    print(f"  - Data shape: {ivim_data.data.shape}")
    print(f"  - B-values: {ivim_data.x}")

    # --- 1. Normalization ---
    norm_method = config["ivim"]["normalization"]["method"]
    norm_data = flexqmri.preprocessing.normalization.normalize_4D_stack(ivim_data.data, norm_method)
    ivim_data.update_data(norm_data)
    print(f"Normalization applied: method='{norm_method}'")

    # --- 2. Denoising ---
    denoise_method = config["ivim"]["denoising"]["method"]
    denoise_criterion = config["ivim"]["denoising"]["pca_criterion"]
    denoised_data, _ = flexqmri.preprocessing.denoising.apply_denoising(
        ivim_data.data,
        method=denoise_method,
        criterion=denoise_criterion,
        mask=ivim_data.mask_array,
    )
    ivim_data.update_data(denoised_data)
    print(f"Denoising applied: method='{denoise_method}'")

    # --- 3. Resample anatomical onto IVIM grid ---
    config['anatomical'] = config['ivim']['anatomical']
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
        quantitative_data=ivim_data,
        anatomical_data=anatomical_data,
    )
    print(f"Resampled anatomical onto IVIM grid. Shape: {aligned['aligned_anatomical'].shape}")
    if aligned['aligned_mask'] is not None:
        ivim_data.mask_array = aligned['aligned_mask']

    save_nifti_bvalues(ivim_data.data, ivim_data)

    # --- 4. Compute IVIM maps ---
    ivim_maps, model_name, ivim_time = ivim.compute_ivim_maps(
        config=config,
        patient_data=ivim_data,
        patient_id=args.patient_id,
        study=args.study,
        pipeline=args.pipeline,
    )

    if config['ivim']['maps']['save']:
        save_mask(ivim_data.mask_array, ivim_data, output_dir, args.patient_id, args.study, args.pipeline, 'ivim')

    plot_maps_slice(
        ivim_maps,
        map_keys=['f', 'D', 'D_star'],
        titles=['Perfusion fraction f', 'Diffusion D', 'Pseudo-diffusion D*'],
        colorbar_labels=['', 'mm²/s', 'mm²/s'],
        modality='ivim',
        model_name=model_name,
        output_dir=output_dir,
        patient_id=args.patient_id,
        study=args.study,
        mask_array=ivim_data.mask_array,
        anatomical=aligned['aligned_anatomical'],
        slice_idx=config['common']['slice'],
        save=config['common']['save'],
        display=config['common']['display'],
    )

    save_roi_stats(
        maps_name='ivim',
        maps=ivim_maps,
        mask=ivim_data.mask_array,
        map_keys=['f', 'D', 'D_star'],
        model_name=model_name,
        patient_id=args.patient_id,
        study=args.study,
        output_dir=output_dir,
        pipeline=args.pipeline,
        inference_time_per_voxel=ivim_time,
    )

if __name__ == "__main__":
    main()
