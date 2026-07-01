import argparse
import os
import shutil
import subprocess

def convert_siemens_dicom_to_nifti(data_path: str, output_path: str, patient_id: str) -> None:
    """Convert Siemens DICOMs to NIfTI format.

    Works for files already ordered by the Siemens MRI computer (interoperable)
    or after having used :func:`order_dicom_directory`.

    Args:
        data_path (str): Path to the root directory containing patient DICOM
            folders.
        output_path (str): Path to the directory where NIfTI files are saved.
        patient_id (str): Patient identifier.

    Returns:
        None

    Raises:
        RuntimeError: If the DICOM-to-NIfTI conversion fails for a series.
    """
    print("Processing patient:", patient_id)
    patient_path = os.path.join(data_path, patient_id)
    if not os.path.exists(patient_path):
        raise FileNotFoundError(f"DICOM source path does not exist: {patient_path}")
    os.makedirs(output_path, exist_ok=True)
    studies = sorted(s for s in os.listdir(patient_path) if s != 'UnknownStudyDate')
    for idx, study in enumerate(studies, start=1):
        anonymized_study = f"Study{idx}"
        series = os.listdir(os.path.join(data_path, patient_id, study))
        series.sort()

        for serie in series:

            dicom_folder_path = os.path.join(data_path, patient_id, study, serie)

            nifti_folder_path = os.path.join(output_path, patient_id, anonymized_study, serie, 'images/') ## all offsets will be saved in the same folder
            if not os.path.exists(nifti_folder_path):
                os.makedirs(nifti_folder_path)
            if os.path.exists(dicom_folder_path):
                try: 
                    print(f"Processing folder: {dicom_folder_path}")
                    cmd = ["dicom2nifti", "-R", f"{dicom_folder_path}", f"{nifti_folder_path}"]
                    subprocess.run(cmd)
                except RuntimeError as e:
                    print(f"Error converting {dicom_folder_path} to NIfTI: {e}")
            else:
                print(f'{dicom_folder_path} does not exist')  

def convert_series_dicom_to_nifti(dicom_folder: str, output_folder: str) -> None:
    """Convert a single DICOM series folder to NIfTI format.

    Runs ``dicom2nifti -R <dicom_folder> <output_folder>`` via subprocess and
    creates *output_folder* if it does not already exist.

    Args:
        dicom_folder (str): Path to the directory containing DICOM files for
            a single series.
        output_folder (str): Path to the directory where the converted NIfTI
            file is written.

    Returns:
        None

    Raises:
        RuntimeError: If the ``dicom2nifti`` command exits with a non-zero
            return code.
    """
    os.makedirs(output_folder, exist_ok=True)
    cmd = ["dicom2nifti", "-R", dicom_folder, output_folder]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"dicom2nifti conversion failed for '{dicom_folder}': {result.stderr.strip()}"
        )


def order_dicom_directory(patient_id: str, dicom_data_path: str, sorted_data_path: str) -> None:
    """Order DICOM files into a structured directory format using dicomsort.

    Args:
        patient_id (str): Patient identifier. Pass ``'0'`` to process all
            patients found under *dicom_data_path*.
        dicom_data_path (str): Path to the root directory containing raw DICOM
            patient folders.
        sorted_data_path (str): Path to the output directory where sorted DICOM
            files are written.

    Returns:
        None
    """

    if patient_id == '0':
        if not os.path.exists(dicom_data_path):
            raise FileNotFoundError(f"DICOM source path does not exist: {dicom_data_path}")
        patient_ids = os.listdir(dicom_data_path)
    else:
        patient_path = os.path.join(dicom_data_path, patient_id)
        if not os.path.exists(patient_path):
            raise FileNotFoundError(f"DICOM source path does not exist: {patient_path}")
        patient_ids = [patient_id]
    os.makedirs(sorted_data_path, exist_ok=True)

    for patient_id in patient_ids:
        cmd = ["dicomsort", "-d", os.path.join(dicom_data_path, patient_id),
               os.path.join(sorted_data_path, patient_id,
                            "%StudyDate/%SeriesDescription_%SeriesNumber_%SeriesInstanceUID_%EchoNumbers/%InstanceNumber.dcm")]
        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_path', type=str, default='./data/dicom_data/', help='Path to the directory containing DICOM files')
    parser.add_argument('-s', '--sorted_data_path', type=str, default='./data/sorted_dicom_data/', help='Path to the directory where sorted DICOM files are stored')
    parser.add_argument('-o', '--output_path', type=str, default='./data/nifti_data/', help='Path to the directory where NIfTI files will be saved')
    parser.add_argument('-p', '--patient_id', type=str, default='0', help='Patient ID to process')
    parser.add_argument('-i', '--siemens_format', type=bool, default=False, help='Set to true if Siemens ordered DICOMs')

    args = parser.parse_args()
    if args.siemens_format:
        convert_siemens_dicom_to_nifti(args.data_path, args.output_path, args.patient_id)
    else:
        order_dicom_directory(args.patient_id, args.data_path, args.sorted_data_path)
        convert_siemens_dicom_to_nifti(args.sorted_data_path, args.output_path, args.patient_id)
        shutil.rmtree(args.sorted_data_path)
        print(f"Deleted sorted DICOM directory: {args.sorted_data_path}")