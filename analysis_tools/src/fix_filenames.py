import os

def process_results_and_videos(results_dir, video_dir):
    """
    Process the results folder and video directory.

    Args:
        results_dir (str): Full path to the Results folder.
        video_dir (str): Full path to the video directory.

    Returns:
        vids (list): List of video file names in the video directory.
        updated_dirs (list): List of updated subdirectory names in the Results folder.
    """
    # Get a list of all video file names in the video directory
    vids = [file for file in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, file))]

    # Initialize a list to store updated subdirectory names
    updated_dirs = []

    # Iterate through subdirectories in the results directory
    for subdir in os.listdir(results_dir):
        full_path = os.path.join(results_dir, subdir)

        # Check if it is a directory and matches the "runX" pattern
        if os.path.isdir(full_path) and subdir.startswith("run") and subdir[3:].isdigit():
            # Extract the numerical portion and give it a leading 0
            run_number = int(subdir[3:])
            new_subdir_name = f"run{run_number:02d}"

            # Rename the directory if necessary
            if subdir != new_subdir_name:
                new_full_path = os.path.join(results_dir, new_subdir_name)
                os.rename(full_path, new_full_path)

            updated_dirs.append(new_subdir_name)

    return vids, updated_dirs

# Example usage
results_path = "/path/to/Results"  # Replace with the actual path
video_path = "/path/to/Videos"    # Replace with the actual path
vids, updated_dirs = process_results_and_videos(results_path, video_path)

print("Video Files:", vids)
print("Updated Directories:", updated_dirs)