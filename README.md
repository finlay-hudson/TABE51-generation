# TABE-51 Dataset Creation

[Finlay GC Hudson](https://finlay-hudson.github.io/), [William AP Smith](https://www-users.york.ac.uk/~waps101/)

University of York

[Paper](https://arxiv.org/pdf/2411.19210) | [Project Page](https://finlay-hudson.github.io/tabe/) | [TABE Codebase](https://github.com/finlay-hudson/TABE) | [TABE-51 Dataset](https://drive.google.com/file/d/1q5u8aqCt2lZUYVb1M9XuveSi7U1EGP7G/view?usp=sharing) 

## Abstract

We present TABE-51 (Track
Anything Behind Everything 51), a video amodal segmentation dataset that provides high-quality ground truth labels
for occluded objects, eliminating the need for human assumptions about hidden pixels. TABE-51 leverages a compositing
approach using real-world video clips: we record
an initial clip of an object without occlusion and then overlay a second clip featuring the occluding object. This
compositing yields realistic video sequences with natural motion and highly accurate ground truth masks, allowing models
to be evaluated on how well they handle occluded objects in authentic scenarios.

<img src="assets/door_walkthrough.gif" alt="Door Walkthrough GIF" width="1400" height="auto">

## Setup

Once this repo is clone locally:

Make venv in preferred format - for instruction, venv will be shown

```
python3 -m venv tabe51-gen_venv
source tabe51-gen_venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git@2b90b9f
```

Download the SAM2 [checkpoint](https://huggingface.co/facebook/sam2-hiera-large).

## Configs 
Within `src/tabe51_gen/configs/video_config.py` The most important configs to set are:
 - `aspect_ratio` _(tuple)_: Defines the aspect ratio of the frames (default: **16:9**). 
 - `video_names` _(tuple)_: List of video file names to process for a single output, as can take in-front and behind frames from different videos to composite together 
 - `data_name` _(str)_: Name you want to set the output data to 
 - `sampled_fps` _(float)_: What do you want to downsample the video to (default: **15.0**). 
 - `data_root_dir` _(Path)_: Root directory for storing processed data. To start has to have a "videos" folder containing the video(s) to work with
 - `in_front_frame_ranges` _(tuple[list[int, int]])_:  Defines the start and stop frames for in-front objects. This can include multiple ranges for different clips or repeated ranges for multiple objects.
 - `behind_frame_range` _(tuple[int, int])_: Defines the frame range for the background object.
 - `sam_checkpoint` _(Path)_: Path to the SAM2 segmentation model checkpoint.

## Running

_Note: This method isn't as automated as would be ideal and is still a bit laborious, work will be done to try and
improve this flow!_

We showcase an example video within `examples`

### 1. Split the video into frames

**Run**: `python split_videos_into_frames.py`  
**Desc**: First stage to split the video into separate frames, saved
to `config.data_root_dir / config.downsampled_frame_dir`

### 2. Line up the frames

**Run**: `python line_up_frames.py`  
**Desc**: manually line up the frames, this will save the aligned frames. The script will open a gui.
to `config.data_root_dir / config.seperated_frames_dir`.  
**Instructions**:

1. **Set Initial Frame Ranges**
    - Define the **background frame range** as `behind_frame_range`.
    - Set the **in-front frame ranges** as `in_front_frame_ranges` (a tuple of tuples of however many in-front frames
      are wanted).

2. **Check Frame Alignment**
    - When the visualizer loads, use the **left** and **right arrow keys** to navigate through frames.
    - The **far-right view** shows an **alpha overlap**, helping visualize alignment.

3. **Adjust Frame Ranges**
    - Close the program, edit the frame ranges, and repeat until the alignment looks correct.

4. **Save the Aligned Frames**
    - In the image viewer, press **'s'** to save the aligned frames.

### 3. Bounding Box Cropping

**Run**: `python auto_bbox_cropper.py`  
**Desc**: We've found that SAM2 performs significantly better when the object of interest is within a cropped area of the full
frame. Attempting to segment directly from the full image can cause details to be lost. To improve accuracy, crop the
bounding boxes of the frames—this will save the cropped frames
to: `config.data_root_dir / config.bbox_clipped_frames_dir`.  
**Instructions**:

It will first start with the behind clip object

1. **Click points on the object**
    - **Left click** for positive points
    - **Middle click** for negative points (not always needed, but can help)

   **Note:** If the object is not yet in the scene, press **'q'** to skip that frame.

2. **Run Segmentation**
    - Once points are selected (**~5 points is a good baseline**, covering all elements of the object), press **'s'** to
      segment the item through the video.

3. **Check Output**
    - If the **bounding box segments are good**, proceed to Step 4.
    - If the **bounding box segments are poor**:
        1. Delete the outputs.
        2. Run `auto_bbox_cropper.py` again, but this time:
            - Press **'m'** once the image loads.
            - **Draw a manual bounding box** around the object of interest.
            - Press **Enter** to create the output.
        3. Either continue using this **manual bounding box** method or return to **Step 1**.
4. Once the **behind clip object** is completed, move to the **in-front object** and repeat the above steps until
   completion.

### 4. Segmenting the Objects
**Run**: `python segment_cropped_frames.py`  
**Desc**: Used to segment the object from these cropped images.  
**Instructions**:

Start with the Behind Clip Object
1. **Click points on the object**
    - Same approach as `auto_bbox_cropper.py`.
    - If the object is not yet in the scene, press **'q'** to skip that frame.

2. **Run Segmentation**
    - Once points are selected, choose one of the following methods:
        - Press **'s'** to segment the object through the entire video.
        - Press **'m'** to segment only the current frame.

   **When to use 'm' instead of 's':**
    - Use **'m'** when only **part of the object is visible**, as propagating incomplete data across the whole video may
      lead to poor segmentation.
    - Use **'s'** if **most or all of the object is visible** to segment the full video.

3. Once the **behind clip object** is completed, move to the **in-front object** and repeat the above steps until
   completion.

`composite_runner.py` - used to composite the segmented objects together into our dataset format

Note: This will run all top level names within the `config.data_root_dir / config.segmented_frames_dir` directory


## BibTeX Citation

If you utilise our code and/or dataset, please consider citing our paper:

```
@article{hudson2024track,
  title={Track Anything Behind Everything: Zero-Shot Amodal Video Object Segmentation},
  author={Hudson, Finlay GC and Smith, William AP},
  journal={arXiv preprint arXiv:2411.19210},
  year={2024}
}
```

## Misc 

We welcome any contributions or collaborations to this work. Also any issues found, we will try and help as best we can in the Issues section :)
