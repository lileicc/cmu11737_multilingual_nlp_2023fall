# How to get GCP credits

GCP provides for $300 free credits for new users. A V100 is priced at roughly $2.5/hr. If you set this up now, you can use it for the entire semester (for other assignments and your final project).

**Important Note**: Please stop your instance as soon as you are done using it. You will be charged for the entire time that it is running and we are **not** responsible if this explodes your usage.

## Step 1: Create a billing account and activate it
Refer to [this video](https://www.youtube.com/watch?v=iZgD6p0slTU&ab_channel=Techdox) on how this can be done.

## Step 2: Increase the quota limit for GPUs from 0 to 1
Refer to [this video](https://www.youtube.com/watch?v=-jsOAv2zsXU&ab_channel=Furcifer) (only the first 5-6 mins) on how this can be done.

## Step 3: Create a new VM
Create a new instance with the following specifications:

- GPU: 1 x NVIDIA V100
- Machine Type: n1-standard-4 15GB
- Image: Deep Learning VM with CUDA 11.8, M110, Debian 11, Python 3.10. With CUDA 11.8 preinstalled.
- Size: 250GB (can be smaller, depends on you)

## Step 4: SSH from your local terminal
Refer to [this video](https://www.youtube.com/watch?v=2ibBF9YqveY&ab_channel=Koderstack) on how you can do this.

## (Optional): Connect your VSCode terminal to the GCP machine using the remote SSH extension for easier development

Similar to how you remote ssh into any VM using VSCode. Please reach out to the TAs if y'all have questions.

