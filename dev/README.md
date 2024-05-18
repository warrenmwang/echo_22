# Applications of Motion Tracking in Temporally Coherent Echocardiography Video Segmentation

## Project Summary
We examined the usage of the motion tracking head’s outputs from Chen et al.’s [1] CLAS-FV framework which automatically segmented the Left Ventricle from echocardiograms allowing for identification of diastole, systole, and computation of ejection fraction. Understanding that the motion tracking information was currently being utilized to help the model achieve better LV segmentations, we believed we could warp the segmentation frames to analyze regional longitudinal strain and global longitudinal strain due to this metric’s usage of helping to identify LV systolic dysfunction [2]. Previous methods entailed frame by frame speckle-tracking [3] to determine LV myocardium length difference; instead, we will approximate the outermost LV segmentation as our myocardium and warp those points frame by frame to determine length difference. We also looked into the effects of applying dropout layers [4] in increasing the generalizability of the model. For experimental results, we were able to compute global longitudinal strains but not regional longitudinal strains.

## Accomplishments
We were able to compute global longitudinal strain for 1065 test patients; however, we were unable to compute regional longitudinal strain due to our inability to correctly apply our motion tracking information to our segmentation data. Global longitudinal strain was calculated by computing the length difference of the LV myocardium approximation taken from the LV segmentations at the end diastolic and end systolic frames. Furthermore, we also observed negligible difference in the computation of ejection fraction and LV dice scores when using dropout layers of p = 0.10 compared to the original model. 

## Student Experience
This is my first experience working as an undergraduate research assistant and my first HTIP student fellowship. This was a very educational experience that helped me to understand the general expectations and procedures that entail an academic research program. I am very grateful for the HTIP’s generous funding and the mentorship provided to me by Joshua Stough and Christopher Haggerty. 

This experience has helped me to gain a broader scope of possibilities about where I would like my future academic career take place. In terms of skewing my personal opinion of whether I should apply to graduate school, I think I still need more time and experiences before I make more concrete decisions in that regard.

> Future Warren here, it looks like I will try to find my place in industry work and will not be pursuing grad school.

## References
[1] Chen, Y., Zhang, X., Haggerty, C. M., & Stough, J. V. (2022). Fully Automated Multi-heartbeat Echocardiography Video Segmentation and Motion Tracking. SPIE: Medical Imaging. http://eg.bucknell.edu/~jvs008/SPIE22_Chen.html
[2] Reisner, S. A., Lysyansky, P., Agmon, Y., Mutlak, D., Lessick, J., & Friedman, Z. (2004). Global Longitudinal Strain: A novel index of left ventricular systolic function. Journal of the American Society of Echocardiography, 17(6), 630–633. https://doi.org/10.1016/j.echo.2004.02.011 
[3] Mondillo, S., Galderisi, M., Mele, D., Cameli, M., Lomoriello, V. S., Zacà, V., Ballo, P., D'Andrea, A., Muraru, D., Losi, M., Agricola, E., D'Errico, A., Buralli, S., Sciomer, S., Nistri, S., & Badano, L. (2011). Speckle-tracking echocardiography. Journal of Ultrasound in Medicine, 30(1), 71–83. https://doi.org/10.7863/jum.2011.30.1.71
[4] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15(56), 1929–1958. http://jmlr.org/papers/v15/srivastava14a.html 
