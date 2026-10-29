# i.MX95 NEO ISP: sharpening for the AR0521 phyCAMs

The NXP libcamera IPA programs only AGC, AWB, BLC, CCM, DRC, gamma, LSC
(UG10215 table 1); edge enhancement and noise filters stay at the kernel
defaults (EE off, BNR on). `0009-...patch` adds an IPA algorithm `Sharpen`
that sets them from the tuning file, and gives `ar0521.yaml` the black
level (pedestal 168 @ 12 bit) and EE gain 4 / coring 2048 / clip 262144.

In the BSP: `meta-phyVIP-imx95/recipes-multimedia/libcamera/` (bbappend on
libcamera 0.7.1.imx, after the meta-ampliphy phyCAM patches 0001-0008).

On a board by hand: `/usr/lib/libcamera/ipa/ipa_nxp_neo.so{,.sign}` from the
build, tuning in `/usr/share/libcamera/ipa/nxp/neo/ar0521.yaml`; the
`Sharpness` control (libcamerasrc `sharpness=`) scales the EE gain, 1.0 = file.

Measured (Siemens star, blur circle at half contrast, screen px, mono / colour):
original 20/20, + black level 20/23, + EE 4/2048 13/14, + BNR off 13/13.
