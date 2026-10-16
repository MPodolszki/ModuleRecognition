# ISP setup for the focus aid (`phytec-focus`)

The demo reads the monochrome VM-017 raw through the ISI. `phytec-focus`
reads it through the i.MX8MP ISP (`imx8-phycam-isp` service) at the full
2592x1944 instead, which needs the ISP in sensor mode 2 with the lens the
camera actually carries.

* `run_isp.sh.patch` -- `/opt/imx8-isp/bin/run_isp.sh` resets mode and lens on
  every start; the patch passes `ISP_MODE_CSI1/2` / `ISP_LENS_CSI1/2` on to
  `isp-mode-select-csi1` (original kept as `run_isp.sh.orig` on the board).
* `imx8-phycam-isp-phytec-detect.conf` -- drop-in for
  `/etc/systemd/system/imx8-phycam-isp.service.d/`: mode 2 on both cameras,
  lens AO070.A1 on CSI1 (CSI2 keeps the default AO062; its XML is set to
  denoise 1 / sharpen 4 on the board, original kept as `.orig`).
* `VM-017-BW_AO070.A1_2592x1944.xml` -- tuning file as adjusted on the board
  (`/opt/imx8-isp/bin/xml/`), filter denoise 1 / sharpen 8.

After changing the XML: `systemctl restart imx8-phycam-isp`, then restart
`phytec-focus`.
