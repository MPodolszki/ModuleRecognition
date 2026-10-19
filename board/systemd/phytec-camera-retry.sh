#!/bin/sh
# FPD-Link deserializers (DS90UB954/960) that failed to probe at boot --
# "Failed to lock all RX ports" when the camera link came up late -- get
# bound again, a few tries, before the ISP service looks for cameras.
for try in 1 2 3 4 5; do
    missing=0
    for dev in /sys/bus/i2c/devices/*; do
        [ "$(cat "$dev/name" 2>/dev/null)" = ds90ub960 ] || continue
        [ -e "$dev/driver" ] && continue
        missing=1
        echo "binding $(basename "$dev") (try $try)"
        echo "$(basename "$dev")" > /sys/bus/i2c/drivers/ds90ub960/bind 2>/dev/null
    done
    [ $missing = 0 ] && exit 0
    sleep 2
done
exit 0
