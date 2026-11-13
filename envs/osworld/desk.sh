# Starts the KDE desktop in an OSWorld container without s6 /init, the same on every system, so it can be forked.
# Entry point: the container command, `bash -c "$(cat desk.sh)"` (envs/osworld/env.py run_args). No arguments.
# What it sets up:
#   - Xvfb :1 and the window manager as user abc via setpriv (runuser lingers as root and breaks repeated CRIU dumps)
#   - AT-SPI accessibility on; LibreOffice asked for GTK3 (SAL_USE_VCLPLUGIN; on qt5 Calc's tree holds only row 1),
#     which it gets only with the plugin --lo-gtk3 1 copies in, else it stays on qt5
#   - a LibreOffice profile with no first-run bar, tips, keep-format dialog (scorers save with Ctrl+S) or recovery
#   - VLC as on the OSWorld VM: web interface on :8080 (password "password", read by the VLC scorer), X11 output
#   - HOME=/home/user, so apps save paths as on the OSWorld VM (scorers compare path strings)
export DISPLAY=:1 QT_ACCESSIBILITY=1 QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1 NO_AT_BRIDGE=0
H=/config; mkdir -p $H/.XDG $H/Desktop /tmp/.X11-unix; chmod 1777 /tmp/.X11-unix
LO=$H/.config/libreoffice/4/user; mkdir -p $LO
cat > $LO/registrymodifications.xcu <<'XCU'
<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="ShowTipOfTheDay" oor:op="fuse"><value>false</value></prop></item>
<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="FirstRun" oor:op="fuse"><value>false</value></prop></item>
<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="LastTipOfTheDayShown" oor:op="fuse"><value>99999</value></prop></item>
<item oor:path="/org.openoffice.Setup/Product"><prop oor:name="ooSetupLastVersion" oor:op="fuse"><value>24.2</value></prop></item>
<item oor:path="/org.openoffice.Office.Common/Save/Document"><prop oor:name="WarnAlienFormat" oor:op="fuse"><value>false</value></prop></item>
<item oor:path="/org.openoffice.Office.Recovery/RecoveryInfo"><prop oor:name="Enabled" oor:op="fuse"><value>false</value></prop></item>
</oor:items>
XCU
mkdir -p $H/.config/vlc
printf '[qt]\nqt-privacy-ask=0\n[core]\nextraintf=http\nvout=xcb_x11\nmetadata-network-access=0\n[lua]\nhttp-password=password\n[avcodec]\navcodec-hw=none\n' > $H/.config/vlc/vlcrc
if [ "$(id -u)" = 0 ]; then chown -R abc:abc $H; AS="setpriv --reuid=abc --regid=abc --init-groups --"; else AS=""; fi
Xvfb :1 -screen 0 1920x1080x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
for i in $(seq 100); do xdpyinfo >/dev/null 2>&1 && break; sleep 0.1; done
$AS env HOME=/home/user DISPLAY=:1 XDG_RUNTIME_DIR=$H/.XDG QT_ACCESSIBILITY=1 QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1 NO_AT_BRIDGE=0 SAL_USE_VCLPLUGIN=gtk3 bash /defaults/startwm.sh >/tmp/wm.log 2>&1 &
wait
