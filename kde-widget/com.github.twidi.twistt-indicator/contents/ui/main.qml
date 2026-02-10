import QtQuick
import QtQuick.Layouts
import org.kde.plasma.plasmoid
import org.kde.plasma.core as PlasmaCore
import org.kde.plasma.components as PlasmaComponents
import org.kde.kirigami as Kirigami
import org.kde.plasma.plasma5support as Plasma5Support

/**
 * Twistt Indicator — KDE Plasma panel widget.
 *
 * Displays a microphone icon whose colour reflects the current Twistt state:
 *   idle           → cyan    (#40B8C8)   — static
 *   recording      → orange  (#E05030)   — pulsing to #802818
 *   speech_active  → green   (#30C050)   — pulsing to #1A7030
 *   post_treatment → violet  (#B040D0)   — pulsing to #602070
 *
 * The state is read from ~/.local/share/twistt/plasma-widget-state every 300 ms
 * via the Plasma5Support executable DataSource.
 */
PlasmoidItem {
    id: root

    // ── State management ──────────────────────────────────────────
    property string currentState: "idle"
    property bool appRunning: false

    readonly property var stateColors: ({
        "idle":           { bright: "#40B8C8", dim: "#40B8C8" },
        "recording":      { bright: "#E05030", dim: "#802818" },
        "speech_active":  { bright: "#30C050", dim: "#1A7030" },
        "post_treatment": { bright: "#B040D0", dim: "#602070" }
    })

    readonly property bool isPulsing: currentState !== "idle" && appRunning
    readonly property color brightColor: stateColors[currentState] ? stateColors[currentState].bright : "#40B8C8"
    readonly property color dimColor: stateColors[currentState] ? stateColors[currentState].dim : "#40B8C8"

    // Animated colour that pulses between bright and dim for active states.
    property color animatedColor: brightColor

    // ── Tooltip ───────────────────────────────────────────────────
    toolTipMainText: "Twistt"
    toolTipSubText: {
        if (!appRunning)
            return "Not running"
        switch (currentState) {
            case "recording":      return "Listening..."
            case "speech_active":  return "Transcribing..."
            case "post_treatment": return "Post-processing..."
            default:               return "Idle"
        }
    }
    toolTipTextFormat: Text.PlainText

    // ── Pulsing animation ─────────────────────────────────────────
    SequentialAnimation {
        id: pulseAnimation
        running: root.isPulsing
        loops: Animation.Infinite

        ColorAnimation {
            target: root
            property: "animatedColor"
            from: root.brightColor
            to: root.dimColor
            duration: 300
        }
        ColorAnimation {
            target: root
            property: "animatedColor"
            from: root.dimColor
            to: root.brightColor
            duration: 300
        }
    }

    // When pulsing stops (back to idle), snap to bright colour.
    onIsPulsingChanged: {
        if (!isPulsing) {
            pulseAnimation.stop()
            animatedColor = brightColor
        }
    }

    // When state changes while pulsing, restart animation with new colours.
    onBrightColorChanged: {
        if (isPulsing) {
            pulseAnimation.restart()
        } else {
            animatedColor = brightColor
        }
    }

    // ── State file reader ─────────────────────────────────────────
    // The shell expands $XDG_DATA_HOME (or falls back to ~/.local/share).
    readonly property string stateFileCmd: "cat \"${XDG_DATA_HOME:-$HOME/.local/share}/twistt/plasma-widget-state\" 2>/dev/null"

    Plasma5Support.DataSource {
        id: stateReader
        engine: "executable"
        // Connect the command as a persistent source; `interval` controls re-execution.
        connectedSources: [root.stateFileCmd]
        interval: 300

        onNewData: function(sourceName, data) {
            var stdout = data["stdout"].trim()

            if (stdout === "") {
                // State file does not exist or is empty → app is not running.
                root.appRunning = false
                root.currentState = "idle"
                return
            }

            // Validate that the state is one we know about.
            if (root.stateColors.hasOwnProperty(stdout)) {
                root.appRunning = true
                root.currentState = stdout
            } else {
                // Unknown content → treat as not running.
                root.appRunning = false
                root.currentState = "idle"
            }
        }
    }

    // ── Compact representation (panel icon) ───────────────────────
    compactRepresentation: Item {
        id: compactRoot

        readonly property bool isVertical: Plasmoid.formFactor === PlasmaCore.Types.Vertical

        // Horizontal panel: height is fixed by the panel → preferred width = height → square.
        // Vertical panel:   width  is fixed by the panel → preferred height = width → square.
        Layout.preferredWidth:  isVertical ? -1 : height
        Layout.preferredHeight: isVertical ? width : -1
        Layout.minimumWidth:  Kirigami.Units.iconSizes.small
        Layout.minimumHeight: Kirigami.Units.iconSizes.small

        Canvas {
            id: micCanvas
            anchors.fill: parent
            anchors.margins: Math.round(Math.min(parent.width, parent.height) * 0.08)

            // Repaint whenever the animated colour changes.
            property color drawColor: root.appRunning ? root.animatedColor : "#888888"
            onDrawColorChanged: requestPaint()

            // Also repaint on resize.
            onWidthChanged: requestPaint()
            onHeightChanged: requestPaint()

            onPaint: {
                var ctx = getContext("2d")
                var sz = Math.min(width, height)
                if (sz <= 0) return

                ctx.reset()
                ctx.clearRect(0, 0, width, height)

                // Centre the drawing if the canvas is not square.
                var ox = (width - sz) / 2
                var oy = (height - sz) / 2
                ctx.translate(ox, oy)

                var color = drawColor.toString()

                // ── Microphone head (capsule shape) ──
                var micW = sz * 0.38
                var micH = sz * 0.48
                var micX = (sz - micW) / 2
                var micY = sz * 0.04
                var radius = micW / 2

                ctx.fillStyle = color
                ctx.beginPath()

                // Top semicircle
                ctx.arc(micX + radius, micY + radius, radius, Math.PI, 0, false)

                // Right side down
                ctx.lineTo(micX + micW, micY + micH - radius)

                // Bottom semicircle
                ctx.arc(micX + radius, micY + micH - radius, radius, 0, Math.PI, false)

                // Left side up (closed automatically)
                ctx.closePath()
                ctx.fill()

                // ── Stand arc (U-shape) ──
                var lineW = Math.max(1.5, sz * 0.06)
                var arcMargin = sz * 0.20
                var arcTop = micY + micH - sz * 0.06
                var arcBottom = arcTop + sz * 0.30
                var arcCenterX = sz / 2
                var arcCenterY = (arcTop + arcBottom) / 2
                var arcRadiusX = (sz - 2 * arcMargin) / 2
                var arcRadiusY = (arcBottom - arcTop) / 2

                ctx.strokeStyle = color
                ctx.lineWidth = lineW
                ctx.lineCap = "round"
                ctx.beginPath()

                // Draw the U-shape as an arc from 0 to PI (bottom half of ellipse).
                // Canvas arc draws circles, so we approximate the elliptical arc
                // with a manual ellipse path.
                var steps = 32
                for (var i = 0; i <= steps; i++) {
                    var angle = Math.PI * i / steps  // 0 to PI
                    var px = arcCenterX - arcRadiusX * Math.cos(angle)
                    var py = arcCenterY + arcRadiusY * Math.sin(angle)
                    if (i === 0)
                        ctx.moveTo(px, py)
                    else
                        ctx.lineTo(px, py)
                }
                ctx.stroke()

                // ── Vertical stem ──
                var stemX = sz / 2
                var stemTop = arcCenterY + arcRadiusY
                var stemBottom = sz * 0.88

                ctx.beginPath()
                ctx.moveTo(stemX, stemTop)
                ctx.lineTo(stemX, stemBottom)
                ctx.stroke()

                // ── Base ──
                var baseW = sz * 0.32
                ctx.beginPath()
                ctx.moveTo(stemX - baseW / 2, stemBottom)
                ctx.lineTo(stemX + baseW / 2, stemBottom)
                ctx.stroke()
            }
        }
    }

    // ── Full representation (popup when clicked) ──────────────────
    // Minimal info display since this is a read-only status indicator.
    fullRepresentation: Item {
        Layout.minimumWidth: 200
        Layout.minimumHeight: 80
        Layout.preferredWidth: 240
        Layout.preferredHeight: 100

        Column {
            anchors.centerIn: parent
            spacing: 8

            PlasmaComponents.Label {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "Twistt"
                font.bold: true
                font.pointSize: 14
                color: root.animatedColor
            }

            PlasmaComponents.Label {
                anchors.horizontalCenter: parent.horizontalCenter
                text: {
                    if (!root.appRunning)
                        return "Application not running"
                    switch (root.currentState) {
                        case "recording":      return "Listening..."
                        case "speech_active":  return "Transcribing..."
                        case "post_treatment": return "Post-processing..."
                        default:               return "Idle — waiting for input"
                    }
                }
                font.pointSize: 10
                color: root.appRunning ? root.brightColor : Kirigami.Theme.disabledTextColor
            }
        }
    }

    // Prefer the compact view in the panel.
    preferredRepresentation: compactRepresentation
}
