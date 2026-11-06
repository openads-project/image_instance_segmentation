#!/usr/bin/env python3
# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

"""Open the demo image viewer maximized."""

import sys

from rqt_gui.main import Main


class MaximizedImageView(Main):
    """Use the standard rqt image viewer with a maximized main window."""

    def create_application(self, argv):
        """Maximize the window after rqt finishes creating it."""
        app = super().create_application(argv)
        from python_qt_binding.QtCore import QTimer
        from python_qt_binding.QtWidgets import QMainWindow

        def maximize_window():
            """Adapt the viewer size to the current screen."""
            for widget in app.topLevelWidgets():
                if isinstance(widget, QMainWindow):
                    widget.showMaximized()

        QTimer.singleShot(0, maximize_window)
        return app


if __name__ == "__main__":
    sys.exit(
        MaximizedImageView().main(
            argv=[sys.argv[0], "--standalone", "rqt_image_view/ImageView", "--args", "/demo/overlay/instance"]
        )
    )
