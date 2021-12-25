import threading
import numpy as np
from vispy import scene, app, gloo
from vispy.visuals import ImageVisual

from queue import Queue
from threading import Thread


class Viewer(object):
    def __init__(self):
        self.image_queue = Queue()
        self.pose_queue = Queue()

        # self.thread = Thread(target=self.start_vis)

        # self.thread.start()
        self.timer = app.Timer()

        self.showing_plot = False

    def update_pose(self, pose):
        if pose is None:
            return
        self.pose_queue.put(pose.matrix())

    def update_image(self, image):
        if image is None:
            return
        self.image_queue.put(image)

    def start_vis(self):
        # create canvas to draw on, Window size is fixed to (1280, 720)
        canvas = scene.SceneCanvas(keys="interactive", size=(1280, 720), show=True)
        canvas.title="Scaramuzza Hire Me!!"
        self.showing_plot = True

        # Attach key handler to close window
        @canvas.events.key_press.connect
        def on_key_press(event):
            print(f"Canvas event trigged, pressed key: {event.key}")
            if event.key == "X":
                self.showing_plot = False

        # top-level widget, holding the three ViewBoxes,
        # which are automaticall resized with the grid
        grid = canvas.central_widget.add_grid()


        # Add trajectory view
        trajectory_view = grid.add_view(row=0, col=0)

        # Add image view
        img_view = grid.add_view(row=0, col=1)
        interpolation = "nearest"

        img_view.camera = scene.PanZoomCamera(aspect=1)
        img_view.camera.flip = (0, 1, 0)

        img_vispy: ImageVisual = scene.visuals.Image(None,
                                        interpolation="nearest",
                                        parent=img_view.scene,
                                        method="subdivide")
        trajectory = []
        img_data = None

        def update(ev):
            if not self.showing_plot:
                self.timer.stop()
            # get newest pose if available
            if not self.pose_queue.empty():
                while not self.pose_queue.empty():
                    pose = self.pose_queue.get()
                trajectory.append(pose[:3, 3])

            # get newest image if available
            if not self.image_queue.empty():
                while not self.image_queue.empty():
                    img_data = self.image_queue.get()
                img_vispy.set_data(img_data)
                img_view.camera.set_range()


        self.timer.connect(update)
        self.timer.start(0)


