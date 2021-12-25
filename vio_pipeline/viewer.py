import threading
import numpy as np
from vispy import scene, app, gloo
from vispy.visuals import ImageVisual
from vispy.scene import visuals

from queue import Queue
from threading import Thread
from params import * 


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
        canvas = scene.SceneCanvas(keys="interactive",
                                   size=(1280, 720),
                                   show=True)
        canvas.title = "Scaramuzza Hire Me!!"
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
        point_cloud_scatter = visuals.Markers()
        trajectory_view.add(point_cloud_scatter)
        # Add XYZ Axis at the origin as a reference
        axis = visuals.XYZAxis(parent=trajectory_view.scene)

        # Add image view
        img_view = grid.add_view(row=0, col=1)
        img_view.camera = scene.PanZoomCamera(aspect=1)
        img_view.camera.flip = (0, 1, 0)
        # since we don't have the image data at this point,
        # leave data none and crate the image AFTER the view,
        # otherwise vispy does not allow updates on the image data
        img_vispy: ImageVisual = scene.visuals.Image(None,
                                                     interpolation="nearest",
                                                     parent=img_view.scene,
                                                     method="subdivide")

        # Sample visualization of some points
        # Gets removed as soon as a pose comes in
        # TODO: remove this as soon as poses sent to the pose_queue
        trajectory = np.random.normal(size=(1 << 10, 3))
        point_cloud_scatter.set_data(np.array(trajectory),
                                     edge_color=None,
                                     face_color=(1, 1, 1, .5),
                                     size=5)
        trajectory_view.camera = "turntable"

        def update(ev):
            if not self.showing_plot:
                self.timer.stop()

            # get newest pose if available
            if not self.pose_queue.empty():
                while not self.pose_queue.empty():
                    pose = self.pose_queue.get()
                trajectory.append(pose[:3, 3])
                point_cloud_scatter.set_data(np.array(trajectory),
                                             edge_color=None,
                                             face_color=(1, 1, 1, .5),
                                             size=5)
                trajectory_view.camera = "turntable"

            # get newest image if available
            if not self.image_queue.empty():
                while not self.image_queue.empty():
                    img_data = self.image_queue.get()
                img_vispy.set_data(img_data)
                img_view.camera.set_range()

        # start the timer & update at the desired framerate
        self.timer.connect(update)
        self.timer.start(1/TARGET_FRAMERATE)
