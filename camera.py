import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import time
# Rq : realsense display images in rgb
import json

class RealCamera:
    def __init__(self):
        self.frame = None
        self.pipelineStarted = False
        self.depth = None
        self.rgb = None
        self.depth_scale = None
        self.align = None
        self.color_image = None
        self.depth_image = None

    def start_pipe(self, align=True, usb3=True):
        if not self.pipelineStarted:
            if align:
                print('Etablissement de la connection caméra')
                # Create a config and configure the pipeline to stream
                #  different resolutions of color and depth streams
                self.pipeline = rs.pipeline()

                # Create a config and configure the pipeline to stream
                #  different resolutions of color and depth streams
                config = rs.config()
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

                # Start streaming
                self.profile = self.pipeline.start(config)

                align_to = rs.stream.color
                self.align = rs.align(align_to)

                time.sleep(1)

                # self.pipeline = rs.pipeline()
                # config = rs.config()
                #
                # if usb3:
                #     config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
                #     config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                #
                # else:
                #     self.profile = config.resolve(self.pipeline)  # does not start streaming
                #
                # self.profile = self.pipeline.start(config)
                # self.pipelineStarted = True
                # # Align the two streams
                # align_to = rs.stream.color
                # self.align = rs.align(align_to)
                self.pipelineStarted = True
                # Get depth scale
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()

                # Création des filtres
                self.hole_filling = rs.hole_filling_filter()
                self.temporal_filter = rs.temporal_filter()
                self.spatial_filter = rs.spatial_filter()
                self.depth_to_disparity = rs.disparity_transform(False)
                # Get Intrinsic parameters
                self.get_intrinsic()
                print('Caméra Ouverte')

    def stop_pipe(self):
        if self.pipelineStarted:
            print('Camera Fermée')
            self.pipeline.stop()
            self.pipelineStarted = False

    def show(self):
        temp_depth = self.depth_image
        # mini = np.min(self.depth_image)
        # mean = np.mean(self.depth_image)
        # median = np.median(self.depth_image)
        # temp_depth[temp_depth<median] = mini
        temp_depth[temp_depth==0.] = np.max(temp_depth)
        print(np.min(temp_depth), np.mean(temp_depth))
        plt.subplot(1, 3, 1)
        plt.imshow(temp_depth)
        plt.subplot(1, 3, 2)
        plt.imshow(self.color_image)
        plt.subplot(1, 3, 3)
        plt.imshow(self.mask_color)
        plt.show()

    def get_frame(self):
        # Get frameset of color and depth
        self.frame = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(self.frame)

        # Get aligned frames
        frames = []
        for x in range(40):
            temp_filtered = self.temporal_filter.process(aligned_frames.get_depth_frame())
        # aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image

        color_frame = aligned_frames.get_color_frame()
        # Processing
        self.depth_image = self.hole_filling.process(temp_filtered)

        self.depth_image = self.spatial_filter.process(self.depth_image)
        self.depth_image = self.depth_to_disparity.process(self.depth_image)

        self.depth_image = np.asanyarray(self.depth_image.get_data())*self.depth_scale
        self.depth_image[self.depth_image > 6] = 0.

        self.depth_mask = np.copy(self.depth_image)
        self.depth_mask[self.depth_mask > 0.] = 1
        self.depth_mask = self.depth_mask.astype('uint8')

        self.color_image = np.asanyarray(color_frame.get_data())

        self.mask_color = np.copy(self.color_image)
        self.mask_color[:, :, 0] *= self.depth_mask
        self.mask_color[:, :, 1] *= self.depth_mask
        self.mask_color[:, :, 2] *= self.depth_mask

        # Mettre ces images au carré
        self.depth_image, self.color_image = self.depth_image[:, 160:], self.color_image[:, 160:]

        return self.depth_image, self.color_image

    def transform_3D(self, u, v, image=None, param=None):
        '''
        input : Depthmap
        :return: return the view in 3D space
        '''
        if image is None:
            d = self.depth_image[v, u]
        else:
            print(image.shape)
            d = image[v, u]
        if param is None:
            depth_scale = self.depth_scale
            fx, fy, Cx, Cy = self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy
        else:
            fx, fy, Cx, Cy, depth_scale = param
        z = d / depth_scale
        # print('Cy : {} ; Cx : {} ; image dimension {}'.format(Cy, Cx, self.depth_image.shape))
        y = (u - Cy) * z / fy
        x = (v - Cx) * z / fx
        P = np.array([x, y, z])
        return P

    def transform_image_to_3D(self, image=None):
        '''

        :return:
        '''
        if image is None:
            depth_img = self.depth_image

        else:
            depth_img = image

        depth_scale = self.depth_scale
        fx, fy, Cx, Cy = self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy
        list3d = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), np.float32)
        for (u, v), d in np.ndenumerate(depth_img):
            z = d / depth_scale
            x = (u - Cy) * z / fy
            y = (v - Cx) * z / fx
            list3d[u, v, :] = np.array([x, y, z])
        return list3d

    def erase_background(self):
        self.np.mean()
        pass

    def store(self):
        pass

    def get_intrinsic(self):
        # pipeline = rs.pipeline()
        # cfg = pipeline.start()  # Start pipeline and get the configuration it found
        # profile = cfg.get_stream(rs.stream.depth)  # Fetch stream profile for depth stream
        # intr = profile.as_video_stream_profile().get_intrinsics()  # Downcast to video_stream_profile and fetch intrinsics
        profile = self.profile.get_stream(rs.stream.depth)  # Fetch stream profile for depth stream
        self.intr = profile.as_video_stream_profile().get_intrinsics()  # Downcast to video_stream_profile and fetch intrinsics


if __name__=='__main__':
    Cam = RealCamera()
    Cam.start_pipe(usb3=True)
    Cam.get_intrinsic()
    while True:
        Cam.get_frame()
        Cam.show()
        stop = input('Stop ?')
        if stop=='0':
            break

    Cam.stop_pipe()