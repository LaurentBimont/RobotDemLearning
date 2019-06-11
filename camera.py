import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# Rq : realsense display images in rgb


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
                # Create a config and configure the pipeline to stream
                #  different resolutions of color and depth streams
                config = rs.config()
                # if usb3:
                #     print(2)
                #     config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
                #     config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

                self.pipeline = rs.pipeline()
                self.config = rs.config()
                self.profile = config.resolve(self.pipeline)  # does not start streaming

                self.profile = self.pipeline.start(config)
                self.pipelineStarted = True
                # Align the two streams
                align_to = rs.stream.color
                self.align = rs.align(align_to)

                # Get depth scale
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()

                # Get Intrinsic parameters
                self.get_intrinsic()

    def stop_pipe(self):
        if self.pipelineStarted:
            self.pipeline.stop()
            self.pipelineStarted = False

    def show(self):
        plt.subplot(1, 3, 1)
        plt.imshow(self.depth_image)
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
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        self.depth_image = np.asanyarray(aligned_depth_frame.get_data())*self.depth_scale
        self.depth_image[self.depth_image > 6] = 0.
        self.depth_mask = np.copy(self.depth_image)
        self.depth_mask[self.depth_mask > 0.] = 1
        self.depth_mask = self.depth_mask.astype('uint8')

        self.color_image = np.asanyarray(color_frame.get_data())

        self.mask_color = np.copy(self.color_image)
        print(self.mask_color.dtype)
        self.mask_color[:, :, 0] *= self.depth_mask
        self.mask_color[:, :, 1] *= self.depth_mask
        self.mask_color[:, :, 2] *= self.depth_mask

        return self.depth_image, self.color_image

    def transform_3D(self, u, v, image=None, param=None):
        '''
        input : Depthmap
        :return: return the view in 3D space
        '''
        if image is None:
            d = self.depth_image[v, u]
        else:
            d = image[v, u]
        if param is None:
            depth_scale = self.depth_scale
            fx, fy, Cx, Cy = self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy
        else:
            depth_scale = self.depth_scale
            fx, fy, Cx, Cy = param[0], param[1],self.intr.ppx, self.intr.ppy
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
            print('ici')
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
    Cam.get_frame()
    Cam.stop_pipe()
    Cam.show()
