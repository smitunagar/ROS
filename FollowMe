import copy
import math
import os
import pickle
import sys
import time

import cv2
import numpy as np
import tensorflow._api.v2.compat
from PIL import Image as pil_img
from cv_bridge import CvBridge
from matplotlib.backend_bases import MouseEvent
from numpy import ndarray

import TransformerBase
import color_detector
import followMeFunctions
from LogLevel import LogLevel
from FollowMeImage import FollowMeImage
from pose import FollowMePose
from short_term_classifier import ShortTermClassifier, KalmanFilter
from tf_pose_estimation4.tf_pose.estimator import TfPoseEstimator
from tf_pose_estimation4.tf_pose.networks import get_graph_path, model_wh

import tensorflow.compat.v1

REID_LIBS = os.path.join(os.environ['RTL_ROOT'], 'libs', "facer2vm_watch_list")
REID_FRAME_SKIP = 180
REID_PATCH_PADDING_FACTOR = 1.05
REID_TRESHOLD = .4597426985
REID_MAX_TRIES = int(math.pi * math.e ** math.pi) * 4


class ClassName(object):
    pass

tensorflow.compat.v1.disable_eager_execution()

t = ClassName()  # just an empty object to dynamically create attributes in


def tick(): return time.time()


class FollowMe(object):
    depth_image: ndarray

    # image: ndarray

    def __init__(self, goalPublisher, selectedFlagPublisher, hubPublisher, log, args,
                 transformer: TransformerBase.TransformerBase):
        self.log = log
        self.goalPublisher = goalPublisher
        self.hubPublisher = hubPublisher
        self.selectedFlagPublisher = selectedFlagPublisher
        self.image = self.depth_image = None
        args.camera = "camera_front"
        self.target_frame_pose = "base_link"  # "map"
        self.source_frame_camera = "%s_depth_optical_frame" % args.camera
        self.bridge = CvBridge()
        self.depth_scale = 1e-3
        #######################
        self.humans = None
        self.fps_time = 0
        self.xy = (-1, -1)
        self.stc = ShortTermClassifier()
        self.kalman = KalmanFilter()
        self.timer = 0
        self.run_flag = False
        self.first_signal = False  # signals if closest should be selected
        self.frames_kalman_color_lost = 0
        self.last_published_pose = None
        self.reid_tries = 0
        self.reid_frames_tracked = 0
        self._pose: FollowMePose = None
        self._distance: float = 0
        #######################

        self.threaded = []

        self.cd = color_detector.ColorDetector()
        if args.transform or True:
            log("setup transform point 2d -> 3d", LogLevel.WARNING)
            # sys.path.insert(2, os.path.expanduser("~/.local/lib/python2.7/site-packages"))
            self.transformer = transformer

            def transform_setup(s: FollowMe):
                try:
                    if not s.transformer.setup(s.depth_image, s.body_parts):
                        return
                    s._distance = s.transformer.distance(s.depth_scale)  # is in meter
                    if s._distance is None:
                        return
                    ps = s.transformer.transform_uv2pose(s._distance, s.target_frame_pose, s.source_frame_camera)
                    if ps is None:
                        return
                    s._x = ps.position.x
                    s._y = ps.position.y
                    s._z = ps.position.z
                    s._pose = copy.deepcopy(ps)
                except Exception as e:
                    s.log('Failed transform 2d -> 3d %r' % e, LogLevel.ERROR)

            self.threaded.append(transform_setup)
        # /transform

        if args.waving:
            self.WavingSetup()
        else:
            self.waving = lambda _: None
        # /waving

        if args.surrey:
            self.SurreySetup()
        # /surrey

        if args.reid:
            self.ReidSetup()

        else:
            self.add_new_user_to_watch_list_run_time_reid = lambda _: None
            self.add_sample_existing_user_in_watch_list_run_time_reid = lambda _: None
            self.process_frame_reid = lambda _: None
        # /reid

        # BEGIN tf-pose
        self.log("setup open pose", LogLevel.WARNING)
        # https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
        config = tensorflow.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        sys.path.insert(1, os.path.join(os.environ['RTL_ROOT'], 'libs', 'tf-pose-estimation'))

        w, h = model_wh(args.resize)
        if w > 0 and h > 0:
            tSize = (w, h)
        else:
            tSize = (432, 368)
        self.tfpe = TfPoseEstimator(get_graph_path(args.model), target_size=tSize, tf_config=config)
        # remember for later:
        self.r2default = (w > 0 and h > 0)
        self.resize_out_ratio = args.resize_out_ratio
        # END tf-pose

        self.log("initialized!", LogLevel.WARNING)

    def ReidSetup(self):
        self.log("setup Surrey reid", LogLevel.WARNING)
        sys.path.insert(1, REID_LIBS)
        import reid_matching_api
        self.reidHuman = None
        self.reid = reid_matching_api.reid_matching_api(trshld=REID_TRESHOLD)

        def bound(low, high, value):
            return max(low, min(high, value))

        def cutBody(self, call, *args, **kwargs):
            try:
                image_w = self.image.shape[1]
                image_h = self.image.shape[0]

                boxDict = self.reidHuman.get_upper_body_box(image_w, image_h)
                if boxDict is None: raise AssertionError("no body box")
                boxDict["w"] *= REID_PATCH_PADDING_FACTOR
                boxDict["h"] *= REID_PATCH_PADDING_FACTOR
                boxXmin = (boxDict["x"] - boxDict["w"] / 2)
                boxXmax = (boxDict["x"] + boxDict["w"] / 2)
                boxYmin = (boxDict["y"] - boxDict["h"] / 2)
                boxYmax = (boxDict["y"] + boxDict["h"] / 2)

                boxXmin = int(bound(0, image_w, boxXmin))
                boxYmin = int(bound(0, image_h, boxYmin))
                boxXmax = int(bound(0, image_w, boxXmax))
                boxYmax = int(bound(0, image_h, boxYmax))

                patch = self.image[boxYmin:boxYmax, boxXmin:boxXmax, :]

                return call(pil_img.fromarray(patch), *args, **kwargs)
            except Exception as e:
                self.log('reid failed %r' % e.message, LogLevel.ERROR)

        self.add_new_user_to_watch_list_run_time_reid = lambda: cutBody(self,
                                                                        self.reid.add_new_user_to_watch_list_run_time_reid,
                                                                        person_name="operator")
        self.add_sample_existing_user_in_watch_list_run_time_reid = lambda: cutBody(self,
                                                                                    self.reid.add_sample_existing_user_in_watch_list_run_time_reid,
                                                                                    person_name="operator")
        self.process_frame_reid = lambda: cutBody(self, self.reid.process_frame_reid)  # , display_mode=0, test=False)

    def WavingSetup(self):
        self.log("setup waving", LogLevel.WARNING)
        import who_is_waving
        wavernator = who_is_waving.Who_is_waving()

        def waving(self):
            try:
                wavernator.handle_humans(self.humans)
            except Exception as e:
                self.log('Failed waving %r' % e, LogLevel.ERROR)

        self.waving = waving

    def SurreySetup(self):
        self.log("setup Surrey face id and age estimation", LogLevel.WARNING)
        sys.path.insert(1, os.path.join(os.environ['RTL_ROOT'], 'libs', "facer2vm_watch_list"))
        import face_matching_age_estimation_api as fmea
        sys.path.insert(1, os.path.join(os.environ['RTL_ROOT'], 'libs',
                                        "caffeSurrey-protobuf-3.8.0/distribute/python"))
        import caffe
        from rtl_human_descriptor import headExtractionFunction as hef
        wl_api = fmea.face_matching_age_estimation_api()

        def surrey(self):
            # Surrey face id, age
            try:
                """
      Caffe fails to use GPU in a new thread #4178
      https://github.com/BVLC/caffe/issues/4178#issuecomment-221386875
      ajtulloch commented on May 24, 2016
      the Caffe::mode_ variable that controls this is thread-local, so ensure you're calling caffe.set_mode_gpu() in each thread before running any Caffe functions. That should solve your issue.
      """
                caffe.set_device(0)
                caffe.set_mode_gpu()

                img = hef.headExtraction(self.image, self.body_parts)
                shp = img.shape[:2]
                if min(shp) < self.args.headsizemin: raise AssertionError("face patch to small %dx%d" % shp)

                # id_name, id_age = wl_api.process_frame(frame, True, 0)
                self._id_name, self._id_age = wl_api.process_frame(img, True, 0)
                self._re_id = self._id_name  # todo: Surrey re id
            except Exception as e:
                self.log('Failed Surrey face id, age %r' % e.message, LogLevel.ERROR)

        self.threaded.append(surrey)

    def processDepth(self, data: ndarray):
        self.depth_scale = 1e-3
        self.depth_image = data

    # TODO make cv2 overload
    def cb_depth(self, data: FollowMeImage):
        print("cb_depth")
        self.depth_scale = {"16UC1": 1e-3, "32FC1": 1}.get(data.encoding, None)
        print(self.depth_scale)
        print(data)
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except Exception as e:
            print(e)
        print(self.depth_image)
    def pipeline(self, image: FollowMeImage):
        # define pipeline
        print("pipeline")
        print(image)
        global t
        t.start = tick()
        t.postprc = 0

        self.image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        humans = self.tfpe.inference(self.image, resize_to_default=self.r2default, upsample_size=self.resize_out_ratio)
        self.hubPublisher(pickle.dumps(humans))
        t.infer = tick() - t.start

        # colors
        self.cd.search_colors(self.image, humans)

        # we need all that extra attributes shit here in humans
        for h in humans:
            h.selected = False
            h.distance = 0
            h.angle = 0
            h.selection_score = 0
            self.body_parts = h.body_parts
            self.threaded[0](self)
            h.pose = self._pose

        self.humans = humans
        self.waving(self)

        t.tot = tick() - t.start
        self.log("tf_pose humans %2d, infer %.4f, postprc %.4f, tot %.4f, hz %.1f" % (
            len(humans), t.infer, t.postprc, t.tot, 1 / t.tot), LogLevel.DEBUG)

        self.fps_time = 1 / t.tot

        # if not self.run_flag: return
        """
    callbacks are already running in their own threads!
    https://answers.ros.org/question/304117/opencv-namedwindow-goes-grey-is-not-getting-events/
    """

        try:
            if self.first_signal:
                self.reidHuman = followMeFunctions.select_closest(humans)  # selects the closest human
                self.add_new_user_to_watch_list_run_time_reid(self)

            # searches and saves selected of any human in human for one frame
            if self.humans and len(self.humans) > 0:
                self.first_signal = True
                self.humans[0].selected = True
                self.log("first human is a target")

            self.xy = (-1, -1)
            if self.first_signal:
                self.add_new_user_to_watch_list_run_time_reid(self)  # reset reid on click

            # print self.first_signal
            # if rospy.get_param("followMe_wave_flag", False):
            #     try:
            #         for human in humans:
            #             if human.wavingConf >= 0.5: human.selected = True
            #     except Exception as identifier:
            #         print("fuk waving")
            print("getAngle")
            followMeFunctions.get_angle(self.image, humans, imgcopy=False)
            print("get dist depth")
            print("depth image: ", self.depth_image)
            followMeFunctions.get_distance_depth(self.depth_image, humans, self.depth_scale, imgcopy=False)

            # SET AND REMEMBER OF SELECTED HUMAN
            # after a human is selected thru click this function will save him
            self.stc.set_target_human(humans, self.image)
            # this function selects a human on the base of previous selected human
            # (without this function the algorithm can not remember humans for longer than a single frame)
            self.stc.remember_target_human_kalman_color_closest(humans, self.image)
            self.stc.search_for_highest_score_and_select(humans, self.image)

            # Publish values of selected
            if self.stc.search_for_selected(humans):

                self.selectedFlagPublisher(True)
                self.frames_kalman_color_lost = 0
                self.reid_tries = 0
                self.reid_frames_tracked += 1

                if self.stc.target_human.pose:
                    pose_transformed: FollowMePose = self.stc.target_human.pose
                    print("publish pose")
                    print(pose_transformed.position.x)
                    print(pose_transformed.position.y)
                    pose_transformed.position.z = 0
                    self.stc.set_posX_of_selected(pose_transformed.position.x)
                    self.stc.set_posY_of_selected(pose_transformed.position.y)

                    behind_x, behind_y = FollowMe.get_position_behind(pose_transformed.position,
                                                                      self.stc.get_dist_of_selected(), 1.7)
                    print(behind_x)
                    print(behind_y)
                    #pose_transformed.position.x = behind_x
                    #pose_transformed.position.y = behind_y

                    angle = math.radians(-self.stc.get_angle_of_selected())

                    is_new_goal_far_enough = (
                            self.last_published_pose is None or
                            followMeFunctions.distance_between(pose_transformed, self.last_published_pose) >= 0.1
                    )

                    if (self.timer > 5) and is_new_goal_far_enough:
                        self.last_published_pose = pose_transformed
                        self.goalPublisher(pose_transformed.position.x, pose_transformed.position.y, angle)
                        self.timer = 0
                        # self.pubSelectedLocation.publish(pose_transformed)

                if self.reid_frames_tracked % REID_FRAME_SKIP == 0:
                    self.reidHuman = self.stc.target_human
                    self.add_sample_existing_user_in_watch_list_run_time_reid(self)

            else:
                self.frames_kalman_color_lost += 1

                if self.frames_kalman_color_lost >= 180:

                    if self.reid_tries > REID_MAX_TRIES:
                        self.selectedFlagPublisher(False)
                        print("all is lost! ruuuun!")
                    else:
                        # >>> ask reid to find operator
                        print("lost operator, try %r to re-find by reid ..." % self.reid_tries)
                        self.reid_tries += 1

                        operatorFound = False
                        for h in humans:
                            self.reidHuman = h
                            name = self.process_frame_reid(self)
                            if name == "operator":
                                h.selected = True
                                self.stc.set_target_human(humans, self.image)
                                print("found operator again!")
                                operatorFound = True
                                break

                        if not operatorFound:
                            # no operator was found
                            self.selectedFlagPublisher(False)

                        # <<<
            # /else

            self.xy = (0, 0)
            self.timer += 1
            self.first_signal = False

        except Exception as e:
            self.log(e, LogLevel.ERROR)
            self.log(e.args, LogLevel.ERROR)
            self.first_signal = False
            self.selectedFlagPublisher(False)

    def mouse_click_location(self, event, x, y, flags, param):
        if event == cv2.EVENT_FLAG_LBUTTON: self.xy = (x, y)

    def mouse_click(self, event: MouseEvent):
        x = event.xdata
        y = event.ydata
        self.xy = (x, y)
        pass

    @staticmethod
    def get_position_behind(person_pose, distance, distance_to_keep):
        modifier = 0
        if distance != 0: modifier = (distance - distance_to_keep) / distance
        x_behind = modifier * person_pose.x
        y_behind = modifier * person_pose.y
        return x_behind, y_behind

    def handle_run_flag(self, obj):
        self.run_flag = obj.data
        if self.run_flag:
            self.first_signal = True  # signals if closest should be selected
