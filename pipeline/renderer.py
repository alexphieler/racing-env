from direct.showbase.ShowBase import ShowBase

from direct.actor.Actor import Actor
from panda3d.core import WindowProperties

# In your import statements:
from panda3d.core import AmbientLight
from panda3d.core import Vec4
from panda3d.core import GeomNode
from panda3d.core import Camera, Lens, OrthographicLens

from panda3d.core import DirectionalLight, Spotlight
from panda3d.core import PointLight

from panda3d.core import FrameBufferProperties, WindowProperties
from panda3d.core import GraphicsPipe, GraphicsOutput
from panda3d.core import Texture
from panda3d.core import NodePath
from panda3d.core import SamplerState, CardMaker, PandaNode

from panda3d.core import ShaderTerrainMesh, Shader, load_prc_file_data
from panda3d.core import loadPrcFileData

from panda3d.core import GeomVertexReader

from direct.task import Task

import numpy as np

import cv2
import os

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    def get_package_share_directory(package_name):
        if package_name == "pacsim":
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pacsim_ws", "src", "pacsim"))
        raise ImportError("ament_index_python is not installed")

import simplepbr

from typing import List

from line_profiler import profile

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(PIPELINE_ROOT, "Models")

def getConfigFilePath(name, dir='params'):
    if os.path.isabs(name):
        return name
    return os.path.join(PIPELINE_ROOT, name)

class ImageBuffer:
    BUFFER_W = 320
    BUFFER_H = 320

    def __init__(
            self,
            width: float,
            height: float,
            engine: ShowBase,
            lens: Lens,
            position: np.array,
            orientation: np.array,
            frame_buffer_property=None,
    ):
        self.BUFFER_W = width
        self.BUFFER_H = height
        if frame_buffer_property is None:
            self.buffer = engine.win.makeTextureBuffer("camera23", width, height, to_ram=True)
        else:
            self.buffer = engine.win.makeTextureBuffer("camera", width, height, fbp=frame_buffer_property)

        self.origin = engine.my_camera2
        # this takes care of setting up their camera properly
        self.engine = engine

        self.lens = lens
        self.cam = self.engine.makeCamera(self.buffer, lens=lens)
        self.cam.reparentTo(self.origin)
        # self.cam.setPos(0,0,0)
        # self.cam.setHpr(0,0,0)
        self.cam.setPos(position[0],position[1],position[2])
        self.cam.setHpr(orientation[0],orientation[1],orientation[2])
        self.display_region = None


        winprops = WindowProperties.size(width, height)
        fbprops = FrameBufferProperties()
        fbprops.setDepthBits(1)
        self.depthBuffer = self.engine.graphicsEngine.makeOutput(
            engine.pipe, "depth buffer", -2,
            fbprops, winprops,
            GraphicsPipe.BFRefuseWindow,
            engine.win.getGsg(), engine.win)
        self.depthTex = Texture()
        self.depthTex.setFormat(Texture.FDepthComponent)
        self.depthBuffer.addRenderTexture(self.depthTex,
            GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
        # self.depthBuffer.set
        # lens = self.cam.node().getLens()
        # lens = my_cam2.getLens()
        # the near and far clipping distances can be changed if desired
        # lens.setNear(5.0)
        # lens.setFar(500.0)
        self.depthCam = self.engine.makeCamera(self.depthBuffer,
            lens=lens,
            scene=engine.render)
        self.depthCam.reparentTo(self.cam)
        self.depthCam.setPos(position[0],position[1],position[2])
        self.depthCam.setHpr(orientation[0],orientation[1],orientation[2])

        # see init() accepts more parameters, like msaa, etc.
        self.pbrpipe=simplepbr.init(camera_node=self.cam, window=self.buffer, render_node=engine.render, enable_shadows=True)
        # self.pbrpipe=simplepbr.init(camera_node=self.cam, window=self.buffer, render_node=render)

    @profile
    def get_rgb_array(self):
        self.engine.graphicsEngine.renderFrame()
        # print(self.engine)
        # print(self.buffer.getTexture())
        # print(self.buffer.getTexture().mightHaveRamImage())
        origin_img = self.buffer.getDisplayRegion(1).getScreenshot()
        img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)
        img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), 4))
        img = img[::-1]
        img = img[..., :-1]
        return img

    def get_camera_depth_image(self):
        """
        Returns the camera's depth image, which is of type float32 and has
        values between 0.0 and 1.0.
        """
        data = self.depthTex.getRamImage()
        depth_image = np.frombuffer(data, np.float32)
        depth_image.shape = (self.depthTex.getYSize(), self.depthTex.getXSize(), self.depthTex.getNumComponents())
        depth_image = np.flipud(depth_image)
        return depth_image

    def add_display_region(self, display_region: List[float]):
        self.display_region = self.engine.win.makeDisplayRegion(*display_region)
        self.display_region.setCamera(self.buffer.getDisplayRegions()[1].camera)

    def remove_display_region(self):
        self.engine.win.removeDisplayRegion(self.display_region)
        self.display_region = None


class Game(ShowBase):
    def update(self):
        # print("wow")
        # print(self.camera.getPos())
        self.iters += 0.02
        self.updateSteering(90*np.sin(self.iters))
        self.updateWheelRotations(self.iters, self.iters, self.iters, self.iters)
        return
        # return Task.again
    
    def updateCarPose(self, x,y,rot):
        # car model has origin at front axle, compensate for that
        pFA = np.array([0.9,0.0,0.0])
        angle = rot * np.pi / 180.0
        r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        pCog = np.array(np.array([x,y]) + r @ pFA[0:2])
        # print(x)
        # print(y)
        # print(pCog)
        self.car.setPos(pCog[0], pCog[1], 0)
        # self.car.setPos(x, y, 0)
        self.car.setH(rot)

        lightPosition = [-self.shadowDistClose*self.lightDirection[0]+x, -self.shadowDistClose*self.lightDirection[1]+y, -self.shadowDistClose*self.lightDirection[2]]        
        self.light.setPos(*lightPosition)
        # lightPosition = [x, y, 0.0]        
        # self.light2.setPos(*lightPosition)
        # self.light.setHpr(-100, -2, 0)
        # self.light.setHpr(-90, 0, 0)
        # self.light.setH(-90)
        # self.light.setY(20)
        # self.light.setZ(1)
        # self.light.setPos(5, -1, 2)

        return


    def updateSteering(self, steering_angle):
        # steering_angle = 90
        # a = self.car.ls()
        # dummySteering = self.car.controlJoint(None, "modelRoot", "Joint Name")
        # a = self.car.getGeomNode()
        FL = self.car.find("**/FL_Inside")
        FL.setH(steering_angle*0.23)

        FR = self.car.find("**/FR_Inside")
        FR.setH(steering_angle*0.23)

        steering = self.car.find("**/Steering_Wheel")
        steering.setP(-steering_angle)

    def updateWheelRotations(self, fl, fr, rl, rr):
        # steering_angle = 90
        # a = self.car.ls()
        # dummySteering = self.car.controlJoint(None, "modelRoot", "Joint Name")
        # a = self.car.getGeomNode()
        FL = self.car.find("**/FL_Outside")
        FL.setR(fl)
        FR = self.car.find("**/FR_Outside")
        FR.setR(fr)

        RL = self.car.find("**/RL_Outside")
        RL.setR(rl)
        RR = self.car.find("**/RR_Outside")
        RR.setR(rr)


    def __init__(self):
        self.usepbr = True
        ShowBase.__init__(self, windowType='offscreen')

        # loadPrcFileData("", "egl-device-index 0")
        # loadPrcFileData("", "load-display p3headlessgl")
        # loadPrcFileData("", "window-type offscreen")
        # loadPrcFileData("", f"win-size 1920 1080")
        # loadPrcFileData("", "audio-library-name null")

        if(self.usepbr):
          spbr = simplepbr.init()
          # spbr.shadow_bias = 0.001
          # spbr.use_normal_maps = True
          # spbr.enable_shadows = True
        # spbr.use_normal_maps = True

        self.imgSize = [int(2448/4), int(2048/4)]
        self.rootNode = render.attachNewNode('rootNode')
        self.ConesNode = render.attachNewNode('conesNode')
        self.ConesNode.reparentTo(self.rootNode)
        trackNames = ["track_asphalt.glb", "track_asphalt2.glb", "track_pflaster.glb"]
        self.environment = loader.loadModel(os.path.join(MODELS_ROOT, "track", trackNames[2]))
        self.environment.reparentTo(self.rootNode)
        # print(self.environment.ls())
        # print(self.environment.findTexture("*"))
        self.environment.findTexture("*").setAnisotropicDegree(16)
        # self.environment.findTexture("asph2").setAnisotropicDegree(16)
        # print(self.environment.findAllTextures())
        # Iterate through all the texture stages applied to the model
        # for i in range(self.environment.getNumTextures()):
        #     texture = self.environment.getTexture(i)
        #     print(texture)
            # if texture is not None:
                # textures.append(texture)

        # cm = CardMaker('')
        # cm.setFrame(-2, 2, -2, 2)
        # floor = self.rootNode.attachNewNode(PandaNode("floor"))
        # for y in range(200):
        #     for x in range(200):
        #         nn = floor.attachNewNode(cm.generate())
        #         nn.setP(-90)
        #         nn.setPos((x - 6) * 4, (y - 6) * 4, 0)
        # #floor.setTexture(floorTex)
        # floor.flattenStrong()

        # self.environment = floor
        # self.environment.reparentTo(self.rootNode)

        # Load a skybox - you can safely ignore this code
        skybox = self.loader.loadModel(os.path.join(MODELS_ROOT, "skysphere.glb"))
        skybox.reparent_to(self.render)
        skybox.setPos(0,0,-20)
        # skybox.setHpr(0,0,-90)
        skybox.set_scale(3000)

        self.iters = 0
        # self.updateTask = taskMgr.add(self.update, "update")

        # properties = WindowProperties()
        # # properties.setSize(1000, 750)
        # properties.setSize(self.imgSize[0], self.imgSize[1])
        # self.win.requestProperties(properties)
        # self.disableMouse()

        # self.environment = loader.loadModel(getConfigFilePath("Models/Misc/Environment/environment"))
        # self.environment = loader.loadModel(getConfigFilePath("Models/track.glb"))
        # image = self.loader.load_texture("/home/devuser/efr_ws/src/efr_panda3d/Models/track/asph2.jpg")
        # self.environment.set_texture( self.environment.find_texture_stage('*'), image, 1)




        self.car = loader.loadModel(os.path.join(MODELS_ROOT, "car.glb"), noCache=True)
        
        self.car.reparentTo(self.rootNode)
        self.car.setPos(0, 0, 0)
        # self.car

        self.blueCones = []
        self.yellowCones = []
        # self.addBlueCones()
        # self.addYellowCones()

        my_cam2 = Camera("cam2")
        my_camera2 = self.car.attachNewNode(my_cam2)
        my_camera2.setName("camera2")
        my_camera2.setPos(-1,0,1)
        # my_camera2.setP(-15)
        my_camera2.setP(0)
        my_camera2.setH(-90)
        self.my_camera2 = my_camera2
        # my_cam2.get_lens().setFov(120)
        my_cam2.get_lens().setFilmSize(6.71/1000, 5.61/1000)
        my_cam2.get_lens().setFocalLength(4.5/1000)
        # my_cam2.get_lens().setFocalLength(2.5/1000)
        my_cam2.get_lens().setNear(0.1)


        # Needed for camera image


        # self.camera.setH(-90)
        # my_cam2.get_lens().setFov(120)
        # self.camera.get_lens().setFilmSize(6.71/1000, 5.61/1000)
        # self.camera.get_lens().setFocalLength(4.5/1000)


        # my_camera2.lookAt(self.car)

        self.dr = self.camNode.getDisplayRegion(0)
        dr = base.camNode.getDisplayRegion(0)
        dr.setActive(0) # Or leave it (dr.setActive(1))

        window = dr.getWindow()
        dr1 = window.makeDisplayRegion(0, 1, 0, 1)
        dr1.setSort(dr.getSort())
        dr1.setCamera(my_camera2)

        self.camLens.setFilmSize(6.71, 5.61)
        self.camLens.setFocalLength(4.5)
        # self.camLens.setFocalLength(3.0)
        self.camLens.setNear(0.1)

        self.imbufclass = ImageBuffer(int(2448/8), int(2048/8), self, self.camLens, np.array([0,0,0]), np.array([0,-15,0]))
        self.imbufclass2 = ImageBuffer(int(2448/8), int(2048/8), self, self.camLens, np.array([0,0,0]), np.array([70,-15,0]))
        self.imbufclass3 = ImageBuffer(int(2448/8), int(2048/8), self, self.camLens, np.array([0,0,0]), np.array([-70,-15,0]))
        # self.imbufclass.add_display_region([0.5, 1, 0, 1])


        # coneB = loader.loadModel("/home/devuser/efr_ws/src/efr_panda3d/Models/blue.glb")
        # coneB.reparentTo(self.rootNode)
        # coneB.setPos(5,0,0)
        # coneB2 = loader.loadModel("/home/devuser/efr_ws/src/efr_panda3d/Models/blue.glb")
        # coneB2.reparentTo(self.rootNode)
        # coneB2.setPos(5,-0.4,0)


        ambientLight = AmbientLight("ambient light")
        ambientLight.setColor(Vec4(0.1, 0.1, 0.1, 1))
        self.ambientLightNodePath = self.rootNode.attachNewNode(ambientLight)
        # self.render.setLight(self.ambientLightNodePath)
        self.rootNode.setLight(self.ambientLightNodePath)

        ambientLightCones = AmbientLight("cones light")
        ambientLightCones.setColor(Vec4(0.1, 0.1, 0.1, 1))
        self.ambientLightConesNodePath = self.ConesNode.attachNewNode(ambientLightCones)
        # self.render.setLight(self.ambientLightNodePath)
        self.ConesNode.setLight(self.ambientLightConesNodePath)


        # In the body of your code
        # mainLight = DirectionalLight("main light")
        # mainLight.setColor((1.0, 1.0, 1.0, 1))
        #         # Enable shadows; we need to set a frustum for that.
        # mainLight.get_lens().set_near_far(1, 300)
        # mainLight.get_lens().set_film_size(20, 40)
        # mainLight.setShadowCaster(True, 2048, 2048)
        # self.mainLightNodePath = render.attachNewNode(mainLight)
        # self.mainLightNodePath.node().setScene(render)
        # # Turn it around by 45 degrees, and tilt it down by 45 degrees
        # self.mainLightNodePath.setHpr(45, -45, 0)
        # self.mainLightNodePath.set_pos(0,0,10)
        # # self.environment.setLight(self.mainLightNodePath)
        # self.render.setLight(self.mainLightNodePath)


        self.light = self.rootNode.attachNewNode(DirectionalLight("Spot"))
        self.light.node().setColor(Vec4(1.01, 1.01, 1.01, 1))
        # self.light = self.rootNode.attachNewNode(PointLight("main light"))
        # self.light.setHpr(45, -45, 0)
        self.lightDirection = np.array([100,0,-10])
        self.lightDirection = self.lightDirection / np.linalg.norm(self.lightDirection)
        self.lightDirection = self.lightDirection.tolist()
        self.light.lookAt(*self.lightDirection)
        # self.light.setHpr(-100, -2, 0)
        # self.light.setHpr(-90, 0, 0)
        # self.light.setH(-90)
        # self.light.setY(20)
        # self.light.setZ(1)
        # self.light.setPos(5, -1, 2)
        # self.light.node().setScene(self.rootNode)
        self.light.node().setShadowCaster(True, 2048, 2048)
        # self.light.node().showFrustum()
        # self.light.node().getLens().setFov(360)
        self.shadowDistClose = 20
        # self.shadowDistFar = self.shadowDistClose
        self.light.node().get_lens().set_film_size(2*self.shadowDistClose, 2*self.shadowDistClose)
        self.light.node().getLens().setNearFar(3,2*self.shadowDistClose)
        self.rootNode.setLight(self.light)

        # self.light2 = self.rootNode.attachNewNode(DirectionalLight("Spot2"))
        # self.light2.node().setColor(Vec4(1.01, 1.01, 1.01, 1))
        # self.light2.lookAt(*self.lightDirection)
        # self.light2.node().setShadowCaster(True, 2*2048, 2*2048)
        # # self.light.node().showFrustum()
        # # self.light.node().getLens().setFov(360)
        # # self.shadowDist = 10
        # self.light2.node().get_lens().set_film_size(self.shadowDistFar, self.shadowDistFar)
        # self.light2.node().getLens().setNearFar(self.shadowDistClose,2*self.shadowDistFar)
        # self.rootNode.setLight(self.light2)



        # self.conesLight = self.ConesNode.attachNewNode(DirectionalLight("Spot"))
        # self.conesLight.node().setColor(Vec4(20.01, 20.01, 20.01, 1))
        # self.conesLight.setHpr(45, -45, 0)
        # self.conesLight.node().setScene(self.ConesNode)
        # self.ConesNode.setLight(self.conesLight)

        # buffer = self.light.node().getShadowBuffer(self.win.gsg)
        # buffer.active = True

        if(not self.usepbr):
          self.render.setShaderAuto()

    def addBlueCones(self, coords):
        for i in coords:    
        #   coneB = loader.loadModel("/home/devuser/efr_ws/src/efr_panda3d/Models/blue.glb")
            coneB = loader.loadModel(os.path.join(MODELS_ROOT, "blue.glb"))
            coneB.reparentTo(self.ConesNode)
            coneB.setPos(i[0], i[1], i[2])
            self.blueCones.append(coneB)

    def addYellowCones(self, coords):
        for i in coords:    
          # coneY = loader.loadModel(getConfigFilePath("Models/yellow.glb"))
        #   coneY = loader.loadModel("/home/devuser/efr_ws/src/efr_panda3d/Models/yellow.glb")
                        coneY = loader.loadModel(os.path.join(MODELS_ROOT, "yellow.glb"))
                        coneY.reparentTo(self.ConesNode)
                        coneY.setPos(i[0], i[1], i[2])
                        self.yellowCones.append(coneY)

    def removeCones(self):
        for i in self.blueCones:
            i.detachNode()
        for i in self.yellowCones:
            i.detachNode()

    def get_vertices(self,nodepath):
        vertices = []

        # Find all GeomNodes inside this model
        for node_path in nodepath.findAllMatches('**/+GeomNode'):
            geom_node = node_path.node()  # Now we're working with GeomNode

            for i in range(geom_node.getNumGeoms()):
                geom = geom_node.getGeom(i)
                vdata = geom.getVertexData()
                reader = GeomVertexReader(vdata, 'vertex')

                while not reader.isAtEnd():
                    vertex = reader.getData3f()
                    vertices.append(vertex)

        return vertices
    

    def getConeCoords(self):
        points = []
        ang_steps = np.linspace(0, 2*np.pi, 100)
        # top ring
        z_top = 0.319846
        r_top = 0.0185
        for i in ang_steps:
            points.append(np.array([r_top*np.cos(i), r_top*np.sin(i), z_top]))
        # bottom ring
        z_bot = 0.025813
        r_bot = 0.0791
        for i in ang_steps:
            points.append(np.array([r_bot*np.cos(i), r_bot*np.sin(i), z_bot]))
        # base ring
        z_base = 0.025813
        r_base = 0.109
        for i in ang_steps:
            points.append(np.array([r_base*np.cos(i), r_base*np.sin(i), z_base]))    
        # base 
        edge = 0.111583
        points.append(np.array([edge, edge, 0.0]))
        points.append(np.array([edge, -edge, 0.0]))
        points.append(np.array([-edge, edge, 0.0]))
        points.append(np.array([-edge, -edge, 0.0]))
        return points

    def getConeCoordsKP(self):
        pointsTop = []
        pointsBot = []
        ang_steps = np.linspace(0, 2*np.pi, 100)
        # top ring
        z_top = 0.319846
        r_top = 0.0185
        for i in ang_steps:
            pointsTop.append(np.array([r_top*np.cos(i), r_top*np.sin(i), z_top]))
        # bottom ring
        z_bot = 0.025813
        r_bot = 0.0791
        for i in ang_steps:
            pointsBot.append(np.array([r_bot*np.cos(i), r_bot*np.sin(i), z_bot]))
        return pointsTop, pointsBot

    def processCones(self, imbuf, cones):
        outPoints = []
        outBoxes = []
        kpPoints = []
        outKps = []
        outPositions = []
        coneCoords = self.getConeCoords()
        coneCoordsKP = self.getConeCoordsKP()
        # f = 1641.72876304
        # px = 1224.0
        # py = 1024.0
        f = imbuf.BUFFER_W * imbuf.lens.focal_length / imbuf.lens.film_size[0]
        px = imbuf.BUFFER_W / 2.0
        py = imbuf.BUFFER_H / 2.0

        for cone in cones:
            currentConePoints = []
            trans = imbuf.cam.get_transform(cone).get_inverse().get_mat()

            rot = np.array([[trans[0][0], trans[0][1], trans[0][2]], [trans[1][0], trans[1][1], trans[1][2]], [trans[2][0], trans[2][1], trans[2][2]]])
            translation = np.array([trans[3][0], trans[3][1], trans[3][2]])
            if(translation[1] < 0.2):
                continue
            for i in coneCoords:
                position = i
                position = position@rot + translation
                yImg = -((f/position[1])*position[2]) + py
                xImg = ((f/position[1])*position[0]) + px
                # if(position[1] > 0):
                    # outPoints.append(np.array([xImg, yImg]))
                currentConePoints.append(np.array([xImg, yImg]))
            if(len(currentConePoints) > 0):
                minX = np.inf
                maxX = -np.inf
                minY = np.inf
                maxY = -np.inf
                for i in currentConePoints:
                    if(i[0] > maxX):
                        maxX = i[0]
                    if(i[0] < minX):
                        minX = i[0]
                    if(i[1] > maxY):
                        maxY = i[1]
                    if(i[1] < minY):
                        minY = i[1]
                outPoints.append(currentConePoints)
                outBoxes.append(np.array([minX, minY, maxX, maxY]))
                outPositions.append(np.array([translation[1], -translation[0], translation[2]]))


            topPoint = [0.0,-1e6]
            imageSpaceTop = []
            for i in coneCoordsKP[0]:
                position = i
                position = position@rot + translation
                yImg = -((f/position[1])*position[2]) + py
                xImg = ((f/position[1])*position[0]) + px

                imageSpaceTop.append(np.array([xImg, yImg]))
            meanTop = np.mean( np.array(imageSpaceTop), axis=0 )
            midTop = 0.5*(np.max(np.array(imageSpaceTop), axis=0) + np.min(np.array(imageSpaceTop), axis=0))

            for i in coneCoordsKP[0]:
                position = i
                position = position@rot + translation
                yImg = -((f/position[1])*position[2]) + py
                xImg = ((f/position[1])*position[0]) + px

                # if(position[1] > 0):
                if(yImg >= meanTop[1]):
                    if((xImg-midTop[0])**2 < (topPoint[0]-midTop[0])**2):
                        topPoint = np.array([xImg, yImg])
            kpPoints = []
            kpPoints.append(topPoint)
            
            leftPoint = [1e6,0.0]
            rightPoint = [-1e6,0.0]
            for i in coneCoordsKP[1]:
                position = i
                position = position@rot + translation
                yImg = -((f/position[1])*position[2]) + py
                xImg = ((f/position[1])*position[0]) + px
                # if(position[1] > 0):
                if(xImg < leftPoint[0]):
                    leftPoint = np.array([xImg, yImg])
                if(xImg > rightPoint[0]):
                    rightPoint = np.array([xImg, yImg])
            kpPoints.append(leftPoint)
            kpPoints.append(rightPoint)
            outKps.append(kpPoints)

        return outPoints, outBoxes, outKps, outPositions

    def filterData(self, imbuf, points, boxes, kpPoints, positions, classes):
        outPoints = []
        outBoxes = []
        outKPs =  []
        outPositions =  []
        outClasses = []
        resHor = imbuf.BUFFER_W
        resVer = imbuf.BUFFER_H
        for i in range(0,len(points)):
            ps = points[i]
            box = boxes[i]
            kps = kpPoints[i]
            pos = positions[i]
            cl = classes[i]
            outsideOfImg = False
            if(box[2] <= 0):
                outsideOfImg = True
            if(box[3] <= 0):
                outsideOfImg = True
            if(box[0] >= resHor):
                outsideOfImg = True
            if(box[1] >= resVer):
                outsideOfImg = True
            if(not outsideOfImg):
                outPoints.append(ps)
                outBoxes.append(box)
                outKPs.append(kps)
                outPositions.append(pos)
                outClasses.append(cl)
        return (outPoints, outBoxes, outKPs, outPositions, outClasses)

    def getLabels(self, imbuf):

        outPoints, outBoxes, kpPoints, positions = self.processCones(imbuf, self.blueCones)
        outClasses = [0] * len(positions)
        outPointsY, outBoxesY, kpPointsY, positionsY = self.processCones(imbuf, self.yellowCones)

        outPoints.extend(outPointsY)
        outBoxes.extend(outBoxesY)
        kpPoints.extend(kpPointsY)
        positions.extend(positionsY)
        outClasses.extend([1] * len(positionsY))

        outPoints, outBoxes, kpPoints, positions, outClasses = self.filterData(imbuf, outPoints, outBoxes, kpPoints, positions, outClasses)

        return outPoints, outBoxes, kpPoints, positions, outClasses
    
