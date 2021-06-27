import os
import json
import numpy as np


class colmapFormat(object):
    """
    Module to convert colmap output-to-txt format to compatible nerf format
    """
    def __init__(self, read_dir):
        """
        TODO: support other different colmap output format as input?
        Args:
            read_dir (str), directory where the colmap txt file resides
        """
        self.cameras = self.loadCamerasTxt(os.path.join(read_dir, 'cameras.txt'))
        self.captures = self.loadImagesTxt(os.path.join(read_dir, 'images.txt'))
        self.getExtrinsics()    # get extrinsic matrix from capture quaternion and translation


    def loadCamerasTxt(self, read_name):
        """
        Load the camera poses from camers.txt
        Args:
            read_name (str), path to the text file
        returns:
            cameras (list of dictionary), intrinsics for each camera
        """
        cameras = []
        with open(read_name) as file:
            for i, line in enumerate(file.readlines()):
                if i > 2:   # first 3 lines are comments, skip
                    cur = line.replace('\n', '').split(' ')
                    cameras.append({
                        'CAMERA_ID':int(cur[0]), 
                        'MODEL':cur[1], 
                        'WIDTH':int(cur[2]), 
                        'HEIGHT':int(cur[3]), 
                        'PARAMS':[float(x) for x in cur[4:]],
                        })        
        # print(cameras[0])
        return cameras
        

    def loadImagesTxt(self, read_name):
        """
        Load extrinsics of captures from images.txt
        Args:
            read_name (str), path to the text file
        returns:
            captures (list of dictionary), extrinsic parameters for each capture
        """
        captures = []
        with open(read_name) as file:
            for i, line in enumerate(file.readlines()):
                if i > 3 and i % 2 == 0:         # first 4 lines are comments, skip; rest are camera / coordiantes
                    cur = line.replace('\n', '').split(' ')
                    captures.append({
                        'IMAGE_ID': int(cur[0]),
                        'QW': float(cur[1]),
                        'QX': float(cur[2]),
                        'QY': float(cur[3]),
                        'QZ': float(cur[4]),
                        'TX': float(cur[5]),
                        'TY': float(cur[6]),
                        'TZ': float(cur[7]),
                        'CAMERA_ID': int(cur[8]) - 1,
                        'NAME': cur[9],
                        })    
        print(captures[0])
        return captures


    def quaternionToRotationMatrix(self, qw, qx, qy, qz):
        """
        Equation from: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
        Args:
            qw, qx, qy, qz (scalars), scalar-first quaternion representation of quaternion
        Returns:
            R (numpy array of [3,3]), rotation matrix
        """
        R = np.zeros([3,3])

        R[0,0] = 1 - 2*qy*qy - 2*qz*qz
        R[0,1] = 2*qx*qy - 2*qz*qw
        R[0,2] = 2*qx*qz + 2*qy*qw

        R[1,0] = 2*qx*qy + 2*qz*qw
        R[1,1] = 1 - 2*qx*qx - 2*qz*qz
        R[1,2] = 2*qy*qz - 2*qx*qw

        R[2,0] = 2*qx*qz - 2*qy*qw
        R[2,1] = 2*qy*qz + 2*qx*qw
        R[2,2] = 1 - 2*qx*qx - 2*qy*qy

        return R


    def getExtrinsics(self):
        """
        Add a key "rotation matrix" to self.captures using quarternion
        """
        for capture in self.captures:
            capture['R'] = self.quaternionToRotationMatrix(
                capture['QW'], capture['QX'], capture['QY'], capture['QZ'])
            capture['T'] = np.array([capture['TX'], capture['TY'], capture['TZ']])
            capture['extrinsic'] = np.concatenate([capture['R'], capture['T'].reshape(-1, 1)], axis=1)
            capture['extrinsic'] = np.concatenate([capture['extrinsic'], np.array([0, 0, 0, 1]).reshape(1, -1)], axis=0)


    def dumpJson(self, write_dir="temp", write_name="transforms.json"):
        """
        Dump the object intrinsic and extrinsic data to json file to match synthetic dataset format
        Args:
            write_dir (str) directory to write json
            write_name (str) name to write
        Return:
            res (dictionary) parameters as dumped to the json file
        """
        frames = []
        for capture in self.captures:
            camera = self.cameras[capture['CAMERA_ID']]
            frame = {
               "file_path": capture["NAME"],
               "rotation": camera["PARAMS"][-1],
               "transform_matrix": capture['extrinsic'].tolist(),
                }
            frames.append(frame)
        res = {
            "camera_angle_x": np.arctan2(camera['PARAMS'][1], camera['PARAMS'][0]) * 2, 
            "frames": frames,
            }

        if not os.path.isdir(write_dir):
            os.makedirs(write_dir)
        
        with open(os.path.join(write_dir, write_name), 'w') as json_file:
            json.dump(res, json_file, indent=4)
        
        return res



if __name__ == "__main__":
    model = colmapFormat(read_dir="Z:\\COLMAP\\data\\lego\\sparse")
    model.dumpJson(
        write_dir = "Z:\\Github\\nerf-pytorch\\data\\nerf_synthetic\\lego",
        write_name = "transforms_val1.json",
        )


