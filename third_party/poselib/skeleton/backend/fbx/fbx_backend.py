"""
This script reads an fbx file and returns the joint names, parents, and transforms.

NOTE: It requires the Python FBX package to be installed.
"""

import sys

import numpy as np

FBX_IMPORT_ERROR = None

try:
    import fbx
    import FbxCommon
except ImportError as e:
    FBX_IMPORT_ERROR = e
    print("Error: FBX library failed to load - importing FBX data will not succeed. Message: {}".format(e))
    print("FBX tools must be installed from https://help.autodesk.com/view/FBX/2020/ENU/?guid=FBX_Developer_Help_scripting_with_python_fbx_installing_python_fbx_html")


def fbx_to_npy(file_name_in, root_joint_name, fps):
    """
    This function reads in an fbx file, and saves the relevant info to a numpy array

    Fbx files have a series of animation curves, each of which has animations at different 
    times. This script assumes that for mocap data, there is only one animation curve that
    contains all the joints. Otherwise it is unclear how to read in the data.

    If this condition isn't met, then the method throws an error

    :param file_name_in: str, file path in. Should be .fbx file
    :return: nothing, it just writes a file.
    """

    if FBX_IMPORT_ERROR is not None:
        raise ImportError(
            "Autodesk FBX Python SDK import failed. Ensure both `fbx` and `FbxCommon` are "
            "importable in this environment. Original error: {}".format(FBX_IMPORT_ERROR)
        ) from FBX_IMPORT_ERROR

    # Create the fbx scene object and load the .fbx file
    fbx_sdk_manager, fbx_scene = FbxCommon.InitializeSdkObjects()
    FbxCommon.LoadScene(fbx_sdk_manager, fbx_scene, file_name_in)

    """
    To read in the animation, we must find the root node of the skeleton.
    
    Unfortunately fbx files can have "scene parents" and other parts of the tree that are 
    not joints
    
    As a crude fix, this reader just takes and finds the first thing which has an 
    animation curve attached
    """

    search_root = (root_joint_name is None or root_joint_name == "")

    # Get the root node of the skeleton, which is the child of the scene's root node
    possible_root_nodes = [fbx_scene.GetRootNode()]
    found_root_node = False
    max_key_count = 0
    root_joint = None
    while len(possible_root_nodes) > 0:
        joint = possible_root_nodes.pop(0)
        if not search_root:
            if root_joint_name in joint.GetName():
                root_joint = joint
        try:
            curve = _get_animation_curve(joint, fbx_scene)
        except RuntimeError:
            curve = None
        if curve is not None:
            key_count = curve.KeyGetCount()
            if key_count > max_key_count:
                found_root_node = True
                max_key_count = key_count
                root_curve = curve
            if search_root and not root_joint:
                root_joint = joint
        for child_index in range(joint.GetChildCount()):
            possible_root_nodes.append(joint.GetChild(child_index))
    if not found_root_node:
        raise RuntimeError("No root joint found!! Exiting")

    joint_list, joint_names, parents = _get_skeleton(root_joint)

    """
    Read in the transformation matrices of the animation, taking the scaling into account
    """

    anim_range, frame_count, frame_rate = _get_frame_count(fbx_scene)

    local_transforms = []
    # Use the FPS reported by FBX time mode, not frame_count/time_range, which can be
    # inconsistent across SDK bindings/time modes.
    time_sec = anim_range.GetStart().GetSecondDouble()
    fbx_fps = float(frame_rate)
    if fps is not None:
        fbx_fps = float(fps)
    print("FPS: ", fbx_fps)
    while time_sec < anim_range.GetStop().GetSecondDouble():
        fbx_time = fbx.FbxTime()
        fbx_time.SetSecondDouble(time_sec)
        fbx_time = fbx_time.GetFramedTime()
        transforms_current_frame = []

        # Fbx has a unique time object which you need
        #fbx_time = root_curve.KeyGetTime(frame)
        for joint in joint_list:
            arr = np.array(_recursive_to_list(joint.EvaluateLocalTransform(fbx_time)))
            scales = np.array(_recursive_to_list(joint.EvaluateLocalScaling(fbx_time)))
            if not np.allclose(scales[0:3], scales[0]):
                raise ValueError(
                    "Different X, Y and Z scaling. Unsure how this should be handled. "
                    "To solve this, look at this link and try to upgrade the script "
                    "http://help.autodesk.com/view/FBX/2017/ENU/?guid=__files_GUID_10CDD"
                    "63C_79C1_4F2D_BB28_AD2BE65A02ED_htm"
                )
            # Adjust the array for scaling
            arr /= scales[0]
            arr[3, 3] = 1.0
            transforms_current_frame.append(arr)
        local_transforms.append(transforms_current_frame)

        time_sec += (1.0/fbx_fps)

    local_transforms = np.array(local_transforms)
    print("Frame Count: ", len(local_transforms))

    return joint_names, parents, local_transforms, fbx_fps



def _get_preferred_anim_stack(fbx_scene):
    num_anim_stacks = fbx_scene.GetSrcObjectCount(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimStack.ClassId)
    )
    if num_anim_stacks <= 0:
        raise RuntimeError("No animation stack found in FBX scene")

    # Prefer the stack with the highest frame rate, then longest duration.
    # Some files contain multiple stacks with the same duration but different time modes
    # (e.g. 30 FPS and 120 FPS variants of the same take).
    best_stack = None
    best_fps = -1.0
    best_duration = -1.0
    for index in range(num_anim_stacks):
        anim_stack = fbx_scene.GetSrcObject(
            FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimStack.ClassId), index
        )
        if anim_stack is None:
            continue

        span = anim_stack.GetLocalTimeSpan()
        duration = span.GetDuration().GetSecondDouble()
        global_time_mode = span.GetDuration().GetGlobalTimeMode()
        frame_rate = span.GetDuration().GetFrameRate(global_time_mode)

        if frame_rate > best_fps or (np.isclose(frame_rate, best_fps) and duration > best_duration):
            best_fps = frame_rate
            best_duration = duration
            best_stack = anim_stack

    if best_stack is None:
        raise RuntimeError("Failed to resolve a valid animation stack in FBX scene")
    return best_stack

def _get_frame_count(fbx_scene):
    # Get the animation stack used for frame timing.
    anim_stack = _get_preferred_anim_stack(fbx_scene)

    anim_range = anim_stack.GetLocalTimeSpan()
    duration = anim_range.GetDuration()
    global_time_mode = duration.GetGlobalTimeMode()
    fps = duration.GetFrameRate(global_time_mode)

    # FBX Python SDK APIs differ by version:
    # - some accept GetFrameCount(bool)
    # - newer bindings require GetFrameCount(FbxTime.EMode)
    try:
        frame_count = duration.GetFrameCount(global_time_mode)
    except TypeError:
        try:
            frame_count = duration.GetFrameCount()
        except TypeError:
            frame_count = duration.GetFrameCount(True)

    return anim_range, frame_count, fps

def _get_animation_curve(joint, fbx_scene):
    # Get the animation stack used for frame timing.
    anim_stack = _get_preferred_anim_stack(fbx_scene)

    num_anim_layers = anim_stack.GetSrcObjectCount(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimLayer.ClassId)
    )
    if num_anim_layers != 1:
        raise RuntimeError(
            "More than one animation layer was found. "
            "This script must be modified to handle this case. Exiting"
        )
    animation_layer = anim_stack.GetSrcObject(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimLayer.ClassId), 0
    )

    def _check_longest_curve(curve, max_curve_key_count):
        longest_curve = None
        if curve and curve.KeyGetCount() > max_curve_key_count[0]:
            max_curve_key_count[0] = curve.KeyGetCount()
            return True

        return False

    max_curve_key_count = [0]
    longest_curve = None
    for c in ["X", "Y", "Z"]:
        curve = joint.LclTranslation.GetCurve(
            animation_layer, c
        )  # sample curve for translation
        if _check_longest_curve(curve, max_curve_key_count):
            longest_curve = curve

        curve = joint.LclRotation.GetCurve(
            animation_layer, "X"
        )
        if _check_longest_curve(curve, max_curve_key_count):
            longest_curve = curve

    return longest_curve


def _get_skeleton(root_joint):

    # Do a depth first search of the skeleton to extract all the joints
    joint_list = [root_joint]
    joint_names = [root_joint.GetName()]
    parents = [-1]  # -1 means no parent

    def append_children(joint, pos):
        """
        Depth first search function
        :param joint: joint item in the fbx
        :param pos: position of current element (for parenting)
        :return: Nothing
        """
        for child_index in range(joint.GetChildCount()):
            child = joint.GetChild(child_index)
            joint_list.append(child)
            joint_names.append(child.GetName())
            parents.append(pos)
            append_children(child, len(parents) - 1)

    append_children(root_joint, 0)
    return joint_list, joint_names, parents


def _recursive_to_list(array):
    """
    Takes some iterable that might contain iterables and converts it to a list of lists 
    [of lists... etc]

    Mainly used for converting the strange fbx wrappers for c++ arrays into python lists
    :param array: array to be converted
    :return: array converted to lists
    """
    try:
        return float(array)
    except TypeError:
        return [_recursive_to_list(a) for a in array]


def parse_fbx(file_name_in, root_joint_name, fps):
    return fbx_to_npy(file_name_in, root_joint_name, fps)
