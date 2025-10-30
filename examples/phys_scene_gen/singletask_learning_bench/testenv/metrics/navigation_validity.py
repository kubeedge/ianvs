# Copyright 2025 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import numpy as np
import compress_json
from sedna.common.class_factory import ClassFactory, ClassType
import math
from typing import Dict, Any

import cv2
from skimage.graph import route_through_array
SKIMAGE_AVAILABLE = True

OBJAVERSE_ASSET_DIR = "/mnt/f/AI_Datasets/.objathor-assets"
if not os.path.exists(OBJAVERSE_ASSET_DIR):
    # print(f"CRITICAL WARNING: OBJAVERSE_ASSET_DIR not found at: {OBJAVERSE_ASSET_DIR}")
    # print("Metric will fail due to missing object dimensions.")
    pass

IMAGE_SIZE = 256
ROBOT_RADIUS_M = 0.15

def get_asset_metadata(obj: Dict[str, Any], objaverse_asset_dir: str) -> Dict[str, Any]:
    if "assetId" not in obj:
        raise ValueError(f"Object {obj.get('id')} is missing 'assetId'.")
    
    metadata_path = os.path.join(objaverse_asset_dir, obj["assetId"], "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata for object {obj.get('id')} ({obj['assetId']}) at {metadata_path}")
        
    with open(metadata_path, "r") as f:
        return json.load(f)

def get_bbox_dims(obj_data: Dict[str, Any], objaverse_asset_dir: str) -> np.ndarray:
    if "bbox" in obj_data:
        return np.array(obj_data["bbox"]["size"])

    asset_metadata = get_asset_metadata(obj_data, objaverse_asset_dir)
    
    if "boundingBox" in asset_metadata.get("secondaryProperties", {}):
        bbox_info = asset_metadata["secondaryProperties"]["boundingBox"]
        if "size" in bbox_info:
            return np.array(bbox_info["size"])
        elif "min" in bbox_info and "max" in bbox_info:
            min_p = np.array(bbox_info["min"])
            max_p = np.array(bbox_info["max"])
            return max_p - min_p
    
    raise ValueError(f"Could not determine bounding box size for asset {obj_data.get('assetId')}")

@ClassFactory.register(ClassType.GENERAL, alias="navigation_validity")
def navigation_validity(y_true, y_pred, **kwargs):
    scene_file_path = y_pred
    results = {
        "walkable_area_ratio": 0.0,
        "rooms_connected": False
    }

    if not SKIMAGE_AVAILABLE:
        return results["walkable_area_ratio"]

    try:
        scene_data = compress_json.load(scene_file_path)
        rooms, objects, doors = _parse_scene_layout_for_walkable(scene_data)
        if not rooms:
            return results["walkable_area_ratio"]

        walkable_map, room_mask, scale, origin = _generate_walkable_map(rooms, objects, doors, IMAGE_SIZE, ROBOT_RADIUS_M)
        if walkable_map is None:
            return results["walkable_area_ratio"]

        walkable_pixels = np.sum(walkable_map == 255)
        room_pixels = np.sum(room_mask > 0)
        if room_pixels > 0:
            results["walkable_area_ratio"] = walkable_pixels / room_pixels
            # print(f"Walkable Area Ratio: {results['walkable_area_ratio']:.4f} ({walkable_pixels}/{room_pixels} pixels)")
        else:
            results["walkable_area_ratio"] = 0.0

        if len(rooms) > 1:
            results["rooms_connected"] = _check_connectivity(walkable_map, rooms, scale, origin, IMAGE_SIZE)
            # print(f"Rooms Connected: {results['rooms_connected']}")
        elif len(rooms) == 1:
            results["rooms_connected"] = True
            # print(f"Rooms Connected: True (Single room scenario)")

    except FileNotFoundError:
        # print(f"Error: Scene file not found at {scene_file_path}")
        pass
    except Exception as e:
        # print(f"Error processing navigation validity for {scene_file_path}: {e}")
        pass

    return results["walkable_area_ratio"]

def _parse_scene_layout_for_walkable(scene_data):
    parsed_rooms = []
    parsed_objects = []
    parsed_doors = []

    for room_info in scene_data.get("rooms", []):
        floor_poly = room_info.get("floorPolygon")
        if floor_poly and len(floor_poly) > 2:
            vertices_2d = np.array([[p['x'], p['z']] for p in floor_poly])
            parsed_rooms.append({"id": room_info.get("id", "unknown_room"), "vertices": vertices_2d})
        else:
            # print(f"Warning: Room {room_info.get('id')} has invalid floorPolygon.")
            pass

    for obj_info in scene_data.get("objects", []):
        category = obj_info.get("object_name", "unknown").split('-')[0].replace('_', ' ')
        position = obj_info.get("position")
        rotation = obj_info.get("rotation")
        asset_id = obj_info.get("assetId")

        if position and rotation and asset_id:
            try:
                bbox_size = get_bbox_dims(obj_info, OBJAVERSE_ASSET_DIR)
                
                parsed_objects.append({
                    "id": obj_info.get("id"),
                    "category": category,
                    "position": [position['x'], position['y'], position['z']],
                    "rotation_y_deg": rotation.get('y', 0),
                    "size": bbox_size
                })
            except Exception as e:
                # print(f"Warning: Skipping object {obj_info.get('id')} ({asset_id}). Reason: {e}")
                pass

    for door_info in scene_data.get("doors", []):
        segment = door_info.get("doorSegment")
        if segment and len(segment) == 2:
            parsed_doors.append({"points": np.array(segment), "id": door_info.get("id")})

    return parsed_rooms, parsed_objects, parsed_doors

def _map_world_to_image(points_world, scale, origin, image_size):
    points_world_np = np.asarray(points_world)
    if points_world_np.ndim == 1:
        points_world_np = points_world_np.reshape(1, -1)
        
    points_img = ((points_world_np - origin) * (image_size / 2 / scale)) + (image_size / 2)
    return np.round(points_img[:, ::-1]).astype(int)

def _get_object_footprint_poly(obj_info):
    size = obj_info.get("size")
    position = obj_info.get("position")
    rotation_y_deg = obj_info.get("rotation_y_deg")

    if size is None or position is None or rotation_y_deg is None:
        return None

    size_x, _, size_z = size
    half_x, half_z = size_x / 2.0, size_z / 2.0
    corners_local = np.array([
        [-half_x, -half_z], [ half_x, -half_z], [ half_x,  half_z], [-half_x,  half_z]
    ])

    angle_rad = math.radians(rotation_y_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rot_matrix_2d = np.array([[ cos_a, -sin_a],
                            [ sin_a,  cos_a]])

    world_center_xz = np.array([position[0], position[2]])
    corners_world = (rot_matrix_2d @ corners_local.T).T + world_center_xz
    return corners_world

def _generate_walkable_map(rooms, objects, doors, image_size, robot_radius_m):
    if not rooms: return None, None, 1.0, np.array([0,0])

    all_vertices = np.concatenate([room['vertices'] for room in rooms])
    min_coords = np.min(all_vertices, axis=0)
    max_coords = np.max(all_vertices, axis=0)
    center = (min_coords + max_coords) / 2.0
    max_dist = np.max(np.sqrt(np.sum((all_vertices - center)**2, axis=1)))
    scale = max_dist + 0.2
    if scale < 1e-6: scale = 1.0
    origin = center

    room_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    obstacle_map = np.zeros((image_size, image_size), dtype=np.uint8)

    for room in rooms:
        img_vertices = _map_world_to_image(room['vertices'], scale, origin, image_size)
        img_vertices[:, 0] = np.clip(img_vertices[:, 0], 0, image_size - 1)
        img_vertices[:, 1] = np.clip(img_vertices[:, 1], 0, image_size - 1)
        cv2.fillPoly(room_mask, [img_vertices], color=1)

    obstacle_height_threshold = 1.5
    for obj in objects:
        position = obj.get("position")
        
        if position and position[1] < obstacle_height_threshold:
            footprint_poly = _get_object_footprint_poly(obj)
            if footprint_poly is not None:
                img_footprint = _map_world_to_image(footprint_poly, scale, origin, image_size)
                img_footprint[:, 0] = np.clip(img_footprint[:, 0], 0, image_size - 1)
                img_footprint[:, 1] = np.clip(img_footprint[:, 1], 0, image_size - 1)
                cv2.fillPoly(obstacle_map, [img_footprint], color=1)

    pixel_per_meter = image_size / (scale * 2)
    robot_radius_pixels = int(robot_radius_m * pixel_per_meter)
    kernel_size = max(1, (robot_radius_pixels * 2) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_obstacles = cv2.dilate(obstacle_map, kernel)

    walkable_map = np.where((room_mask > 0) & (dilated_obstacles == 0), 255, 0).astype(np.uint8)

    door_thickness_pixels = max(1, kernel_size)
    for door in doors:
        door_points = np.array(door["points"])
        if door_points.shape == (2, 2):
            p1_img = _map_world_to_image(door_points[0], scale, origin, image_size)[0]
            p2_img = _map_world_to_image(door_points[1], scale, origin, image_size)[0]
            cv2.line(walkable_map, tuple(p1_img), tuple(p2_img), color=255, thickness=door_thickness_pixels)
        else:
            # print(f"Warning: Invalid door points format for door {door.get('id')}. Skipping door drawing.")
            pass


    return walkable_map, room_mask, scale, origin


def _check_connectivity(walkable_map, rooms, scale, origin, image_size):
    if len(rooms) <= 1: return True

    room_centers_img = []
    for i, room in enumerate(rooms):
        center_world = np.mean(room['vertices'], axis=0)
        center_img_col_row = _map_world_to_image(center_world, scale, origin, image_size)[0]
        r, c = center_img_col_row[1], center_img_col_row[0]

        if 0 <= r < image_size and 0 <= c < image_size and walkable_map[r, c] == 255:
            room_centers_img.append((r, c))
        else:
            room_mask_single = np.zeros_like(walkable_map)
            img_vertices = _map_world_to_image(room['vertices'], scale, origin, image_size)
            cv2.fillPoly(room_mask_single, [img_vertices], color=1)
            walkable_in_room = np.argwhere((room_mask_single > 0) & (walkable_map == 255))
            
            if len(walkable_in_room) > 0:
                found_point = tuple(walkable_in_room[0])
                room_centers_img.append(found_point)
            else:
                # print(f"ERROR: Cannot find any walkable point inside room {room.get('id')}. Assuming disconnected.")
                return False

    if len(room_centers_img) < len(rooms): return False

    cost_map = np.where(walkable_map == 255, 1.0, np.inf)

    start_node = room_centers_img[0]
    for i in range(1, len(room_centers_img)):
        end_node = room_centers_img[i]
        try:
            path, cost = route_through_array(cost_map, start=start_node, end=end_node, fully_connected=True)
            if not path or cost == np.inf:
                # print(f"Connectivity check failed: No path from room center {start_node} to {end_node}")
                return False
        except Exception as e:
            # print(f"Error during path finding (start={start_node}, end={end_node}): {e}")
            return False

    return True