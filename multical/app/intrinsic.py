from multical.image.detect import common_image_size
from multical.threading import map_lists
from multical.io.logging import setup_logging
from multical.io.logging import info

from .arguments import add_intrinsic_args, parse_with

from structs.struct import map_none, map_list
from multical import image

from .calibrate import get_paths, init_boards
from structs.numpy import struct, shape

def calibrate_intrinsic(args):
 
    paths = get_paths(args)

    setup_logging(args.log_level, [], log_file=paths.log_file)
    info(args) 

    boards = init_boards(args)

    cameras = map_none(str.split, args.cameras, ",")
    camera_paths = image.find.find_cameras(args.image_path, cameras, args.camera_pattern)

    camera_names, image_files = image.find.find_images_unmatched(camera_paths)

    image_counts = {k:len(files) for k, files in zip(camera_names, image_files)}
    info("Found camera directories with images {}".format(image_counts))


    info("Loading images..")
    images = image.detect.load_images(image_files,  prefix=args.image_path, j=args.j)
    image_sizes = map_list(common_image_size, images)

    info({k:image_size for k, image_size in zip(camera_names, image_sizes)})

    


    # ws.find_images_matching(args.image_path, cameras, args.camera_pattern, master = args.master)
 

    # ws.load_images(j=args.j)
    # ws.detect_boards(boards, cache_file=paths.detection_cache, 
    #   load_cache=not args.no_cache, j=args.j)
    
    # ws.calibrate_single(args.distortion_model, fix_aspect=args.fix_aspect, 
    #   has_skew=args.allow_skew, max_images=args.intrinsic_images)

    # motion_model = None
    # if args.motion_model == "rolling":
    #   motion_model = RollingFrames
    # elif args.motion_model == "static":
    #   motion_model = StaticFrames
    # else:
    #   assert False, f"unknown motion model {args.motion_model}, (static|rolling)"

    # ws.initialise_poses(motion_model=motion_model)
    # return ws




if __name__ == '__main__':
    args = parse_with(add_intrinsic_args)
    calibrate_intrinsic(args)
