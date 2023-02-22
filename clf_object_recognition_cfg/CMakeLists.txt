cmake_minimum_required(VERSION 3.0)
project(clf_object_recognition_cfg)

find_package(catkin REQUIRED
    COMPONENTS
    dynamic_reconfigure
    )

## Gencfg
generate_dynamic_reconfigure_options(
  cfg/Yolox.cfg
  cfg/Detect3d.cfg
)

catkin_package()
