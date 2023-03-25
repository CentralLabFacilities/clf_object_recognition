#include "clf_object_recognition_3d/model_provider.h"


ModelProvider::ModelProvider(ros::NodeHandle nh) {
    model_sub = nh.subscribe("/ecwm/SimpleGeometryViz/models", 1, &ModelProvider::ModelCallback, this);
}

void ModelProvider::ModelCallback(const ecwm_msgs::ModelVisArrayPtr& msg) {
    latest_models = *msg;
}

std::string ModelProvider::GetModelPath(std::string model_name) {
    for(auto model : latest_models.models) {
        if(model.type != model_name) continue;
        if(model.colliders.size() > 0) {
            auto maybe_mesh = model.colliders.front();
            if(maybe_mesh.type == visualization_msgs::Marker::MESH_RESOURCE) {
                return maybe_mesh.mesh_resource;
            } else {
                //have to create shape from marker, no path
            }
        }
        if(model.geometries.size() > 0) {
            auto maybe_mesh = model.geometries.front();
            ROS_INFO_STREAM_NAMED("ModelProvider", "model: '" << model_name << "' has no colliders, using geometry");
            if(maybe_mesh.type == visualization_msgs::Marker::MESH_RESOURCE) {
                return maybe_mesh.mesh_resource;
            } else {
                //have to create shape from marker, no path
            }
        }
        ROS_WARN_STREAM_NAMED("ModelProvider", "model: '" << model_name << "' has neither colliders nor geometries");
        return "";
    }
    ROS_WARN_STREAM_NAMED("ModelProvider", "model: '" << model_name << "' not found");
    return "";
}


std::string ModelProvider::IDtoObject(int id) {
    //cache labels
    static std::map<int, std::string> objectLabels;
    if (objectLabels.count(id) == 1)
    {
        return objectLabels[id];
    }else
    {
        std::ostringstream oss;
        oss << "/object_labels/" << id;

        std::string object_label;
        if (ros::param::get(oss.str(), object_label))
        {
            // ensure no spaces in object_label' ' to '_'
            std::replace(object_label.begin(), object_label.end(), ' ', '_');
            objectLabels[id] = object_label;
            return object_label;
        }
        else
        {
            // unknown return id
            ROS_WARN_STREAM_NAMED("ModelProvider", "unkown object_label: " << id);
            return std::to_string(id);
        }
    }
}
std::string ModelProvider::IDtoModel(int id) {
    //cache models
    static std::map<int, std::string> modelTypes;
    if (modelTypes.count(id) == 1)
    {
        return modelTypes[id];
    }else
    {
        std::ostringstream oss;
        oss << "/object_models/" << id;

        std::string model_type;
        if (ros::param::get(oss.str(), model_type))
        {
            modelTypes[id] = model_type;
            return model_type;
        }
        else
        {
            // unknown return id
            ROS_WARN_STREAM_NAMED("ModelProvider", "unkown model_type: " << id);
            return std::to_string(id);
        }
    }
}