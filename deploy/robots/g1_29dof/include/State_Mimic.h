#pragma once

#include "FSM/State_RLBase.h"
#include <cmath>
#include <stdexcept>

class State_Mimic : public FSMState
{
public:
    State_Mimic(int state_mode, std::string state_string);

    void enter();

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

    class MotionLoader_;

    static std::shared_ptr<MotionLoader_> motion; // for obs computation
private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<MotionLoader_> motion_; // for saving

    std::thread policy_thread;
    bool policy_thread_running = false;
    std::array<float, 2> time_range_;
};

class State_Mimic::MotionLoader_
{
public:
    MotionLoader_(std::string motion_file, float fps)
    : dt(1.0f / fps)
    {
        auto data = isaaclab::load_csv(motion_file);
        
        num_frames = data.size();
        duration = num_frames * dt;
        
        constexpr int root_cols = 7;
        constexpr int dof_cols = 29;
        constexpr int base_cols = root_cols + dof_cols;
        for(int i(0); i < num_frames; ++i)
        {
            if(data[i].size() < base_cols) {
                throw std::runtime_error("Motion CSV must contain at least 36 columns: root(7) + joints(29)");
            }
            root_positions.push_back(Eigen::VectorXf::Map(data[i].data(), 3));
            root_quaternions.push_back(Eigen::Quaternionf(data[i][6],data[i][3], data[i][4], data[i][5]));
            dof_positions.push_back(Eigen::VectorXf::Map(data[i].data() + root_cols, dof_cols));
            const int extra_cols = data[i].size() - base_cols;
            if(extra_cols > 0) {
                extra_features.push_back(Eigen::VectorXf::Map(data[i].data() + base_cols, extra_cols));
            } else {
                extra_features.emplace_back();
            }
        }
        dof_velocities = _comupte_raw_derivative(dof_positions);

        update(0.0f);
    }

    void update(float time) 
    {
        float phase = std::clamp(time / duration, 0.0f, 1.0f);
        index_0_ = std::round(phase * (num_frames - 1));
        index_1_ = std::min(index_0_ + 1, num_frames - 1);
        blend_ = std::round((time - index_0_ * dt) / dt * 1e5f) / 1e5f;
    }

    void reset(const isaaclab::ArticulationData & data, float t = 0.0f)
    {
        update(t);
        auto init_to_anchor = isaaclab::yawQuaternion(this->root_quaternion()).toRotationMatrix();
        auto world_to_anchor = isaaclab::yawQuaternion(data.root_quat_w).toRotationMatrix();
        world_to_init_ = world_to_anchor * init_to_anchor.transpose();
    }

    Eigen::VectorXf joint_pos() {
        return dof_positions[index_0_] * (1 - blend_) + dof_positions[index_1_] * blend_;
    }

    Eigen::VectorXf joint_pos_at_offset(int offset) {
        return dof_positions[frame_index(offset)];
    }

    Eigen::VectorXf root_position() {
        return root_positions[index_0_] * (1 - blend_) + root_positions[index_1_] * blend_;
    }

    Eigen::VectorXf joint_vel() {
        return dof_velocities[index_0_] * (1 - blend_) + dof_velocities[index_1_] * blend_;
    }

    Eigen::VectorXf extra_feature_at_offset(int offset) {
        if(extra_features.empty() || extra_features[0].size() == 0) {
            throw std::runtime_error("Requested extended motion command, but motion CSV has no extra feature columns.");
        }
        return extra_features[frame_index(offset)];
    }

    Eigen::Quaternionf root_quaternion() {
        return root_quaternions[index_0_].slerp(blend_, root_quaternions[index_1_]);
    }

    float dt;
    int num_frames;
    float duration;

    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;
    std::vector<Eigen::VectorXf> extra_features;

    Eigen::Matrix3f world_to_init_;
private:
    int index_0_;
    int index_1_;
    float blend_;

    int frame_index(int offset) const
    {
        return std::min(index_0_ + offset, num_frames - 1);
    }

    std::vector<Eigen::VectorXf> _comupte_raw_derivative(const std::vector<Eigen::VectorXf>& data)
    {
        std::vector<Eigen::VectorXf> derivative;
        for(int i = 0; i < data.size() - 1; ++i) {
            derivative.push_back((data[i + 1] - data[i]) / dt);
        }
        derivative.push_back(derivative.back());
        return derivative;
    }
};

REGISTER_FSM(State_Mimic)
