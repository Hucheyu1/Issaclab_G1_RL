#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }

    param::auto_mimic_enabled = vm.count("auto_mimic") > 0;
    param::auto_mimic_state = vm["mimic_state"].as<std::string>();
    param::auto_mimic_fixstand_duration = vm["fixstand_duration"].as<double>();
    param::mimic_override_state = param::auto_mimic_enabled ? param::auto_mimic_state : vm["start_state"].as<std::string>();
    param::mimic_motion_file = vm["mimic_motion_file"].as<std::string>();
    param::mimic_fps = vm["mimic_fps"].as<float>();
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    if(vm.count("auto_mimic"))
    {
        const auto mimic_state = param::auto_mimic_state;
        const auto fixstand_duration = param::auto_mimic_fixstand_duration;
        fsm->scheduleTransition(mimic_state, fixstand_duration);
        fsm->start("FixStand");
        std::cout << "Auto sequence: FixStand (" << fixstand_duration << "s) -> " << mimic_state << ".\n";
    }
    else
    {
        fsm->start(vm["start_state"].as<std::string>());
        std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
        std::cout << "And then press [R1 + X] to start controlling the robot.\n";
        std::cout << "Without a joystick, use --auto_mimic.\n";
    }


    while (true)
    {
        sleep(1);
    }
    
    return 0;
}
