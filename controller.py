import do_mpc
from casadi import SX, vertcat
import numpy as np

class ModelPredictiveController:
    
    def __init__(self, goal_position) -> None:
        
        self.model = self.drone_dynamics()
        self.controller = do_mpc.controller.MPC(self.model)
        self.set_pamrater()
        
        self.state = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0}
        self.state = np.array([self.state[key] for key in self.model.x.keys()])
        print(self.state)
                
        self.set_objective(goal_position)
        self.set_constraints()
        
        self.controller.setup()
        self.set_simulator()
        

    def set_objective(self, goal_position=[1.0, 1.0, 1.0]):
    
        # 목표 지점
        target_x, target_y, target_z = goal_position

        # 목적 함수 설정 (목표 지점에 도달하는 것)
        mterm = (self.model.x['x']-target_x)**2 + (self.model.x['y']-target_y)**2
        lterm = mterm  # 단기 목적 함수

        self.controller.set_objective(mterm=mterm, lterm=lterm)
        print(f"target: {target_x}, {target_y}, {target_z}")
        
    def set_constraints(self):
        
        # MPC Constraints
        self.controller.bounds['lower','_u','vx'] = -1.0
        self.controller.bounds['upper','_u','vx'] = 1.0
        self.controller.bounds['lower','_u','vy'] = -1.0
        self.controller.bounds['upper','_u','vy'] = 1.0
        self.controller.bounds['lower','_u','vz'] = -1.0
        self.controller.bounds['upper','_u','vz'] = 1.0

        

    def set_pamrater(self):

        # MPC parameters
        setup_mpc = {
            'n_horizon': 10,
            't_step': 0.1,
            'n_robust': 1,
            'store_full_solution': True,
        }
        self.controller.set_param(**setup_mpc)

    def drone_dynamics(self):
        model = do_mpc.model.Model('continuous')

        # 상태 변수: 위치 (x, y, z) 및 속도 (vx, vy, vz)
        x = model.set_variable('_x', 'x')
        y = model.set_variable('_x', 'y')
        z = model.set_variable('_x', 'z')
        vx = model.set_variable('_x', 'vx')
        vy = model.set_variable('_x', 'vy')
        vz = model.set_variable('_x', 'vz')

        # 제어 입력: 속도 (vx, vy, vz)
        u_vx = model.set_variable('_u', 'vx')
        u_vy = model.set_variable('_u', 'vy')
        u_vz = model.set_variable('_u', 'vz')

        # 쿼드콥터의 비선형 동역학 모델
        f_x = vx
        f_y = vy
        f_z = vz

        f_vx = u_vx
        f_vy = u_vy
        f_vz = u_vz

        # 모델 방정식 설정
        model.set_rhs('x', f_x)
        model.set_rhs('y', f_y)
        model.set_rhs('z', f_z)
        model.set_rhs('vx', f_vx)
        model.set_rhs('vy', f_vy)
        model.set_rhs('vz', f_vz)

        model.setup()
        return model

    def set_simulator(self):
        self.simulator = do_mpc.simulator.Simulator(self.model)
        self.simulator.set_param(t_step=0.1)
        self.simulator.setup()

    def solve(self):
        
        control_input = self.controller.make_step(self.state)
        
        return control_input
        
        