from robot import Robot
from sensors.lidar import LidarScan
import matplotlib.pyplot as plt
import math
import csv

# -----------------------------------------------------
# Runtime modes & visualization toggles
# -----------------------------------------------------
MODE = "MANUAL"        # MANUAL | AUTO
SHOW_LIDAR = True
SHOW_ODOM = True


# -----------------------------------------------------
# Control commands (shared state)
# -----------------------------------------------------
v = 0.0   # linear velocity [m/s]
w = 0.0   # angular velocity [rad/s]


def on_key(event):
    """Keyboard control & visualization toggles."""
    global v, w, MODE, SHOW_LIDAR, SHOW_ODOM

    # --- visualization toggles ---
    if event.key == 'o':
        SHOW_ODOM = not SHOW_ODOM
        print(f"Odometry visualization: {'ON' if SHOW_ODOM else 'OFF'}")
        return

    if event.key == 'l':
        SHOW_LIDAR = not SHOW_LIDAR
        print(f"LiDAR visualization: {'ON' if SHOW_LIDAR else 'OFF'}")
        return

    # --- mode switching ---
    if event.key == 'm':
        MODE = "MANUAL"
        v = 0.0
        w = 0.0
        print("Switched to MANUAL mode")
        return

    if event.key == 'a':
        MODE = "AUTO"
        print("Switched to AUTO mode")
        return

    # --- manual control ---
    if MODE != "MANUAL":
        return

    if event.key == 'up':
        v += 1.5
    elif event.key == 'down':
        v -= 1.5
    elif event.key == 'left':
        w += 2.0
    elif event.key == 'right':
        w -= 2.0
    elif event.key == ' ':
        v = 0.0
        w = 0.0

    # clamp commands
    v = max(min(v, 6.0), -6.0)
    w = max(min(w, 6.0), -6.0)


if __name__ == "__main__":

    lidar = LidarScan(max_range=4.0)
    robot = Robot()

    plt.close('all')
    fig = plt.figure(num=2)
    fig.canvas.manager.set_window_title("Autonomy Debug View")
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=False)

    dt = 0.01 

    # -------------------------------------------------
    # Main simulation loop
    target = 0 

    path = []
    try : 
        with open("path.csv", "r") as file :
            values = csv.DictReader(file)
            for rows in values:
             real_x = float(rows['x'])
             real_y = float(rows['y'])
             path.append((real_x,real_y))

    except FileNotFoundError:
        print("Error: path.csv not found!")
    # -------------------------------------------------
    while plt.fignum_exists(fig.number):

        # ground truth pose
        real_x, real_y, real_theta = robot.get_ground_truth()
        # odometry estimate
        ideal_x, ideal_y, ideal_theta = robot.get_odometry()
        # LiDAR scan 
        lidar_ranges, lidar_points, lidar_rays, lidar_hits = lidar.get_scan((real_x, real_y, real_theta))

        if MODE == "AUTO":

            # ---------------------------------------------
            # write your autonomous code here!!!!!!!!!!!!!
            # ---------------------------------------------
            init_vel = 0.2
            L_d = 1.2*init_vel + 0.6
                
            
            target_pt = path[-1]
            for i in range(target, len(path)):
                dist = math.sqrt((path[i][0] - ideal_x)**2 + (path[i][1] - ideal_y)**2)
                if dist >= L_d:
                    target_pt = path[i]
                    target = i
                    break
                   
                   
            dx = target_pt[0] - ideal_x
            dy = target_pt[1] - ideal_y

            delta_y = -dx * math.sin(ideal_theta) + dy * math.cos(ideal_theta)

            final_dist = math.sqrt( (path[-1][0] - ideal_x)**2 + (path[-1][1] - ideal_y)**2)

            if final_dist < 0.1:
                 v = 0.0
                 w = 0.0
            else:
                curvature = (2 * delta_y) / (L_d**2)

                v = 1 / (1 + 1.2 * abs(curvature))
                w = v*curvature

        #--------------------lidar obstacle detection------------------ 
        
            

            # ---------------------------------------------
            # Allowed inputs:
            #   - real_x, real_y, real_theta
            #   - ideal_x, ideal_y, ideal_theta (odometry you have to use for logic)
            #   - lidar_ranges (lidar data you have to use for logic, array of length 36 corresponding to 36 beams)
            #
            # Required outputs:
            #   - v, w (linear and angular velocity commands)


            # ---------------------------------------------
            # don't edit below this line (visualization & robot stepping)
            # ---------------------------------------------
        robot.step(
            lidar_points,
            lidar_rays,
            lidar_hits,
            v,
            w,
            dt,
            show_lidar=SHOW_LIDAR,
            show_odom=SHOW_ODOM
        )

        plt.pause(dt)


