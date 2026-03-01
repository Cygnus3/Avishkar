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
    # Path following state
    # -------------------------------------------------
    target = 0
    init_vel = 0.2
    L_d = 1.2 * init_vel + 0.8

    # -------------------------------------------------
    # Obstacle avoidance state
    # -------------------------------------------------
    state = "TO_GOAL"          # TO_GOAL | WALL_FOLLOW
    arc_direction = 1          # +1 left | -1 right
    entry_dist_to_goal = float('inf')
    stuck_timer = 0
    STUCK_LIMIT = 150          # steps before declaring stuck
    prev_dist_to_goal = float('inf')
    no_progress_timer = 0
    NO_PROGRESS_LIMIT = 200    # steps without progress before override

    # -------------------------------------------------
    # Safety thresholds
    # -------------------------------------------------
    STOP_DISTANCE = 0.8
    SLOW_DISTANCE = 2.0
    SAFE_LATERAL  = 0.8  

    # -------------------------------------------------
    # Main simulation loop
    # -------------------------------------------------

    path = []

    try:
        with open("path.csv", "r") as file:
            values = csv.DictReader(file)
            for rows in values:
                px = float(rows['x'])
                py = float(rows['y'])
                path.append((px, py))
    except FileNotFoundError:
        print("Error: path.csv not found!")

    while plt.fignum_exists(fig.number):

        real_x, real_y, real_theta = robot.get_ground_truth()
        ideal_x, ideal_y, ideal_theta = robot.get_odometry()
        lidar_ranges, lidar_points, lidar_rays, lidar_hits = lidar.get_scan((real_x, real_y, real_theta))

        if MODE == "AUTO":

            # -------------------------------------------------
            # Lidar zones (used in both states)
            # -------------------------------------------------
            forward_indices = list(range(0, 4)) + list(range(33, 36))
            left_indices    = list(range(4, 14))
            right_indices   = list(range(23, 33))

            min_forward_dist = min(lidar_ranges[i] for i in forward_indices)
            left_dist        = min(lidar_ranges[i] for i in left_indices)
            right_dist       = min(lidar_ranges[i] for i in right_indices)

            # -------------------------------------------------
            # Distance to final waypoint (always valid)
            # -------------------------------------------------
            final_dist = math.sqrt(
                (path[-1][0] - ideal_x)**2 +
                (path[-1][1] - ideal_y)**2
            )

            # -------------------------------------------------
            # Exit condition checks 
            # -------------------------------------------------
            forward_clear = all(
                lidar_ranges[i] > SLOW_DISTANCE
                for i in forward_indices
            )

            left_clear = any(
                all(lidar_ranges[j] > SLOW_DISTANCE
                    for j in range(i, i + 4))
                for i in range(4, 11)
            )

            right_clear = any(
                all(lidar_ranges[j] > SLOW_DISTANCE
                    for j in range(i, i + 4))
                for i in range(23, 28)
            )

            path_clear = forward_clear or left_clear or right_clear

            # -------------------------------------------------
            # Convert lidar_hits to robot frame
            # only keep forward hits within ±45°
            # -------------------------------------------------
            forward_hits = []
            for (hx, hy) in lidar_hits:
                dx_h = hx - ideal_x
                dy_h = hy - ideal_y
                hx_robot =  dx_h * math.cos(ideal_theta) + dy_h * math.sin(ideal_theta)
                hy_robot = -dx_h * math.sin(ideal_theta) + dy_h * math.cos(ideal_theta)
                angle = math.atan2(hy_robot, hx_robot)
                if hx_robot > 0 and abs(angle) < math.pi / 4:
                    forward_hits.append((hx_robot, hy_robot))

            # =================================================
            # STATE: TO_GOAL — pure pursuit + collision prevent
            # =================================================
            if state == "TO_GOAL":

                # pure pursuit lookahead
                target_pt = path[-1]
                for i in range(target, len(path)):
                    dist = math.sqrt((path[i][0] - ideal_x)**2 +(path[i][1] - ideal_y)**2)
                    if dist >= L_d:
                        target_pt = path[i]
                        target = i
                        break

                dx = target_pt[0] - ideal_x
                dy = target_pt[1] - ideal_y
                delta_y = -dx * math.sin(ideal_theta) + dy * math.cos(ideal_theta)

                curvature = 0.0

                if final_dist < 0.1:
                    v = 0.0
                    w = 0.0
                

                else:
                    # pure pursuit commands
                    curvature = (2 * delta_y) / (L_d**2)
                    v = 1 / (1 + 1.2 * abs(curvature))
                    w = v * curvature
                    L_d = 1.2 * v + 0.8

                    # Question 6 (part 3): collision prevention

                    if min_forward_dist < STOP_DISTANCE:
                        v = 0.0
                        w = 0.0

                    elif min_forward_dist < SLOW_DISTANCE:
                        scale = (min_forward_dist - STOP_DISTANCE) / (SLOW_DISTANCE - STOP_DISTANCE)
                        scale = max(0.0, min(1.0, scale))
                        v = v * scale
                        w = v * curvature
                        L_d = 1.2 * v + 0.8

                    # Question 6 (part 4) : obstacle avoidance   

                    if min_forward_dist < STOP_DISTANCE:
                        state = "WALL_FOLLOW"
                        entry_dist_to_goal = final_dist
                        stuck_timer = 0
                        no_progress_timer = 0
                        prev_dist_to_goal = final_dist

                        # decide arc direction once on entry (turn towards side with more space)
                        if left_dist >= right_dist:
                            arc_direction = 1    # turn left
                        else:
                            arc_direction = -1   # turn right

            # choosing center and forming circular arc   

            elif state == "WALL_FOLLOW":

                if forward_hits:
                    # closest forward hit point
                    closest = min(
                        forward_hits,
                        key=lambda h: math.sqrt(h[0]**2 + h[1]**2)
                    )
                    hx_robot, hy_robot = closest

                    # lateral error from desired safe distance
                    lateral_error = SAFE_LATERAL - abs(hy_robot)

                    # smooth arc steering proportional to lateral error
                    v = 0.4
                    w = arc_direction * 1.5 * lateral_error

                    # clamp w to prevent spinning
                    w = max(min(w, 2.0), -2.0)

                    # stuck detection
                    # if robot barely moving increment stuck timer
                    stuck_timer += 1
                    if stuck_timer > STUCK_LIMIT:
                        # reverse slightly then continue arc
                        v = -0.2
                        w = arc_direction * 1.5
                        stuck_timer = 0

                else:
                    # no forward hits
                    # keep moving forward slowly 
                    v = 0.3
                    w = arc_direction * 0.5
                    stuck_timer = 0

                # no progress detection
                # if not getting closer to goal
                if final_dist >= prev_dist_to_goal:
                    no_progress_timer += 1
                else:
                    no_progress_timer = 0
                    prev_dist_to_goal = final_dist

                if no_progress_timer > NO_PROGRESS_LIMIT:
                    # flip arc direction and try other way
                    arc_direction *= -1
                    no_progress_timer = 0
                    prev_dist_to_goal = final_dist
                    print("No progress! Flipping arc direction")

                # exit condition: your lidar idea + progress made
                if path_clear and final_dist < entry_dist_to_goal:

                    # find nearest AHEAD waypoint to resume from
                    for i in range(target, len(path)):
                        dx_p = path[i][0] - ideal_x
                        dy_p = path[i][1] - ideal_y
                        forward_component = (
                            dx_p * math.cos(ideal_theta) +
                            dy_p * math.sin(ideal_theta)
                        )
                        if forward_component > 0:
                            target = i
                            break

                    state = "TO_GOAL"
                    stuck_timer = 0
                    no_progress_timer = 0
                    prev_dist_to_goal = float('inf')
                    print("Path clear! Resuming pure pursuit")

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