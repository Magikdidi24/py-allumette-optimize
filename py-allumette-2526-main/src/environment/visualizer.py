# environment/visualizer.py [FULLY PROVIDED TO STUDENTS]
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

class RaceVisualizer:
    def __init__(self, track):
        self.track = track

    def animate(self, state_history, dt=0.03, save_path=None, title="SAE Autonomous Racing"):
        fig, ax = plt.subplots(figsize=(11, 9), facecolor='#2e5c1e')
        ax.set_facecolor('#2e5c1e')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, color='white', pad=10, fontsize=13)

        # Asphalt surface
        outer_pts = np.column_stack([self.track.x_outer, self.track.y_outer])
        inner_pts = np.column_stack([self.track.x_inner, self.track.y_inner])[::-1]
        track_poly = np.vstack([outer_pts, inner_pts])
        ax.add_patch(patches.Polygon(track_poly, closed=True,
                                     facecolor='#3a3a3a', edgecolor='none', zorder=1))

        # Kerbs (red base + white dashes)
        for side_x, side_y in [(self.track.x_inner, self.track.y_inner),
                               (self.track.x_outer, self.track.y_outer)]:
            ax.plot(side_x, side_y, color='#c1272d', linewidth=5, zorder=2)
            ax.plot(side_x, side_y, color='white',   linewidth=5,
                    linestyle='--', dashes=(2, 2), zorder=3)

        # Centerline dashes
        ax.plot(self.track.x_center, self.track.y_center,
                color='#aaaaaa', linewidth=0.8, linestyle='--', dashes=(6, 6), zorder=4)

        # Start / finish line
        ax.plot([self.track.x_inner[0], self.track.x_outer[0]],
                [self.track.y_inner[0], self.track.y_outer[0]],
                color='white', linewidth=4, zorder=5)
        ax.text(self.track.x_center[0] + 0.2, self.track.y_center[0] + 0.2,
                "S/F", color='white', fontsize=8, zorder=5)

        # Car patches (EPITA colors) - Using standard Rectangle to support set_angle!
        car_l, car_w = 0.43, 0.18
        car_body = patches.Rectangle((0, 0), car_l, car_w,
                                     fc='#00ACE9', ec='white',
                                     linewidth=1.0, zorder=10)
        
        car_nose = patches.Rectangle((0, 0), car_l * 0.22, car_w * 0.75,
                                     fc='#B80E80', ec='none', zorder=11)
        
        ax.add_patch(car_body)
        ax.add_patch(car_nose)

        # Trajectory trail
        trail_line, = ax.plot([], [], color='#00ACE9', lw=2.0, alpha=0.55, zorder=6)
        
        # Target checkpoints tracking visualization
        checkpoints_points, = ax.plot([], [], 'o', color='#FFD700', markersize=3, alpha=0.8, zorder=7)

        # HUD: lap time counter
        hud_text = ax.text(0.02, 0.96, '', transform=ax.transAxes,
                           color='white', fontsize=11, va='top',
                           fontfamily='monospace', zorder=20)

        def update(frame):
            x, y, phi = state_history[frame, 0], state_history[frame, 1], state_history[frame, 2]

            # Body: place so (x, y) is the CoG
            bx = x - (car_l/2)*np.cos(phi) + (car_w/2)*np.sin(phi)
            by = y - (car_l/2)*np.sin(phi) - (car_w/2)*np.cos(phi)
            
            car_body.set_xy((bx, by))
            car_body.set_angle(np.degrees(phi))

            # Nose: front-center of car
            nose_offset_l = car_l/2 - car_l*0.22
            nose_offset_w = car_w*0.125
            nx_ = x + nose_offset_l*np.cos(phi) + nose_offset_w*np.sin(phi)
            ny_ = y + nose_offset_l*np.sin(phi) - nose_offset_w*np.cos(phi)
            
            car_nose.set_xy((nx_, ny_))
            car_nose.set_angle(np.degrees(phi))

            # --- TRAIL LOGIC FIX ---
            # Default trail length is 60 frames
            s = max(0, frame - 60)
            
            # Extract the history slice for the trail
            xs = state_history[s:frame+1, 0]
            ys = state_history[s:frame+1, 1]
            
            # Draw the cleaned trail
            trail_line.set_data(xs, ys)

            # Ploting Checkpoints dynamically based on tracked positions
            idx = self.track.get_closest_idx(state_history[frame])
            targets_x, targets_y = [], []
            for k in range(0, 40, 2):  # Assuming N=40 Lookahead
                target_idx = (idx + 1 + k) % len(self.track.x_center)
                targets_x.append(self.track.x_center[target_idx])
                targets_y.append(self.track.y_center[target_idx])
            checkpoints_points.set_data(targets_x, targets_y)

            # HUD
            elapsed = frame * dt
            hud_text.set_text(f"t = {elapsed:5.2f} s   frame {frame:4d}")

            return car_body, car_nose, trail_line, checkpoints_points, hud_text

        ani = animation.FuncAnimation(fig, update, frames=len(state_history),
                                      interval=int(dt * 1000), blit=True)

        plt.tight_layout()
        if save_path:
            ani.save(save_path, writer='ffmpeg', fps=int(1/dt))
            print(f"Saved to {save_path}")
        else:
            plt.show()