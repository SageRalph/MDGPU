package mdgpu;

/*
 * "Physics" part of code adapted from Dan Schroeder's applet at:
 *     http://physics.weber.edu/schroeder/software/mdapplet.html
 */
import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;
import com.amd.aparapi.device.OpenCLDevice;
import java.awt.*;
import javax.swing.*;

public class MDGPU {

// Size of simulation
final static int N = 2000;   // Number of "atoms"
final static float BOX_WIDTH = 100.0f;

// Initial state - controls temperature of system
//final static float VELOCITY = 3.0f ;  // gaseous
final static float VELOCITY = 2.0f;  // gaseous/"liquid"
//final static float VELOCITY = 1.0f ;  // "crystalline"

final static float INIT_SEPARATION = 2.2f;  // in atomic radii

// Simulation
final static float DT = 0.01f;  // Time step

// Display
final static int WINDOW_SIZE = 800;
final static int DELAY = 0;
final static int OUTPUT_FREQ = 10;

// Physics constants
final static float ATOM_RADIUS = 0.5f;
final static float WALL_STIFFNESS = 500.0f;
final static float GRAVITY = 0.005f;
final static float FORCE_CUTOFF = 3.0f;

// Atom positions
static float[] x = new float[N];
static float[] y = new float[N];

// Atom velocities
static float[] vx = new float[N];
static float[] vy = new float[N];

// Atom accelerations
static float[] ax = new float[N];
static float[] ay = new float[N];

static MDKernel kernel; // APARAPI Kernel for execution on GPU or JTP
static final OpenCLDevice device = getDevice();
final static int maxTime = 10 * 1000; // 60 seconds

public static void main(String args[]) throws Exception {

    Display display = new Display();

    kernel = new MDKernel(x, y, ax, ay, FORCE_CUTOFF, N);
    kernel.execute(Range.create(device, 1)); // Force creation before timer starts

    // Run optimised sequential algorithm
    simulate(display, false);

    // Run parallel algorithm sequentially (one thread, run in loop)
    kernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
    simulate(display, true);

    // Run parallel algorithm in Java Thread Pool (one thread per core)
    kernel.setExecutionMode(Kernel.EXECUTION_MODE.JTP);
    simulate(display, true);
    // Run parallel algorithm on GPU via OpenCL (N threads)
    kernel.setExecutionMode(Kernel.EXECUTION_MODE.GPU);
    simulate(display, true);
}

static void simulate(Display display, boolean useKernel) throws InterruptedException {

    long startTime = System.currentTimeMillis();

    // Define initial state of atoms
    int sqrtN = (int) (Math.sqrt((float) N) + 0.5);
    float initSeparation = INIT_SEPARATION * ATOM_RADIUS;
    for (int i = 0; i < N; i++) {
        // lay out atoms regularly, so no overlap
        x[i] = (0.5f + i % sqrtN) * initSeparation;
        y[i] = (0.5f + i / sqrtN) * initSeparation;
        vx[i] = (float) ((2 * Math.random() - 1) * VELOCITY);
        vy[i] = (float) ((2 * Math.random() - 1) * VELOCITY);
    }

    int iter = 0;
    while (System.currentTimeMillis() - startTime < maxTime) {

        if (iter % OUTPUT_FREQ == 0) {
            //System.out.println("iter = " + iter + ", time = " + iter * DT);
            display.repaint();
            Thread.sleep(DELAY);
        }

        // Verlet integration:
        // http://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
        float dtOver2 = 0.5f * DT;
        float dtSquaredOver2 = 0.5f * DT * DT;
        for (int i = 0; i < N; i++) {
            x[i] += (vx[i] * DT) + (ax[i] * dtSquaredOver2);
            // update position
            y[i] += (vy[i] * DT) + (ay[i] * dtSquaredOver2);
            vx[i] += (ax[i] * dtOver2);  // update velocity halfway
            vy[i] += (ay[i] * dtOver2);
        }

        computeAccelerations(useKernel);

        for (int i = 0; i < N; i++) {
            vx[i] += (ax[i] * dtOver2);
            // finish updating velocity with new acceleration
            vy[i] += (ay[i] * dtOver2);
        }

        iter++;
    }

    String modeString = useKernel
            ? kernel.getExecutionMode().name()
            : "Sequential Optimised";
    System.out.println("\nExecution mode = " + modeString);
    System.out.println("Completed " + iter + " iterations in "
            + maxTime / 1000 + " seconds");
}

// Compute accelerations of all atoms from current positions:
static void computeAccelerations(boolean useKernel) {

    // first check for bounces off walls, and include gravity (if any):
    for (int i = 0; i < N; i++) {
        if (x[i] < ATOM_RADIUS) {
            ax[i] = WALL_STIFFNESS * (ATOM_RADIUS - x[i]);
        } else if (x[i] > (BOX_WIDTH - ATOM_RADIUS)) {
            ax[i] = WALL_STIFFNESS * (BOX_WIDTH - ATOM_RADIUS - x[i]);
        } else {
            ax[i] = 0.0f;
        }
        if (y[i] < ATOM_RADIUS) {
            ay[i] = (WALL_STIFFNESS * (ATOM_RADIUS - y[i]));
        } else if (y[i] > (BOX_WIDTH - ATOM_RADIUS)) {
            ay[i] = (WALL_STIFFNESS * (BOX_WIDTH - ATOM_RADIUS - y[i]));
        } else {
            ay[i] = 0;
        }
        ay[i] -= GRAVITY;
    }

    // Now compute interaction forces (Lennard-Jones potential).
    // This is where the program spends most of its time.
    if (useKernel) {
        computeInteractionForcesWithKernel();
    } else {
        computeInteractionForces();
    }

}

// KERNEL ALGORITHM
// Compute interaction forces (Lennard-Jones potential)
static void computeInteractionForcesWithKernel() {

//  // Uncomment for explicit buffer managment, 
//  // disabled due to low performance impact.
//    
//    kernel.setExplicit(true);
//    kernel.put(x);
//    kernel.put(y);
//    kernel.put(ax);
//    kernel.put(ay);
    if (kernel.getExecutionMode().equals(Kernel.EXECUTION_MODE.SEQ)) {
        // Set group size to 1 for sequential
        kernel.execute(Range.create(device, N, 1));
    } else {
        // Otherwise choose automatically
        kernel.execute(Range.create(device, N));
    }

//    kernel.get(ax);
//    kernel.get(ay);
}

// SEQUENTIAL ALGORITHM
// Compute interaction forces (Lennard-Jones potential)
static void computeInteractionForces() {

    float dx, dy;  // separations in x and y directions
    float dx2, dy2, rSquared, rSquaredInv, attract, repel, fOverR, fx, fy;

    float forceCutoff2 = FORCE_CUTOFF * FORCE_CUTOFF;

    // (NOTE: use of Newton's 3rd law below to essentially half number
    // of calculations needs some care in a parallel version.
    // A naive decomposition on the i loop can lead to a race condition
    // because you are assigning to ax[j], etc.
    // You can remove these assignments and extend the j loop to a fixed
    // upper bound of N, or, for extra credit, find a cleverer solution!)
    for (int i = 1; i < N; i++) {
        for (int j = 0; j < i; j++) { // loop over all distinct pairs
            dx = x[i] - x[j];
            dx2 = dx * dx;
            if (dx2 < forceCutoff2) { // make sure they're close enough to bother
                dy = y[i] - y[j];
                dy2 = dy * dy;
                if (dy2 < forceCutoff2) {
                    rSquared = dx2 + dy2;
                    if (rSquared < forceCutoff2) {
                        rSquaredInv = 1.0f / rSquared;
                        attract = rSquaredInv * rSquaredInv * rSquaredInv;
                        repel = attract * attract;
                        fOverR = 24.0f * ((2.0f * repel) - attract) * rSquaredInv;
                        fx = fOverR * dx;
                        fy = fOverR * dy;
                        ax[i] += fx;  // add this force on to i's acceleration (mass = 1)
                        ay[i] += fy;

                        ax[j] -= fx;  // Newton's 3rd law
                        ay[j] -= fy;
                    }
                }
            }
        }
    }
}

static class Display extends JPanel {

static final float SCALE = WINDOW_SIZE / BOX_WIDTH;

static final int DIAMETER = Math.max((int) (SCALE * 2 * ATOM_RADIUS), 2);

Display() {

    setPreferredSize(new Dimension(WINDOW_SIZE, WINDOW_SIZE));

    JFrame frame = new JFrame("MD");
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.setContentPane(this);
    frame.pack();
    frame.setVisible(true);
}

public void paintComponent(Graphics g) {
    g.setColor(Color.WHITE);
    g.fillRect(0, 0, WINDOW_SIZE, WINDOW_SIZE);
    g.setColor(Color.BLUE);
    for (int i = 0; i < N; i++) {
        g.fillOval((int) (SCALE * (x[i] - ATOM_RADIUS)),
                WINDOW_SIZE - 1 - (int) (SCALE * (y[i] + ATOM_RADIUS)),
                DIAMETER, DIAMETER);
    }
}
}

// Returns first NVIDIA or AMD GPU
static OpenCLDevice getDevice() {
    OpenCLDevice d = OpenCLDevice.select((OpenCLDevice l, OpenCLDevice r) -> {
        if (deviceFromVendors(l)) {
            return l;
        } else if (deviceFromVendors(r)) {
            return r;
        } else {
            return null;
        }
    });
    System.out.println("Found GPU: " + d.getOpenCLPlatform().getName());
    return d;
}

public static boolean deviceFromVendors(OpenCLDevice d) {
    String[] vendors = {"NVIDIA", "AMD"};

    String deviceVendor = d.getOpenCLPlatform().getVendor();
    for (String vendor : vendors) {
        if (deviceVendor.contains(vendor)) {
            return true;
        }
    }
    return false;
}

}
